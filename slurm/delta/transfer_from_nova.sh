#!/usr/bin/env bash
# Push what NCSA Delta needs from Nova. Run this ON NOVA (login node is fine --
# it only reads, stats and rsyncs; it never touches Slurm and never writes here).
#
#   bash slurm/delta/transfer_from_nova.sh <subcommand>
#
# Subcommands -- sizes, readiness and the order to run them in are in
# slurm/delta/MANIFEST.md:
#   status          what is ready on Nova right now. Sends nothing.
#   code            the repo working tree, uncommitted edits included (tracked
#                   files + new files under configs/ slurm/ eval/)
#   ckpts-sorghum   matched-cut checkpoint + config.json for every sorghum run
#   ckpts-maize     same for maize. A checkpoint not written yet is SKIPPED, not
#                   an error -- re-run after the Nova runs finish; rsync skips
#                   whatever already arrived.
#   caches          outputs/_probe_cache{,_maize}/*.npz (probe features) -> land in
#                   outputs/_probe_cache{,_maize}_nova/ on Delta, never the default dir
#   data-sorghum    Sorghum_15K: the 4 files the loader reads + the 2 root CSVs
#   data-maize      Maize: the 4 files the loader reads + plant_scores.csv
#   all             code, ckpts-sorghum, caches, data-sorghum, data-maize, ckpts-maize
#   globus-list [ckpts|data-sorghum|data-maize]
#                   print a `globus transfer --batch` file on stdout. Sends nothing.
#   pull-results    the REVERSE direction, Delta -> Nova: the control arm's
#                   config.json + training_history.json + epoch-600 checkpoint into
#                   outputs/<PULL_RUN>/ (default e2_pcrgbdt_tg), the Delta probe
#                   tables (reports/probe_p{r,m}_*.csv, reports/e9_p{r,m}_*/ -- the
#                   pr_/pm_ job names submit.sh gives) into reports/delta/, and
#                   logs/delta_* into logs/delta/. Paths no Nova job writes. Not
#                   part of `all`.
#
# Destination (env):
#   DELTA_USER          Delta login (optional if ~/.ssh/config supplies it)
#   DELTA_HOST          default dt-login.delta.ncsa.illinois.edu (the DTN). Set it
#                       to the EMPTY string for a local destination (testing).
#   DELTA_REPO          the repo on Delta; outputs/<run>/... land under it
#   DELTA_SORGHUM_ROOT  Sorghum_15K on Delta -- what data.data_root must say there
#   DELTA_MAIZE_ROOT    Maize on Delta
# Knobs:
#   DRY_RUN=1           rsync -n -i: print what would be sent, create nothing
#   ONLY_SAMPLE="val/Sorghum_10001_00 ..."
#                       data-* only: send just these sample folders (+ root CSVs),
#                       to prove the filter on one folder before the big copy
#   WITH_CAMERA_POSE=1  also send camera_pose.json. Nothing in training or the
#                       probe reads it; the view-regime scripts (E5) do.
#   MIN_CKPT_AGE_S=300  younger checkpoints count as possibly mid-write
#   DELTA_SSH           ssh command for rsync -e (default: ControlMaster, see below)
#
# MFA: every new ssh connection to Delta asks for Duo. The default DELTA_SSH
# multiplexes one authenticated connection (ControlPersist 2h), so a whole `all`
# costs one prompt instead of one per rsync.
#
# For the data (~382 GB sorghum + ~69 GB maize) Globus is more robust than rsync
# over an MFA'd ssh: it survives a dropped session, retries per file and checksums
# by default, where an rsync that loses its connection mid-tree has to rescan
# ~150k folders on restart. `globus-list data-sorghum` prints the batch and the
# exact command. rsync remains the tool for checkpoints and anything incremental.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NOVA_SORGHUM_ROOT="${NOVA_SORGHUM_ROOT:-/work/mech-ai-scratch/alloy/shorgum_data/new_data_50K/Sorghum_15K}"
NOVA_MAIZE_ROOT="${NOVA_MAIZE_ROOT:-/work/mech-ai-scratch/alloy/Maize}"

# `-` not `:-`: an explicitly EMPTY DELTA_HOST means "local destination".
DELTA_HOST="${DELTA_HOST-dt-login.delta.ncsa.illinois.edu}"
DELTA_USER="${DELTA_USER:-}"
DELTA_SSH="${DELTA_SSH:-ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%C -o ControlPersist=2h}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_SAMPLE="${ONLY_SAMPLE:-}"
WITH_CAMERA_POSE="${WITH_CAMERA_POSE:-0}"
MIN_CKPT_AGE_S="${MIN_CKPT_AGE_S:-300}"

# ── run -> matched-cut checkpoint. The ONE place this table lives. ───────────
# Never best_model.pth: it is selected on total val loss, which lands anywhere
# from 42 % to 100 % of schedule depending on the arm, so across arms it bills
# training length to whatever the arm varies. Never outputs/maize_4m_1000ep:
# 329,000 steps with a cosine restart, comparable to none of these.
# The E3 cuts are the last save below the final epoch -- save_freq is scaled per
# arm (257 / 88 / 26), so 6168 / 2024 / 624, not 6169 / 2100 / 631.
# e2_pcrgbdt@600 is also E3's full-data point and E4's base point.
# shellcheck disable=SC2034  # both arrays are read through a nameref (${species^^}_RUNS)
SORGHUM_RUNS=(
    e2_pc:600 e2_pcrgb:600 e2_pcrgbd:600 e2_pcrgbdt:600
    e4_small:600 e4_large:600
    e3_1k:6168 e3_3k:2024 e3_10k:624
)
# shellcheck disable=SC2034
MAIZE_RUNS=(
    maize_4m:600
    maize_e3_1k:6168 maize_e3_3k:2024 maize_e3_10k:624
    maize_e4_small:600 maize_e4_large:600
)

# ── what each loader actually opens (sorghum_dataset.py / sorghum_dataset_4m.py,
# maize_dataset_4m.py) plus the target tables the probes read. Everything else
# in a sample folder is dead weight: sorghum's .obj + _nc.ply + <id>.yml are
# ~82 % of its bytes and are never opened.
SORGHUM_ROOT_FILES=(assignment.csv features.csv)     # eval/linear_probe.py load_targets
SORGHUM_SAMPLE_FILES=(rgb.png depth.png '*_nc_cam.ply' '*_spline.yml')
# summary.json is not read by code; it is 4.5 KB and is the only record that the
# maize split is seed 0 / Mahalanobis, not sorghum's seed 42 / extremeness.
MAIZE_ROOT_FILES=(plant_scores.csv summary.json)     # eval/linear_probe_maize.py
MAIZE_SAMPLE_FILES=(rgb.png depth.png pointcloud_cam.ply 'maize_*_spline.xml')

# ─────────────────────────────── helpers ────────────────────────────────────

log() { printf '%s\n' "$*" >&2; }
die() { printf 'transfer_from_nova: %s\n' "$*" >&2; exit 1; }

TMPD="$(mktemp -d "${TMPDIR:-/tmp}/delta_xfer.XXXXXX")"
trap 'rm -rf "$TMPD"' EXIT

remote() { [[ -n "$DELTA_HOST" ]]; }

dest() {     # dest <dir> -> rsync destination spec
    if remote; then printf '%s%s:%s' "${DELTA_USER:+$DELTA_USER@}" "$DELTA_HOST" "$1"
    else printf '%s' "$1"; fi
}

need() {     # need VAR -- fail early with the variable's name, not an rsync error
    [[ -n "${!1:-}" ]] || die "$1 is not set (see the header of $0)"
}

ensure_dest_dir() {
    # Delta's rsync (RHEL 8, 3.1.x) has no --mkpath, and rsync only creates the
    # LAST missing component of a destination. A dry run creates nothing.
    [[ "$DRY_RUN" == 1 ]] && return 0
    if remote; then
        # shellcheck disable=SC2086  # DELTA_SSH is a command line, split on purpose
        $DELTA_SSH "${DELTA_USER:+$DELTA_USER@}$DELTA_HOST" mkdir -p "$(printf '%q' "$1")"
    else
        mkdir -p "$1"
    fi
}

rsync_cmd() {  # fills the global RSYNC array
    RSYNC=(rsync -a --partial-dir=.rsync-partial)
    if [[ "$DRY_RUN" == 1 ]]; then RSYNC+=(-n --itemize-changes)
    else RSYNC+=(--info=progress2 --human-readable); fi
    if remote; then RSYNC+=(-e "$DELTA_SSH"); fi
}

latest_ckpt() {  # highest checkpoint_epoch_N.pth in a run, by N (not mtime)
    local best=-1 f n
    for f in "$1"/checkpoints/checkpoint_epoch_*.pth; do
        [[ -e "$f" ]] || continue
        n="${f##*_}"; n="${n%.pth}"
        [[ "$n" =~ ^[0-9]+$ ]] && (( n > best )) && best=$n
    done
    if (( best < 0 )); then echo none; else echo "checkpoint_epoch_${best}.pth"; fi
}

ckpt_complete() {
    # torch.save writes straight to the final name, so a checkpoint of a live run
    # can be caught mid-write. It is a zip, and a truncated zip has no end-of-
    # central-directory record: opening it reads only the tail, so this is cheap
    # even for the 4 GB large model. The age check covers the seconds in which
    # the tail exists but the OS has not flushed everything behind it.
    local f="$1" age
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    (( age >= MIN_CKPT_AGE_S )) || { echo "written ${age}s ago (< ${MIN_CKPT_AGE_S}s)"; return 1; }
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).close()' "$f" \
            2>/dev/null || { echo "not a complete zip (truncated?)"; return 1; }
    fi
    return 0
}

gb() { awk -v b="$1" 'BEGIN { printf "%.2f", b / 1e9 }'; }

# ckpt_list <species> <strict 0|1> <out-list>
# Writes repo-relative paths (config.json + checkpoint) for every ready run.
# strict=1 (sorghum -- all finished): a missing checkpoint is fatal.
# strict=0 (maize -- still training): it is reported and skipped.
ckpt_list() {
    local species="$1" strict="$2" out="$3" entry run ep rel dir why
    local -n runs="${species^^}_RUNS"
    : > "$out"
    for entry in "${runs[@]}"; do
        run="${entry%%:*}"; ep="${entry##*:}"
        rel="outputs/$run/checkpoints/checkpoint_epoch_${ep}.pth"
        dir="$REPO/outputs/$run"
        case "$rel" in *best_model*|*maize_4m_1000ep*)
            die "refusing $rel -- not a matched cut (see the table comment)";; esac
        if [[ ! -f "$dir/config.json" ]]; then
            (( strict )) && die "$run: $dir/config.json missing"
            log "  SKIP $run: no config.json yet"; continue
        fi
        if [[ ! -f "$REPO/$rel" ]]; then
            (( strict )) && die "$run: $rel missing"
            log "  SKIP $run: epoch $ep not written yet (latest: $(latest_ckpt "$dir"))"
            continue
        fi
        if ! why="$(ckpt_complete "$REPO/$rel")"; then
            (( strict )) && die "$run: $rel $why"
            log "  SKIP $run: $rel $why"; continue
        fi
        # probes rebuild the model from config.json (active_modalities, model_size)
        printf '%s\n%s\n' "outputs/$run/config.json" "$rel" >> "$out"
    done
}

# ───────────────────────────── subcommands ──────────────────────────────────

cmd_status() {
    local species entry run ep dir f cfg state size
    for species in sorghum maize; do
        local -n runs="${species^^}_RUNS"
        printf '\n%-16s %-30s %-7s %9s %8s  %s\n' "run" "checkpoint" "state" "GB" "config" "latest on disk"
        for entry in "${runs[@]}"; do
            run="${entry%%:*}"; ep="${entry##*:}"; dir="$REPO/outputs/$run"
            f="$dir/checkpoints/checkpoint_epoch_${ep}.pth"
            cfg=$([[ -f "$dir/config.json" ]] && stat -c %s "$dir/config.json" || echo -)
            if [[ -f "$f" ]] && ckpt_complete "$f" >/dev/null; then
                state=ready; size=$(gb "$(stat -c %s "$f")")
            elif [[ -f "$f" ]]; then state=writing; size=-
            else state=pending; size=-; fi
            printf '%-16s %-30s %-7s %9s %8s  %s\n' "$run" "checkpoint_epoch_${ep}.pth" \
                "$state" "$size" "$cfg" "$(latest_ckpt "$dir")"
        done
    done
    printf '\nprobe caches:\n'
    for d in _probe_cache _probe_cache_maize; do
        if [[ -d "$REPO/outputs/$d" ]]; then
            printf '  outputs/%s: ' "$d"
            find "$REPO/outputs/$d" -maxdepth 1 -name '*.npz' ! -name '*.tmp*' -printf '%s\n' \
                | awk '{n++; s+=$1} END {printf "%d files, %.3f GB\n", n, s/1e9}'
        else
            printf '  outputs/%s: absent\n' "$d"
        fi
    done
}

cmd_code() {
    need DELTA_REPO
    local list="$TMPD/code.list"
    # Tracked files as they are in the working tree (uncommitted edits ride
    # along), plus new, non-ignored files in the code directories. Ignored files
    # -- slurm/delta/delta_env.sh, outputs/, logs/ -- never match.
    git -C "$REPO" ls-files -z > "$list"
    git -C "$REPO" ls-files -z -o --exclude-standard -- configs slurm eval >> "$list"
    log "code: $(tr -cd '\0' < "$list" | wc -c) files -> $(dest "$DELTA_REPO")"
    rsync_cmd; ensure_dest_dir "$DELTA_REPO"
    "${RSYNC[@]}" --relative --from0 --files-from="$list" --ignore-missing-args \
        --exclude=slurm/delta/delta_env.sh \
        "$REPO/" "$(dest "$DELTA_REPO")/"
}

cmd_ckpts() {
    local species="$1" strict=1 list="$TMPD/ckpts_$1.list"
    need DELTA_REPO
    [[ "$species" == maize ]] && strict=0
    log "ckpts-$species:"
    ckpt_list "$species" "$strict" "$list"
    if [[ ! -s "$list" ]]; then log "  nothing ready to send"; return 0; fi
    log "  sending $(( $(wc -l < "$list") / 2 )) run(s) -> $(dest "$DELTA_REPO")/outputs/"
    rsync_cmd; ensure_dest_dir "$DELTA_REPO"
    # --relative from the repo root: files land at outputs/<run>/... under DELTA_REPO,
    # which is where the probes look (REPO/outputs/<run>). One rsync, one ssh session.
    "${RSYNC[@]}" --relative --files-from="$list" "$REPO/" "$(dest "$DELTA_REPO")/"
}

cmd_caches() {
    need DELTA_REPO
    local list d f n=0
    # The cache key is <run>__<ckpt stem>__<split>__..., so only files whose stem
    # matches the --ckpt a probe is given are ever hit. Every sorghum file here
    # today is keyed best_model -- and eval/linear_probe.py DEFAULTS --ckpt to
    # best_model.pth, so in Delta's default cache dir a hand-run probe would
    # silently mix Nova-extracted features (other num_workers, other point
    # subsamples, an unmatched cut) into a Delta table. They land in <dir>_nova
    # instead, which no default path reads: to reproduce the old E2 numbers pass
    # --cache-dir outputs/_probe_cache_nova --ckpt best_model.pth explicitly.
    for d in _probe_cache _probe_cache_maize; do
        if [[ ! -d "$REPO/outputs/$d" ]]; then log "  outputs/$d absent -- nothing to send"; continue; fi
        list="$TMPD/caches$d.list"; : > "$list"
        for f in "$REPO/outputs/$d"/*.npz; do
            [[ -e "$f" ]] || continue
            [[ "$f" == *.tmp* ]] && continue      # an extraction still writing
            printf '%s\n' "${f##*/}" >> "$list"
        done
        [[ -s "$list" ]] || continue
        n=$(( n + $(wc -l < "$list") ))
        log "caches: outputs/$d -> outputs/${d}_nova ($(wc -l < "$list") .npz)"
        rsync_cmd; ensure_dest_dir "$DELTA_REPO/outputs/${d}_nova"
        "${RSYNC[@]}" --files-from="$list" "$REPO/outputs/$d/" "$(dest "$DELTA_REPO/outputs/${d}_nova")/"
    done
    log "caches: $n .npz files. NOT sent: .dataset_index_cache/ --"
    log "  its file name is sha256(absolute split path) and it is invalidated by the split"
    log "  dir's mtime_ns, so no Nova entry can hit on Delta. Delta pays the scan once."
}

cmd_pull_results() {
    # Delta -> Nova. The rest of the programme's analysis lives here (e.g.
    # eval/analyze_e2.py reads outputs/<run>/training_history.json), so the
    # control arm and the probe tables come back. Only the matched cut, not all
    # 24 checkpoints; nothing lands where a Nova job writes.
    need DELTA_REPO
    local run="${PULL_RUN:-e2_pcrgbdt_tg}" list="$TMPD/pull.list"
    printf '%s\n' "outputs/$run/config.json" "outputs/$run/training_history.json" \
        "outputs/$run/checkpoints/checkpoint_epoch_600.pth" > "$list"
    rsync_cmd
    if [[ "$DRY_RUN" != 1 ]]; then mkdir -p "$REPO/reports/delta" "$REPO/logs/delta"; fi
    log "pull-results: $(dest "$DELTA_REPO") -> $REPO"
    # --ignore-missing-args: the checkpoint does not exist until epoch 600.
    "${RSYNC[@]}" --relative --ignore-missing-args --files-from="$list" \
        "$(dest "$DELTA_REPO")/" "$REPO/"
    "${RSYNC[@]}" --include='probe_p[rm]_*.csv' --include='e9_p[rm]_*/' --include='e9_p[rm]_*/**' --exclude='*' \
        "$(dest "$DELTA_REPO/reports")/" "$REPO/reports/delta/"
    "${RSYNC[@]}" --include='delta_*' --exclude='*' \
        "$(dest "$DELTA_REPO/logs")/" "$REPO/logs/delta/"
}

cmd_data() {
    local species="$1" src dst_var p s
    local -a root_files sample_files filt srcs
    if [[ "$species" == sorghum ]]; then
        src="$NOVA_SORGHUM_ROOT"; dst_var=DELTA_SORGHUM_ROOT
        root_files=("${SORGHUM_ROOT_FILES[@]}"); sample_files=("${SORGHUM_SAMPLE_FILES[@]}")
    else
        src="$NOVA_MAIZE_ROOT"; dst_var=DELTA_MAIZE_ROOT
        root_files=("${MAIZE_ROOT_FILES[@]}"); sample_files=("${MAIZE_SAMPLE_FILES[@]}")
    fi
    need "$dst_var"
    [[ -d "$src/train" && -d "$src/val" && -d "$src/test" ]] || die "$src has no train/val/test"

    # Include-filter, first match wins. Root files are anchored ('/x') so a
    # same-named file deeper down cannot sneak in; '*/' keeps the recursion alive;
    # the final '*' drops everything not named -- including maize's per-split
    # _params.json and sorghum's .obj / _nc.ply / <id>.yml duplicates.
    filt=()
    for p in "${root_files[@]}"; do filt+=("--include=/$p"); done
    filt+=("--include=*/")
    for p in "${sample_files[@]}"; do filt+=("--include=$p"); done
    [[ "$WITH_CAMERA_POSE" == 1 ]] && filt+=("--include=camera_pose.json")
    filt+=("--exclude=*")

    if [[ -n "$ONLY_SAMPLE" ]]; then
        # Test hook: name the folders, never list a split (105k entries on Lustre).
        # '/./' marks where the relative path starts, so they land at <split>/<folder>.
        srcs=()
        for p in "${root_files[@]}"; do srcs+=("$src/./$p"); done
        for s in $ONLY_SAMPLE; do
            [[ -d "$src/$s" ]] || die "ONLY_SAMPLE: $src/$s is not a directory"
            srcs+=("$src/./$s")
        done
        log "data-$species: ONLY_SAMPLE=$ONLY_SAMPLE -> $(dest "${!dst_var}")"
        rsync_cmd; ensure_dest_dir "${!dst_var}"
        "${RSYNC[@]}" --relative "${filt[@]}" "${srcs[@]}" "$(dest "${!dst_var}")/"
        return
    fi

    log "data-$species: $src -> $(dest "${!dst_var}")"
    log "  150,000 sample folders: the file-list scan alone takes a while on Lustre"
    rsync_cmd; ensure_dest_dir "${!dst_var}"
    "${RSYNC[@]}" "${filt[@]}" "$src/" "$(dest "${!dst_var}")/"
}

cmd_globus_list() {
    local what="${1:-ckpts}" src_repo dst_repo src dst p list species excl
    local -a root_files
    # Absolute POSIX paths on both sides. If a collection is rooted somewhere
    # other than / (check `globus ls <ep>:/`), set these to the collection view.
    src_repo="${GLOBUS_SRC_REPO:-$REPO}"
    dst_repo="${GLOBUS_DST_REPO:-${DELTA_REPO:-/DELTA_REPO}}"
    case "$what" in
    ckpts)
        cat <<EOF
# Checkpoints + config.json at the matched cut. Generated $(date '+%F %T %Z') on $(hostname -s).
#   globus transfer "\$NOVA_EP" "\$DELTA_EP" --batch THIS_FILE \\
#       --label embodiedmae-ckpts --sync-level size
# Maize runs still training are listed as comments; regenerate after they finish.
EOF
        list="$TMPD/g.list"
        for species in sorghum maize; do
            ckpt_list "$species" 0 "$list" 2>&1 | sed 's/^/# /'
            while IFS= read -r p; do
                printf '"%s" "%s"\n' "$src_repo/$p" "$dst_repo/$p"
            done < "$list"
        done
        ;;
    data-sorghum|data-maize)
        species="${what#data-}"
        if [[ "$species" == sorghum ]]; then
            src="${GLOBUS_SRC_ROOT:-$NOVA_SORGHUM_ROOT}"; dst="${GLOBUS_DST_ROOT:-${DELTA_SORGHUM_ROOT:-/DELTA_SORGHUM_ROOT}}"
            # Exclude-list, not rsync's include-list: Globus filter rules can match
            # directory names too, so a set ending in a catch-all exclude risks
            # pruning the sample folders themselves, while an exclude-list can
            # only ever send too much. '*_nc.ply' does not match '*_nc_cam.ply';
            # 'Sorghum_*[0-9].yml' does not match '*_spline.yml'.
            excl="--exclude '*.obj' --exclude '*_nc.ply' --exclude 'Sorghum_*[0-9].yml'"
            root_files=("${SORGHUM_ROOT_FILES[@]}")
        else
            src="${GLOBUS_SRC_ROOT:-$NOVA_MAIZE_ROOT}"; dst="${GLOBUS_DST_ROOT:-${DELTA_MAIZE_ROOT:-/DELTA_MAIZE_ROOT}}"
            excl="--exclude '_params.json'"
            root_files=("${MAIZE_ROOT_FILES[@]}")
        fi
        [[ "$WITH_CAMERA_POSE" == 1 ]] || excl+=" --exclude 'camera_pose.json'"
        cat <<EOF
# $species data for Delta. Generated $(date '+%F %T %Z') on $(hostname -s).
#   globus transfer "\$NOVA_EP" "\$DELTA_EP" --batch THIS_FILE \\
#       --label embodiedmae-$species --sync-level mtime \\
#       $excl
# Filter flags are per task, not per line, which is why they are on the command.
# Check them against \`globus transfer --help\` for your CLI version.
EOF
        for p in "${root_files[@]}"; do printf '"%s" "%s"\n' "$src/$p" "$dst/$p"; done
        for p in train val test; do printf -- '--recursive "%s" "%s"\n' "$src/$p" "$dst/$p"; done
        ;;
    *) die "globus-list: unknown '$what' (ckpts | data-sorghum | data-maize)";;
    esac
}

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }

case "${1:-}" in
    status)         cmd_status ;;
    code)           cmd_code ;;
    ckpts-sorghum)  cmd_ckpts sorghum ;;
    ckpts-maize)    cmd_ckpts maize ;;
    caches)         cmd_caches ;;
    data-sorghum)   cmd_data sorghum ;;
    data-maize)     cmd_data maize ;;
    all)            cmd_code; cmd_ckpts sorghum; cmd_caches
                    cmd_data sorghum; cmd_data maize; cmd_ckpts maize ;;
    globus-list)    shift; cmd_globus_list "${1:-ckpts}" ;;
    pull-results)   cmd_pull_results ;;
    -h|--help|help) usage 0 ;;
    *)              usage 1 ;;
esac
