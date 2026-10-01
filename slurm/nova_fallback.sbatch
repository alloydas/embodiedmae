#!/bin/bash
#SBATCH --job-name=nova_fallback
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:45:00
# Never re-run: a watcher killed between its hold and its rewire would, re-run
# from the top, see NOVA JobHeldUser and leave it held for good.
#SBATCH --no-requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Fallback for a training chain whose next link is a `nova` job. If that job is
# still PENDING GRACE seconds after the chunk it was queued behind has ended
# (the mech-ai account's 17-GPU cap is full, or the two RTX PRO 6000 nodes have
# no room), move the run back onto its scavenger chunks instead of letting the
# whole chain idle behind a job that cannot start.
#
#   sbatch --job-name=fb_<run> --dependency=afterany:<PREV> \
#          slurm/nova_fallback.sbatch <NOVA> <PREV> <NEXT> <TAIL> [<PROBE>]
#
#   NOVA   the nova job queued behind PREV
#   PREV   the scavenger chunk NOVA waits for (this job starts once it has ended)
#   NEXT   the scavenger chunk that waits for NOVA
#   TAIL   the expected last scavenger chunk. The chain is re-walked (pending jobs
#          with NEXT's job name, each waiting on the one before) and the real last
#          link is used; a mismatch with TAIL is logged.
#   PROBE  optional: a pending probe whose dependency list names NOVA
#
#   before:  PREV -> NOVA -> NEXT -> ... -> TAIL        PROBE after NOVA
#   after:   PREV -> NEXT -> ... -> TAIL -> NOVA        PROBE after TAIL
#
# After it fires, NOVA is the chain's LAST link: extend the chain with
# afterany:<NOVA>, never afterany:<TAIL>, or NOVA and the extension would both
# start when TAIL ends.
#
# The one thing this must never do is let two jobs of one run be runnable at
# the same time (two trainers in one output dir, and nothing locks it).
#   1. hold NOVA and confirm it is held and pending. A held job cannot start.
#   2. re-read NEXT and the chain AFTER the hold; anything unexpected -> stop.
#   3. clear NEXT's dependency (PREV has ended, so this is afterany:PREV) and
#      read it back. This comes before step 4 because pointing NOVA at TAIL
#      while NEXT still waits on NOVA is a cycle, which scontrol rejects.
#   4. point NOVA at the real tail, read it back, check the tail has not ended,
#      and only then release the hold. NOVA is kept, not cancelled: at the tail
#      it resumes from the newest checkpoint, so it exits at once if the chain
#      finished and carries on if the chain ran dry.
#   5. swap NOVA for the tail in PROBE's dependency list. Trade-off (review
#      2026-09-30): keeping NOVA in it would stall the probe behind a nova job
#      that may pend on AssocGrpGRES only to exit at once. But if the chain runs
#      dry before epoch 600, NOVA carries on while the probe starts when the tail
#      ends and fails on the missing checkpoint: rerun it by hand after NOVA.
# Failure policy: before NEXT is touched, a problem releases NOVA only if NEXT
# still waits on it (the original, safe chain); after NEXT is touched, NOVA
# stays HELD and the script exits 1. A held job shows as JobHeldUser in squeue
# and costs an idle chain until someone looks -- never a double run.

set -uo pipefail

NOVA="${1:?usage: nova_fallback.sbatch NOVA PREV NEXT TAIL [PROBE]}"
PREV="${2:?usage: nova_fallback.sbatch NOVA PREV NEXT TAIL [PROBE]}"
NEXT="${3:?usage: nova_fallback.sbatch NOVA PREV NEXT TAIL [PROBE]}"
TAIL="${4:?usage: nova_fallback.sbatch NOVA PREV NEXT TAIL [PROBE]}"
PROBE="${5:-}"
GRACE="${GRACE:-900}"

log() { echo "$(date '+%F %T')  $*"; }

# q ID FORMAT: one squeue field. Retries transient controller errors, prints
# "" for a job squeue no longer lists (finished or purged) and "?" if squeue
# keeps failing -- so an error is never mistaken for a finished job.
q() {
  local out i
  for i in 1 2 3 4 5 6; do
    if out=$(squeue -h -j "$1" -o "$2" 2>&1); then echo "${out}" | head -1; return 0; fi
    case "${out}" in *"Invalid job id"*) echo ""; return 0 ;; esac
    sleep 10
  done
  echo "?"
}
state()  { q "$1" %T; }
reason() { q "$1" %r; }
name()   { q "$1" %j; }
# Dependency list of a job as scontrol prints it, "(null)" when it has none;
# "?" if the job cannot be read.
deps() {
  local out i
  for i in 1 2 3 4 5 6; do
    if out=$(scontrol show job "$1" 2>/dev/null); then
      grep -oP '\bDependency=\K\S+' <<<"${out}" | head -1; return 0
    fi
    sleep 10
  done
  echo "?"
}
# names ID DEPSTRING: does the dependency string name this job id as a whole token?
names() { local re="(^|[:,])$1([(:,]|\$)"; [[ "$2" =~ $re ]]; }
show() {
  squeue -h -j "${NOVA},${PREV},${NEXT},${TAIL}${PROBE:+,${PROBE}}" \
         -o '%.10i %.16j %.10P %.8T %.18r %E' 2>/dev/null
}
# Before NEXT is modified: put NOVA back only if the chain is still the original.
restore_or_hold() {
  local nd; nd="$(deps "${NEXT}")"
  if [ "$(state "${NEXT}")" = PENDING ] && names "${NOVA}" "${nd}"; then
    scontrol release "${NOVA}" && log "released NOVA ${NOVA}: chain unchanged (NEXT still waits on it)"
  else
    log "!! NEXT ${NEXT} no longer waits on NOVA (state $(state "${NEXT}"), Dependency=${nd})." \
        "NOVA ${NOVA} STAYS HELD so the two cannot both run; sort the chain out by hand."
  fi
  exit 1
}

log "nova ${NOVA}  prev ${PREV}  next ${NEXT}  tail ${TAIL}  probe ${PROBE:-none}  grace ${GRACE}s"
show

# PREV must be over. This job runs afterany:PREV, but a preempted PREV is
# requeued (pending again) and must not be rewired around.
for _ in $(seq 1 20); do [ "$(state "${PREV}")" = COMPLETING ] && sleep 30 || break; done
s="$(state "${PREV}")"
case "${s}" in
  "") ;;
  *) log "PREV ${PREV} is ${s}, not finished: not touching the chain"; exit 0 ;;
esac

# The chain must still be the one described above.
nd="$(deps "${NEXT}")"
if [ "${nd}" = "?" ]; then log "cannot read NEXT ${NEXT}: not touching the chain"; exit 1; fi
if ! names "${NOVA}" "${nd}"; then
  # Someone rewired it already. Make sure that did not leave NOVA free to start
  # next to NEXT: a pending NOVA with no dependency would do exactly that.
  nvd="$(deps "${NOVA}")"
  if [ "$(state "${NOVA}")" = PENDING ] && [ "${nvd}" = "(null)" ]; then
    scontrol hold "${NOVA}"
    log "!! NEXT ${NEXT} does not wait on NOVA (Dependency=${nd}) and NOVA had no dependency:" \
        "NOVA ${NOVA} is now HELD so it cannot run alongside the chain. Check by hand."
    exit 1
  fi
  log "NEXT ${NEXT} does not wait on NOVA (Dependency=${nd:-none}); NOVA Dependency=${nvd}: already rewired, nothing to do"
  exit 0
fi

end=$(( $(date +%s) + GRACE ))
while :; do
  s="$(state "${NOVA}")"
  if [ "${s}" != PENDING ] && [ "${s}" != "?" ]; then
    log "NOVA ${NOVA} is ${s:-gone from the queue}: nothing to do"; exit 0
  fi
  [ "$(date +%s)" -ge "${end}" ] && break
  sleep 60
done
r="$(reason "${NOVA}")"
log "NOVA ${NOVA} still ${s} after ${GRACE}s, reason ${r}"
case "${r}" in
  Dependency|DependencyNeverSatisfied|BeginTime|JobHeldUser|JobHeldAdmin|"?")
    log "reason ${r} is not a wait for capacity: leaving the chain alone"; exit 0 ;;
esac

# 1. hold NOVA, and confirm it is held and still pending
scontrol hold "${NOVA}" || log "scontrol hold ${NOVA} returned an error; checking its state"
s="$(state "${NOVA}")"; r="$(reason "${NOVA}")"
if [ "${s}" = RUNNING ] || [ "${s}" = CONFIGURING ] || [ "${s}" = "" ]; then
  # It started (or ended) between the last poll and the hold. NEXT still waits
  # on it, which is right; undo any hold so a later requeue of NOVA is not held.
  scontrol release "${NOVA}" 2>/dev/null || true
  log "NOVA ${NOVA} is ${s:-gone} (it started before the hold): NEXT still follows it"; exit 0
fi
if [ "${s}" != PENDING ] || [ "${r}" != JobHeldUser ]; then
  log "NOVA ${NOVA} is ${s} / ${r} after the hold, not PENDING / JobHeldUser: stopping"
  restore_or_hold
fi
log "NOVA ${NOVA} held"

# 2. re-read NEXT and walk the real chain, now that NOVA cannot start
nd="$(deps "${NEXT}")"
if [ "$(state "${NEXT}")" != PENDING ] || ! names "${NOVA}" "${nd}"; then
  log "NEXT ${NEXT} changed during the grace window (state $(state "${NEXT}"), Dependency=${nd})"
  restore_or_hold
fi
chain_name="$(name "${NEXT}")"
pend="$(squeue -u "${USER}" -h -t PD -o '%i|%j|%E' 2>/dev/null)" || { log "cannot list pending jobs"; restore_or_hold; }
cur="${NEXT}"; hops=0
while :; do
  kids="$(awk -F'|' -v n="${chain_name}" '$2 == n {print $1 "|" $3}' <<<"${pend}" \
          | while IFS='|' read -r id dep; do names "${cur}" "${dep}" && [ "${id}" != "${NOVA}" ] && echo "${id}"; done)"
  nk="$(grep -c . <<<"${kids}")"
  [ "${nk}" -eq 0 ] && break
  if [ "${nk}" -gt 1 ]; then log "chain forks after ${cur}: $(echo ${kids})"; restore_or_hold; fi
  cur="${kids}"; hops=$((hops + 1))
  [ "${hops}" -gt 50 ] && { log "chain walk from ${NEXT} did not end"; restore_or_hold; }
done
REAL_TAIL="${cur}"
[ "${REAL_TAIL}" != "${TAIL}" ] && log "note: the chain's real last link is ${REAL_TAIL}, not the TAIL argument ${TAIL}"
if [ "${REAL_TAIL}" = "${NEXT}" ]; then
  log "NEXT ${NEXT} is the chain's only remaining link"
fi

# 3. release NEXT, and read it back
scontrol update jobid="${NEXT}" Dependency= || log "scontrol update of ${NEXT} returned an error; reading it back"
nd="$(deps "${NEXT}")"
if names "${NOVA}" "${nd}"; then
  log "NEXT ${NEXT} still waits on NOVA (Dependency=${nd}): the update did not apply"
  restore_or_hold
fi
if [ "${nd}" = "?" ]; then
  log "!! cannot read NEXT ${NEXT} back after clearing it. NOVA ${NOVA} STAYS HELD."; exit 1
fi
log "NEXT ${NEXT} released (Dependency=${nd})"

# 4. NOVA to the real tail, read back, confirm the tail has not ended, release
# singleton as well: every link carries the run's job name, so NOVA can never
# start while another job of the run is running, whatever else changes later.
if ! scontrol update jobid="${NOVA}" Dependency="afterany:${REAL_TAIL},singleton"; then
  log "!! could not point NOVA ${NOVA} at ${REAL_TAIL}. NEXT is released, so NOVA STAYS HELD;" \
      "release it by hand only once the chain has finished."
  exit 1
fi
nvd="$(deps "${NOVA}")"
ts="$(state "${REAL_TAIL}")"
if ! names "${REAL_TAIL}" "${nvd}" || { [ "${ts}" != PENDING ] && [ "${ts}" != RUNNING ]; }; then
  log "!! NOVA ${NOVA} Dependency=${nvd}, tail ${REAL_TAIL} is '${ts}'. NOVA STAYS HELD: releasing it" \
      "could let it start while the chain runs."
  exit 1
fi
if ! scontrol release "${NOVA}"; then
  log "!! release of ${NOVA} failed: it stays held behind ${REAL_TAIL} (no fallback at the tail)"
fi
log "NOVA ${NOVA} now waits on ${REAL_TAIL} (Dependency=$(deps "${NOVA}"), reason $(reason "${NOVA}")) --" \
    "it is the chain's LAST link: extend with afterany:${NOVA}"

# 5. the probe waits for the tail instead of NOVA
if [ -n "${PROBE}" ] && [ "$(state "${PROBE}")" = PENDING ]; then
  pd="$(deps "${PROBE}")"
  ids="$(grep -oE '[0-9]{5,}' <<<"${pd}" | sed "s/^${NOVA}\$/${REAL_TAIL}/" | sort -u | paste -sd: -)"
  if [ -n "${ids}" ] && scontrol update jobid="${PROBE}" Dependency="afterany:${ids}"; then
    log "PROBE ${PROBE}: ${pd} -> $(deps "${PROBE}")"
    log "   if the chain runs dry before epoch 600, NOVA trains on and PROBE fails on the" \
        "missing checkpoint: rerun PROBE by hand once NOVA is done"
  else
    log "!! PROBE ${PROBE} dependency not updated (was ${pd:-none})"
  fi
fi

log "rewired:"
show
