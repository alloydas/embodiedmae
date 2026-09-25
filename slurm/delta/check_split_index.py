"""Refuse to start on a split whose loader index is short.

    python slurm/delta/check_split_index.py sorghum "$SORGHUM_ROOT"
    python slurm/delta/check_split_index.py maize   "$MAIZE_ROOT"

The batch scripts' `ls -f | grep -c` guard counts FOLDERS. A transfer creates a
folder before it fills it, and the loaders skip a folder missing any of its four
files with a one-line warning, then cache that index keyed on the split
directory's mtime -- which files landing later inside existing folders never
change. So one job started mid-transfer caches a short index and every later job
gets an "index cache hit" on it: fewer steps per epoch under view_sampling, a
probe scored on a plant subset, and a deterministic view that is silently
views[1] where _00 was missing. Nothing errors.

This builds (or reads) the index exactly as the trainer and the probes will --
same class, same root, same cache -- and checks it: every folder indexed, every
plant with all ten views, view _00 first. A short CACHED index is rebuilt once,
because the transfer may have finished since it was written; a short FRESH scan
means the copy is incomplete. The scan it pays on a cold cache (~24 min for
sorghum train on Nova) is the one the job would pay anyway, done by one process
instead of by every DDP rank at once.

Exit 0 = ok, 3 = refuse (the batch scripts' "incomplete data" code).
"""

# Repo root on sys.path: this script lives in slurm/delta/ but imports the
# top-level dataset modules.
import sys as _sys
import pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import os
import sys
from pathlib import Path

# 70/15/15 by plant, ten views each: identical for both species.
WANT = {'train': 105000, 'val': 22500, 'test': 22500}
VIEWS = 10


def _species(name):
    """(build(root, split), is_cached(split_dir), rebuild env var) per species."""
    if name == 'sorghum':
        from sorghum_dataset import _read_index_cache
        from sorghum_dataset_4m import SorghumDataset4M

        def build(root, split):
            return SorghumDataset4M(root, split=split, view_sampling=True)

        def cached(d):
            # SorghumDataset4M reads two caches; the spline one filters the base.
            return (_read_index_cache(d, 'base') is not None
                    and _read_index_cache(d, 'spline') is not None)
        return build, cached, 'SORGHUM_INDEX_REBUILD'

    if name == 'maize':
        import maize_dataset_4m as M

        def build(root, split):
            return M.MaizeDataset4M(root, split=split, view_sampling=True)

        def cached(d):
            return M._read_index_cache(d, 'base') is not None
        return build, cached, 'MAIZE_INDEX_REBUILD'

    sys.exit(f"unknown species '{name}' (sorghum | maize)")


def problems(ds, want):
    """Everything wrong with one split's index, as strings; [] when complete."""
    out = []
    if len(ds.samples) != want:
        out.append(f"{len(ds.samples)} samples indexed, expected {want}")
    if len(ds.plant_views) != want // VIEWS:
        out.append(f"{len(ds.plant_views)} plants, expected {want // VIEWS}")
    short = [pid for pid, v in zip(ds.plant_ids, ds.plant_views) if len(v) != VIEWS]
    if short:
        out.append(f"{len(short)} plants without {VIEWS} views, e.g. {', '.join(short[:5])}")
    # deterministic_view (val/test, and every probe) takes views[0]; it must be _00.
    not00 = [pid for pid, v in zip(ds.plant_ids, ds.plant_views)
             if v and not ds.samples[v[0]].name.endswith('_00')]
    if not00:
        out.append(f"{len(not00)} plants whose first view is not _00, e.g. {', '.join(not00[:5])}")
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[0] + "\nusage: check_split_index.py sorghum|maize <root>")
    build, cached, rebuild_env = _species(sys.argv[1])
    root = Path(sys.argv[2])

    failed = False
    for split, want in WANT.items():
        try:
            hit = cached(root / split)
        except OSError:
            hit = False
        try:
            ds = build(root, split)
            bad = problems(ds, want)
        except (ValueError, OSError) as e:          # e.g. "No valid samples found"
            bad = [f"{type(e).__name__}: {e}"]
        if bad and hit:
            print(f"index {split}: cached index is short ({'; '.join(bad)}) -- "
                  f"rebuilding once, the transfer may have finished since", flush=True)
            os.environ[rebuild_env] = '1'
            try:
                ds = build(root, split)
                bad = problems(ds, want)
            except (ValueError, OSError) as e:
                bad = [f"{type(e).__name__}: {e}"]
            finally:
                del os.environ[rebuild_env]
            hit = False
        if bad:
            failed = True
            print(f"REFUSING: {root / split}: " + '; '.join(bad))
        else:
            print(f"index {split}: {want} samples, {want // VIEWS} plants x {VIEWS} views ok"
                  + (" (cache hit)" if hit else " (scanned, cache written)"))

    if failed:
        print("The copy is incomplete: some folders exist but lack files. Let the transfer\n"
              "finish (Globus task SUCCEEDED, or rsync exit 0), then resubmit -- the short\n"
              "cached index is rebuilt automatically on the next job.")
        sys.exit(3)


if __name__ == '__main__':
    main()
