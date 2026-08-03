"""GT-side exclusion masks, shared by evaluate / recompute / curves.

Single source of truth for "which GT points count, and which reconstructed points
inherit that decision". Before this module the mask logic lived inline in
`evaluate.py` only — `recompute.py` and `curves.py` silently skipped it, which made
the curves on disk inconsistent with the `metrics.json` sitting next to them for
every object carrying a `challenges/excluded.npy` (21 of 27 objects).

Semantics: every mask array is boolean, indexed over the **GT point cloud**, and True
means *drop this point*. Masks combine with logical OR. The reconstruction side inherits
the decision through the nearest-neighbour index (`idx_data2gt`), exactly as
`evaluate.py` has always done.

Design note — auto-discovered masks (2026-07-30)
------------------------------------------------
`invisible.npy` (`visibility_count == 0`) used to be applied by a *separate pass* driven
by `--extra-exclude`, exposed on `recompute` and `curves` but **not** on `evaluate`. That
asymmetry cost 248 cells that never received the mask: a pass has to enumerate a corpus
that four campaigns are writing into, so it will always miss something, and its sharding
re-enumerated the corpus per shard (1473 vs 1474 cells) and lost 145 more.

It is now in `AUTO_EXCLUDE` below: **discovered from disk, applied unconditionally** by
every caller of `load_gt_exclude` — `evaluate`, `recompute`, `curves` alike. A file that
exists is a file that applies. This is deliberately *not* a flag: a flag is a thing you
forget to pass, and forgetting it is exactly the defect this replaces. Homogeneity
becomes structural — any cell that is evaluated carries the mask by construction, so the
catch-up pass has nothing left to catch up.

The applied set is still reported per cell through `metrics["exclude_masks"]`, so the
behaviour stays traceable: the file says what was done rather than what was supposed to
be done.

Polarity guard: an exclusion mask that drops more than `MAX_AUTO_DROP_FRACTION` of the GT
is refused outright. Passing the raw `visibility_count.npy` (a *count* array) would
`astype(bool)` into "drop everything seen at least once" — 91.96% of the points, the
exact inverse of the intent. `invisible.npy` exists precisely so the polarity is explicit
at rest; this guard is the belt to that suspenders, and it fails **closed** (raises).
It applies to auto-discovered masks only — the historical `excluded.npy` legitimately
drops 50.65% on `23_whisk` and must not be second-guessed here.
"""

from pathlib import Path

import numpy as np

#: Masks applied automatically whenever the file exists under `<gt_dir>/challenges/`.
#: Discovered from disk, never from a flag. See the module docstring.
AUTO_EXCLUDE: tuple[str, ...] = ("invisible.npy",)

#: An auto-discovered mask dropping more than this fraction of the GT is refused.
#: Guards against a count array being handed in as a boolean mask (inverted polarity).
MAX_AUTO_DROP_FRACTION = 0.5


def resolve_mask_path(gt_dir: Path, name: str) -> Path:
    """Resolve a mask name to a path.

    A bare name resolves under `<gt_dir>/challenges/<name>.npy`; anything containing a
    path separator, or ending in .npy, is taken as given.
    """
    p = Path(name)
    if p.suffix == ".npy" or len(p.parts) > 1:
        return p if p.is_absolute() else (gt_dir / p)
    return gt_dir / "challenges" / f"{name}.npy"


def load_gt_exclude(
    gt_dir: Path,
    n_gt: int,
    use_excluded: bool = True,
    extra_exclude: list[str] | None = None,
    verbose: bool = True,
    use_auto: bool = True,
) -> tuple[np.ndarray | None, list[str]]:
    """Build the combined boolean *exclusion* mask over GT points.

    Args:
        gt_dir: Groundtruth directory for the object.
        n_gt: Number of GT points; every mask must match this length.
        use_excluded: Load the historical `challenges/excluded.npy` (default on, so
            behaviour is unchanged for existing callers).
        extra_exclude: Additional mask names/paths, OR-ed in. True = drop.
        verbose: Print what was applied.
        use_auto: Apply `AUTO_EXCLUDE` masks found on disk (default on — this is the
            structural guarantee, see module docstring). Turn off only to *reproduce*
            an unmasked baseline for comparison; production paths never do.

    Returns:
        (exclude, applied) where `exclude` is a bool array of length n_gt (True = drop)
        or None if nothing applies, and `applied` lists the sources used.
    """
    exclude: np.ndarray | None = None
    applied: list[str] = []

    sources: list[tuple[Path, bool]] = []  # (path, is_auto)
    if use_excluded:
        sources.append((gt_dir / "challenges" / "excluded.npy", False))
    if use_auto:
        for name in AUTO_EXCLUDE:
            sources.append((gt_dir / "challenges" / name, True))
    for name in extra_exclude or []:
        sources.append((resolve_mask_path(gt_dir, name), False))

    # An auto mask and an explicitly-requested one can resolve to the same file (a caller
    # still passing `--extra-exclude invisible`). Apply each file once; OR-ing twice is
    # harmless but would list the mask twice in `exclude_masks` and make the per-cell
    # declaration lie about what happened.
    seen: set[str] = set()

    for path, is_auto in sources:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        if not path.exists():
            # An absent auto mask is normal: the object may predate the mask. An absent
            # *requested* mask is a typo or a broken deployment — fail loudly.
            if path.name == "excluded.npy" or is_auto:
                continue
            raise FileNotFoundError(f"Exclusion mask not found: {path}")
        seen.add(key)
        m = np.load(path)
        if m.dtype != bool:
            m = m.astype(bool)
        if m.shape[0] != n_gt:
            raise ValueError(
                f"Mask {path} has length {m.shape[0]} but GT has {n_gt} points. "
                "Stale mask or stale GT — refusing to guess."
            )
        if is_auto:
            frac = float(m.sum()) / max(n_gt, 1)
            if frac > MAX_AUTO_DROP_FRACTION:
                raise ValueError(
                    f"Auto exclusion mask {path} would drop {frac:.2%} of the GT "
                    f"({int(m.sum()):,} / {n_gt:,}), above the "
                    f"{MAX_AUTO_DROP_FRACTION:.0%} limit. This is the signature of an "
                    "inverted mask — e.g. a raw visibility_count array cast to bool, "
                    "which drops every *visible* point. Refusing to evaluate."
                )
        exclude = m.copy() if exclude is None else (exclude | m)
        applied.append(str(path))
        if verbose:
            tag = " [auto]" if is_auto else ""
            print(
                f"  Exclusion mask {path.name}{tag}: "
                f"{int(m.sum()):,} / {n_gt:,} GT points dropped"
            )

    if verbose and exclude is not None and len(applied) > 1:
        print(f"  Combined exclusion: {int(exclude.sum()):,} / {n_gt:,} GT points dropped")

    return exclude, applied


def apply_exclude(
    dist_data2gt: np.ndarray,
    dist_gt2data: np.ndarray,
    idx_data2gt: np.ndarray,
    exclude: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a GT exclusion mask to both distance directions.

    The reconstruction side inherits exclusion through its nearest-GT index, which is
    what `evaluate.py` has always done. Reproduces that behaviour bit-for-bit so that
    recomputing from saved `.npy` equals a full re-evaluation.
    """
    if exclude is None:
        return dist_data2gt, dist_gt2data

    gt_keep = ~exclude
    data_keep = gt_keep[idx_data2gt]
    return dist_data2gt[data_keep], dist_gt2data[gt_keep]
