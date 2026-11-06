# Per-region evaluation (challenges)

Scoring a reconstruction on chosen regions of the object (surface properties, excluded zones) instead of the
whole surface, and adding your own region.

[← back to README](../README.md)

This pipeline evaluates **geometry**: Chamfer distance, precision/recall/F-score.

**Region-restricted evaluation (properties).** By supplying per-point property masks on the
ground truth (`challenges/‹property›.npy`: lambertian, specular, …), the same metrics are
computed **per zone** instead of over the whole surface. A property counts for an object **if and
only if its `.npy` exists**; a property covering the whole object ships as an all-ones mask. GT
points in `challenges/excluded.npy` (unreliable GT zones) are never evaluated, in any zone. You can therefore score reconstruction quality on precise parts of an
object (a labelled region, thin structures, cavities, …) rather than as a single global number.

**Adding your own challenge.** A *challenge* is a named region of an object that is scored
separately (e.g. `lambertian`, or any name you pick; `excluded` is reserved for the
exclusion mask). The name is **chosen by you and
taken from the file name**: the pipeline discovers challenges from files, so there is nothing to
list by hand. To add one:

1. In your config `.yaml`, point `paths.gt_root` at the read-only folder holding each object's raw
   ground truth; challenge sources live in its `challenges_raw/` sub-folder (see
   [docs/configuration.md](configuration.md) and [docs/input-layout.md](input-layout.md)).
2. Drop a source named after the challenge into `challenges_raw/`, either
   - a copy of the GT mesh with the target region painted **red**, `‹name›.ply`, or
   - per-view binary masks `‹viewId›_‹name›.png`.
3. Run `preprocess-challenges` → it writes `challenges/‹name›.npy`. From there `apply-masks` and the
   aggregation pick every `challenges/*.npy` up **automatically** and report each method per
   challenge under `‹name›`, with no code change.

**Not yet implemented.** Normal accuracy (the mean absolute error, MAE, on surface normals) is
planned but not part of the current metrics.
