"""Ground truth preprocessing: sampling, curvature, and visibility computation."""

from pathlib import Path

import numpy as np

from ..config import Config
from ..core.camera import load_cameras_auto
from ..core.mesh import load_mesh
from ..core.sampling import downsample_pcd, upsample_mesh
from ..core.visibility import compute_visibility_count


def compute_gt_curvature(mesh, gt_pcd: np.ndarray, radius: float | None = None, tube: float = 0.5,
                         max_edge: float | None = None) -> np.ndarray:
    """Compute curvature values for GT point cloud.

    VCM curvature on the vertices of the proxy of the mesh (edges refined to max_edge, src/core/proxy.py), measured
    tile by tile (tiled_curvature), then transferred to the nearest sampled points.

    Args:
        mesh: GT mesh (raw: the visibility cleaning cuts holes whose rims the VCM would see as edges).
        gt_pcd: Sampled GT point cloud, shape (N, 3).
        radius: VCM ball radius r (None = 1.0).
        tube: VCM tube thickness R.
        max_edge: Longest edge of the proxy (None = no refinement).

    Returns:
        Curvature values, shape (N,).
    """
    from ..core.proxy import tiled_curvature
    points, curvature = tiled_curvature(mesh, max_edge, 1.0 if radius is None else radius, tube, verbose=True)
    return _to_points(curvature, points, gt_pcd)


def _to_points(vertex_curvature, vertices, gt_pcd):
    # VCM needs every vertex (shared Voronoi cells), the points then take their nearest vertex's value.
    from ..core.metrics import compute_distances
    _, nearest_vertex_idx = compute_distances(gt_pcd, vertices)
    return vertex_curvature[nearest_vertex_idx].astype(np.float32)


def _raw_gt_parts(config: Config, object_name: str) -> list[Path]:
    """Raw GT mesh file(s): several parts when they exist (e.g. 25_CT_plant_p*.ply)."""
    mesh_path_default = config.get_gt_mesh_path(object_name, cleaned=False)
    parts = sorted(list(mesh_path_default.parent.glob(f"{object_name.split('_')[0]}_*_p*.ply")))
    if not parts:
        parts = [mesh_path_default] if mesh_path_default.exists() else []
    if not parts:
        raise FileNotFoundError(f"No GT meshes found for {object_name} in {mesh_path_default.parent}")
    return parts


def proxy_curvature(config: Config, object_name: str, force: bool = False):
    """Raw GT -> VCM curvature on the vertices of its proxy, tile by tile (tiled_curvature), cached; no GT point
    cloud involved.

    Creates Groundtruth/attributes/proxy_curvature_r<r>_R<R>_e<max_edge>.npz (r, R and max edge in the name, so that
    other values give other files): points (n, 3) = the proxy vertices, curvature (n,) (convex < 0).

    Returns:
        (proxy vertices, vertex curvature).
    """
    import trimesh

    from ..core.proxy import tiled_curvature
    ev = config.evaluation
    radius = 1.0 if ev.curvature_radius is None else ev.curvature_radius
    tag = f"e{ev.curvature_max_edge:g}" if ev.curvature_max_edge is not None else "gt"
    curv_path = config.get_gt_dir(object_name) / "attributes" / f"proxy_curvature_r{radius:g}_R{ev.curvature_tube:g}_{tag}.npz"
    print(f"Proxy curvature for {object_name} (r={radius:g}, R={ev.curvature_tube:g}, max_edge={ev.curvature_max_edge})")
    if curv_path.exists() and not force:
        print(f"  Using cached {curv_path.name}")
        cached = np.load(curv_path)
        return cached["points"], cached["curvature"]
    # raw parts, not gt_cleaned.ply: the visibility cleaning cuts holes whose rims the VCM would see as edges
    mesh = trimesh.util.concatenate([load_mesh(p) for p in _raw_gt_parts(config, object_name)])
    points, curvature = tiled_curvature(mesh, ev.curvature_max_edge, radius, ev.curvature_tube, verbose=True)
    curv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(curv_path, points=points, curvature=curvature)
    print(f"  Saved {curv_path.name} ({len(points):,} proxy vertices)")
    return points, curvature


def preprocess_gt(
    config: Config,
    object_name: str,
    compute_curvature: bool = True,
    compute_visibility: bool = True,
    clean_gt: bool = True,
    force: bool = False,
) -> None:
    """Preprocess ground truth: clean mesh, sample point cloud and compute attributes.

    Creates:
        - Groundtruth/gt_cleaned.ply (cleaned GT mesh with visibility filtering)
        - Groundtruth/gt_pcd.npy (N, 3)
        - Groundtruth/attributes/curvature_values.npy (N,) float
        - Groundtruth/attributes/visibility_count.npy (N,) int

    Args:
        config: Pipeline configuration.
        object_name: Name of the object to process.
        compute_curvature: Whether to compute curvature values.
        compute_visibility: Whether to compute visibility counts.
        clean_gt: Whether to clean the GT mesh by visibility.
        force: Overwrite existing files.
    """
    gt_dir = config.get_gt_dir(object_name)
    parts = _raw_gt_parts(config, object_name)

    cleaned_mesh_path = config.get_gt_mesh_path(object_name, cleaned=True)
    gt_pcd_path = gt_dir / "gt_pcd.npy"
    attributes_dir = gt_dir / "attributes"

    print(f"Preprocessing GT for {object_name}")
    print(f"  Found {len(parts)} mesh parts")

    # Step 1: Clean GT mesh by visibility if requested
    if clean_gt and (not cleaned_mesh_path.exists() or force):
        print(f"  Cleaning GT mesh by visibility (use_masks={config.cleanup.use_masks})...")
        import trimesh

        from ..core.mesh import filter_mesh_by_vertex_mask, save_mesh
        from ..core.visibility import filter_by_visibility
        
        cameras_path = config.get_cameras_path(object_name)
        masks_dir = config.get_masks_dir(object_name)
        
        # Fail loudly. The previous behaviour printed a warning, set clean_gt = False and
        # CONTINUED, so an object silently ended up sampled from the RAW mesh while the
        # caller believed it had been cleaned. That silent skip is how 17_knife lost its
        # cleaning on 2026-05-04; nothing in the produced artefacts recorded the choice.
        # If cleaning was explicitly requested and cannot be done, that is an error.
        if not cameras_path.exists():
            raise FileNotFoundError(
                f"GT cleaning requested for {object_name} but no camera file was found. "
                f"Looked under {cameras_path.parent} for cameras.npz / sfm.json / "
                f"cameras.json / *.sfm. Pass --no-clean-gt to sample from the raw mesh "
                f"on purpose, but do not let it happen silently."
            )
        if not masks_dir.exists() and config.cleanup.use_masks:
            raise FileNotFoundError(
                f"GT cleaning requested for {object_name} with use_masks=True but the "
                f"masks directory is missing: {masks_dir}. Set cleanup.use_masks=false "
                f"or pass --no-clean-gt explicitly."
            )
        if True:
            # Load and merge all parts
            merged_mesh = None
            for p in parts:
                print(f"    Loading {p.name}...")
                m = load_mesh(p)
                if merged_mesh is None:
                    merged_mesh = m
                else:
                    merged_mesh = trimesh.util.concatenate([merged_mesh, m])

            cameras = load_cameras_auto(cameras_path)

            print(f"  Raw merged GT mesh: {len(merged_mesh.vertices)} vertices, {len(merged_mesh.faces)} faces")
            
            # Get visibility mask
            visible_mask = filter_by_visibility(
                merged_mesh.vertices,
                merged_mesh,
                cameras,
                masks_dir,
                config.cleanup.dilation_radius,
                show_progress=True,
                use_masks=config.cleanup.use_masks,
            )
            
            # Filter mesh by mask
            cleaned_mesh = filter_mesh_by_vertex_mask(merged_mesh, visible_mask)
            print(f"  Cleaned GT mesh: {len(cleaned_mesh.vertices)} vertices, {len(cleaned_mesh.faces)} faces")
            
            gt_dir.mkdir(parents=True, exist_ok=True)
            save_mesh(cleaned_mesh, cleaned_mesh_path)
            print("  Saved cleaned GT mesh")
    
    # Use cleaned mesh if available, otherwise raw mesh
    if cleaned_mesh_path.exists():
        mesh_to_use = cleaned_mesh_path
        print("  Using cleaned GT mesh")
    else:
        # If not cleaned, we still need to merge parts for sampling
        import trimesh
        merged_mesh = None
        for p in parts:
            m = load_mesh(p)
            if merged_mesh is None:
                merged_mesh = m
            else:
                merged_mesh = trimesh.util.concatenate([merged_mesh, m])
        mesh = merged_mesh
        print("  Using raw merged GT mesh")
    
    if cleaned_mesh_path.exists():
        mesh = load_mesh(cleaned_mesh_path)
    
    print(f"  Final mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    if gt_pcd_path.exists() and not force:
        print("  Loading existing gt_pcd.npy")
        gt_pcd = np.load(gt_pcd_path)
    else:
        print(f"  Upsampling mesh (density={config.evaluation.downsample_density})")
        pcd = upsample_mesh(
            mesh.vertices,
            mesh.faces,
            config.evaluation.downsample_density,
        )
        print(f"  Upsampled to {len(pcd)} points")

        # Pass an explicit seed so the shuffle inside downsample_pcd is reproducible.
        # Without it, downsample_pcd falls back to np.random.default_rng(None), which is
        # seeded from OS entropy: two runs on the same mesh shuffled differently and the
        # greedy radius selection then kept a slightly different subset (~0.04% drift).
        print(f"  Downsampling point cloud (seed={config.evaluation.sampling_seed})")
        gt_pcd = downsample_pcd(
            pcd,
            config.evaluation.downsample_density,
            seed=config.evaluation.sampling_seed,
        )
        print(f"  Downsampled to {len(gt_pcd)} points")

        gt_dir.mkdir(parents=True, exist_ok=True)
        np.save(gt_pcd_path, gt_pcd)
        print("  Saved gt_pcd.npy")

        # Companion .ply of the same cloud, for inspection in a mesh viewer. The .npy is
        # the one the pipeline reads; this is purely so the resampled cloud can be looked
        # at without writing a script.
        try:
            import trimesh as _tm
            _tm.PointCloud(gt_pcd).export(gt_dir / "gt_pcd.ply")
            print(f"  Saved gt_pcd.ply ({len(gt_pcd):,} points)")
        except Exception as _e:  # noqa: BLE001
            print(f"  Warning: could not write gt_pcd.ply ({_e})")

        # Provenance. Until now NO produced artefact recorded which mesh gt_pcd.npy was
        # sampled from, so "is this object cleaned?" could only be answered by hunting
        # through SLURM logs -- and the presence of gt_cleaned.ply does NOT prove the
        # point cloud came from it. Write the answer next to the data.
        import json as _json
        from datetime import datetime as _dt
        _prov = {
            "gt_pcd_sampled_from": "gt_cleaned.ply" if cleaned_mesh_path.exists() else "raw_merged",
            "source_mesh_path": str(mesh_to_use) if cleaned_mesh_path.exists() else [str(p) for p in parts],
            "clean_gt_requested": bool(clean_gt),
            "n_points": int(len(gt_pcd)),
            "mesh_vertices": int(len(mesh.vertices)),
            "mesh_faces": int(len(mesh.faces)),
            "use_masks": bool(config.cleanup.use_masks),
            "dilation_radius": (config.cleanup.dilation_radius if isinstance(config.cleanup.dilation_radius, str) else int(config.cleanup.dilation_radius)),
            "density": float(config.evaluation.downsample_density),
            "sampling_seed": int(config.evaluation.sampling_seed),
            "written": _dt.now().isoformat(timespec="seconds"),
        }
        with open(gt_dir / "gt_pcd_provenance.json", "w") as _f:
            _json.dump(_prov, _f, indent=2)
        print(f"  Saved gt_pcd_provenance.json ({_prov['gt_pcd_sampled_from']})")

    attributes_dir.mkdir(parents=True, exist_ok=True)

    if compute_curvature:
        curvature_path = attributes_dir / "curvature_values.npy"
        if curvature_path.exists() and not force:
            print("  Curvature already exists, skipping")
        else:
            # on the proxy of the raw GT (cached), then to gt_pcd, which lies on a subset of that surface
            points, vertex_curvature = proxy_curvature(config, object_name, force=force)
            curvature = _to_points(vertex_curvature, points, gt_pcd)
            del points, vertex_curvature
            np.save(curvature_path, curvature)
            print("  Saved curvature_values.npy")

    if compute_visibility:
        visibility_path = attributes_dir / "visibility_count.npy"
        if visibility_path.exists() and not force:
            print("  Visibility already exists, skipping")
        else:
            cameras_path = config.get_cameras_path(object_name)
            masks_dir = config.get_masks_dir(object_name)

            # Same rule as the cleaning block above: requested-but-impossible is an error,
            # not a warning. A missing visibility_count.npy is silently indistinguishable
            # from one that was never computed.
            if not cameras_path.exists():
                raise FileNotFoundError(
                    f"Visibility requested for {object_name} but no camera file was found "
                    f"under {cameras_path.parent}. Pass --no-visibility to skip on purpose."
                )
            if not masks_dir.exists() and config.cleanup.use_masks:
                raise FileNotFoundError(
                    f"Visibility requested for {object_name} with use_masks=True but the "
                    f"masks directory is missing: {masks_dir}."
                )
            if True:
                print("  Computing visibility counts")
                cameras = load_cameras_auto(cameras_path)
                visibility = compute_visibility_count(
                    gt_pcd, mesh, cameras, masks_dir, show_progress=True,
                    use_masks=config.cleanup.use_masks
                )
                np.save(visibility_path, visibility)
                print(f"  Saved visibility_count.npy (range: {visibility.min()}-{visibility.max()})")

    print(f"  Done preprocessing {object_name}")
