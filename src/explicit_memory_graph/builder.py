import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover
    _ndimage = None

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None

try:
    from .visualizer import SceneGraphVisualizer, VisualizationMode
except Exception:
    SceneGraphVisualizer = None
    VisualizationMode = None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_list(x: Any) -> Any:
    """Convert numpy arrays / scalars to python lists recursively (best-effort)."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _safe_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe_list(v) for v in x]
    return x


@dataclass
class RegionNode:
    """A lightweight 'room-like' region node built from occupancy/island connectivity."""

    region_id: int
    floor_id: str
    created_step: int
    last_seen_step: int
    # boolean 2D mask in voxel grid coordinates, same shape as planner._vol_dim[:2]
    mask: Optional[np.ndarray] = None
    # list of object ids assigned to this region
    objects: List[int] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "room_id": f"{self.floor_id}_{self.region_id}",
            "name": None,
            "floor_id": self.floor_id,
            "objects": self.objects,
            "created_step": self.created_step,
            "last_seen_step": self.last_seen_step,
            # vertices/pcd fields are intentionally omitted in the online lightweight mode
        }


class ExplicitMemoryGraphBuilder:
    """Online, GPU-memory-friendly explicit memory hierarchy builder.

    Goal:
      - build building->floor->region(room)->object hierarchy online
      - save in an HOV-SG-compatible *json layout* (graph/floors, graph/rooms, graph/objects)

    Design constraints:
      - do not run heavy per-point feature fusion
      - keep all data on CPU (numpy / python)
    """

    def __init__(
        self,
        save_root: str,
        voxel_size: float,
        floor_id: str = "0",
        *,
        # region segmentation / tracking
        region_iou_match_threshold: float = 0.6,
        # filter small free-space components (in cells)
        min_region_area_cells: int = 200,
        # object assignment
        assign_dist_m: float = 4.0,
    ):
        self.save_root = save_root
        self.voxel_size = float(voxel_size)
        self.floor_id = str(floor_id)

        self.region_iou_match_threshold = float(region_iou_match_threshold)
        self.min_region_area_cells = int(min_region_area_cells)
        self.assign_dist_m = float(assign_dist_m)

        # region_id -> RegionNode
        self._regions: Dict[int, RegionNode] = {}
        self._next_region_id = 0

        # cached last region mask for fast stable tracking (single best candidate)
        self._last_region_mask: Optional[np.ndarray] = None
        self._last_region_id: Optional[int] = None

        # object_id -> region_id
        self._obj_to_region: Dict[int, int] = {}

        # metadata
        self._created = False
        self._episode_info: Dict[str, Any] = {}

        # trajectory recording for video generation
        self._trajectory_voxels: List[np.ndarray] = []

    def _label_free_space_regions(self, unoccupied: Optional[np.ndarray]) -> List[np.ndarray]:
        """Return a list of connected free-space component masks.

        unoccupied: bool/0-1 array, True indicates free space.
        """
        if unoccupied is None:
            return []
        free = unoccupied.astype(bool)
        if _ndimage is None:
            # scipy not available; fall back to a single region
            return [free]

        # 8-connectivity on the 2D grid
        structure = np.ones((3, 3), dtype=np.int8)
        labeled, n = _ndimage.label(free, structure=structure)
        if n <= 0:
            return []

        regions: List[np.ndarray] = []
        for lab in range(1, n + 1):
            mask = labeled == lab
            if mask.sum() < self.min_region_area_cells:
                continue
            regions.append(mask)
        # sort by area (desc)
        regions.sort(key=lambda m: int(m.sum()), reverse=True)
        return regions

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union > 0 else 0.0

    def _match_or_create_region(self, mask: np.ndarray, step: int) -> int:
        """Match a component mask to existing regions by IoU, else create."""
        best_rid = None
        best_iou = -1.0
        for rid, rn in self._regions.items():
            if rn.mask is None:
                continue
            iou = self._mask_iou(rn.mask, mask)
            if iou > best_iou:
                best_iou = iou
                best_rid = rid

        if best_rid is not None and best_iou >= self.region_iou_match_threshold:
            rn = self._regions[best_rid]
            rn.last_seen_step = step
            rn.mask = mask
            return best_rid

        rid = self._next_region_id
        self._next_region_id += 1
        self._regions[rid] = RegionNode(
            region_id=rid,
            floor_id=self.floor_id,
            created_step=step,
            last_seen_step=step,
            mask=mask,
        )
        return rid

    def start_episode(self, episode_id: str, scene_id: str):
        self._episode_info = {
            "episode_id": str(episode_id),
            "scene_id": str(scene_id),
        }
        self._created = True

    def _assign_agent_region(self, step: int, tsdf_planner: Any) -> int:
        """Compute (or approximate) the region id where the agent currently is."""
        # Prefer free-space connected components, restricted to the current navigable island if available
        unoccupied = getattr(tsdf_planner, "unoccupied", None)
        island = getattr(tsdf_planner, "island", None)
        if unoccupied is None:
            # make sure planner maps were computed at least once
            return self._match_or_create_region(mask=np.ones((1, 1), dtype=bool), step=step)

        free = unoccupied.astype(bool)
        if island is not None:
            free = np.logical_and(free, island.astype(bool))

        # Try stable tracking against previous agent-region mask first
        if self._last_region_mask is not None and self._last_region_id is not None:
            # If last mask still overlaps with current free-space, keep it
            if self._mask_iou(self._last_region_mask, free) >= 0.2:
                # but we still want the precise connected component; fall back if scipy missing
                pass

        comps = self._label_free_space_regions(free)
        if not comps:
            rid = self._match_or_create_region(mask=free, step=step)
            self._last_region_id = rid
            self._last_region_mask = free
            return rid

        # choose the component that best matches last agent region (or the largest as fallback)
        best_mask = comps[0]
        if self._last_region_mask is not None:
            best_iou = -1.0
            for m in comps:
                iou = self._mask_iou(self._last_region_mask, m)
                if iou > best_iou:
                    best_iou = iou
                    best_mask = m

        rid = self._match_or_create_region(mask=best_mask, step=step)
        self._last_region_id = rid
        self._last_region_mask = best_mask
        return rid

    def update(
        self,
        step: int,
        tsdf_planner: Any,
        scene_objects: Dict[int, Dict[str, Any]],
        agent_pts_habitat: np.ndarray,
    ):
        """Update the explicit graph builder.

        Args:
            step: current step index
            tsdf_planner: TSDFPlanner (CPU planner) used by 3D-Mem
            scene_objects: Scene.objects (MapObjectDict-like)
            agent_pts_habitat: (3,) habitat coordinate
        """
        if not self._created:
            raise RuntimeError("ExplicitMemoryGraphBuilder.start_episode() must be called.")

        # Ensure planner maps exist (unoccupied/island computed in update_frontier_map)
        cur_region_id = self._assign_agent_region(step, tsdf_planner)

        # record trajectory in voxel grid coordinates (for video generation)
        try:
            # rough conversion: habitat xyz -> 2D voxel (x, z)
            x_vox = agent_pts_habitat[0] / self.voxel_size
            z_vox = agent_pts_habitat[2] / self.voxel_size
            self._trajectory_voxels.append(np.array([x_vox, z_vox]))
        except Exception:
            pass

        # assign objects to region (simple heuristic: xy distance in habitat to agent within include dist)
        agent_xy = np.asarray(agent_pts_habitat)[[0, 2]]
        for obj_id, obj in scene_objects.items():
            bbox = obj.get("bbox", None)
            if bbox is None or not hasattr(bbox, "center"):
                continue
            center = np.asarray(bbox.center)
            obj_xy = center[[0, 2]]
            dist = float(np.linalg.norm(obj_xy - agent_xy))

            # only add objects we have at least seen once; do not gate by detections count too strictly here
            if obj.get("num_detections", 0) <= 0:
                continue

            # if an object is near current agent position, map it to current region
            # (this prevents drifting assignments when far away)
            if dist <= self.assign_dist_m:
                prev = self._obj_to_region.get(obj_id)
                if prev is not None and prev in self._regions and prev != cur_region_id:
                    # allow reassignment only if it was never finalized; keep latest
                    if obj_id in self._regions[prev].objects:
                        self._regions[prev].objects.remove(obj_id)
                self._obj_to_region[obj_id] = cur_region_id
                if obj_id not in self._regions[cur_region_id].objects:
                    self._regions[cur_region_id].objects.append(obj_id)

    def _compute_dynamic_map_bounds(
        self, scene_objects: Dict[int, Dict[str, Any]]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute dynamic map bounds from scene objects and regions.
        
        Uses the min/max x, z coordinates from all objects and region masks
        to determine the map extent. This ensures proper world_to_pixel alignment.
        
        Args:
            scene_objects: Scene objects dictionary
            
        Returns:
            (min_bound, max_bound) where each is (3,) array [x, y, z],
            or None if no valid bounds can be computed.
        """
        all_x = []
        all_z = []
        
        # Collect coordinates from objects
        for obj_id, obj in scene_objects.items():
            bbox = obj.get("bbox", None)
            if bbox is not None and hasattr(bbox, "center"):
                center = np.asarray(bbox.center)
                all_x.append(center[0])
                all_z.append(center[2])
        
        # Collect coordinates from region masks
        for rid, rn in self._regions.items():
            if rn.mask is not None:
                # Get mask coordinates and convert to world coordinates
                coords = np.argwhere(rn.mask)
                if len(coords) > 0:
                    # coords are in voxel grid, convert to world
                    world_coords = coords * self.voxel_size
                    all_x.extend(world_coords[:, 0].tolist())
                    all_z.extend(world_coords[:, 1].tolist())
        
        # Collect from trajectory
        if self._trajectory_voxels:
            for pt in self._trajectory_voxels:
                all_x.append(pt[0] * self.voxel_size)
                all_z.append(pt[1] * self.voxel_size)
        
        if not all_x or not all_z:
            return None
        
        # Compute bounds with padding
        padding = 1.0  # 1 meter padding
        min_x, max_x = min(all_x) - padding, max(all_x) + padding
        min_z, max_z = min(all_z) - padding, max(all_z) + padding
        
        # Create (3,) arrays: [x, y, z] where y is height (we use 0 as default)
        min_bound = np.array([min_x, 0.0, min_z])
        max_bound = np.array([max_x, 3.0, max_z])  # Assume 3m ceiling height
        
        return (min_bound, max_bound)

    def _floor_json(self) -> Dict[str, Any]:
        # keep it minimal but compatible with HOV-SG Floor.save() schema
        return {
            "floor_id": self.floor_id,
            "name": f"floor_{self.floor_id}",
            "rooms": [f"{self.floor_id}_{rid}" for rid in sorted(self._regions.keys())],
            "vertices": [],
            "floor_height": None,
            "floor_zero_level": None,
        }

    def _room_json(self, rid: int, snapshot_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        base = self._regions[rid].to_json()
        base["snapshots"] = list(snapshot_keys) if snapshot_keys else []
        return base

    def _object_json(self, obj_id: int, obj: Dict[str, Any],
                     observed_by_snapshots: Optional[List[str]] = None) -> Dict[str, Any]:
        bbox = obj.get("bbox", None)
        center = None
        if bbox is not None and hasattr(bbox, "center"):
            center = np.asarray(bbox.center)

        embedding = obj.get("clip_ft", None)
        if embedding is not None:
            try:
                # move to cpu numpy if it's a torch tensor
                import torch

                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.detach().float().cpu().numpy()
            except Exception:
                pass

        room_id = self._obj_to_region.get(obj_id)
        room_id_str = f"{self.floor_id}_{room_id}" if room_id is not None else None

        vertices = []
        if bbox is not None and hasattr(bbox, "get_box_points"):
            try:
                vertices = np.asarray(bbox.get_box_points())
            except Exception:
                vertices = []

        return {
            "object_id": str(obj_id),
            "room_id": room_id_str,
            "name": obj.get("class_name", None),
            "gt_name": None,
            "vertices": _safe_list(vertices) if len(vertices) else [],
            "bbox_center": _safe_list(center) if center is not None else None,
            "confidence": _safe_float(obj.get("conf", None)),
            "num_detections": int(obj.get("num_detections", 0)),
            "embedding": _safe_list(embedding) if embedding is not None else "",
            "image": obj.get("image", None),
            "observed_by_snapshots": list(observed_by_snapshots) if observed_by_snapshots else [],
            "semantic_entropy": _safe_float(obj.get("semantic_entropy", None)),
            "belief_stable": bool(obj.get("belief_stable", True)),
        }

    def save(
        self,
        scene_objects: Dict[int, Dict[str, Any]],
        *,
        scene_snapshots: Optional[Dict[str, Any]] = None,
        # Visualization mode control
        visualization_mode: Optional[str] = None,  # "textured" | "topology" | None (all)
        # Textured mode parameters
        top_down_map: Optional[np.ndarray] = None,
        agent_position: Optional[np.ndarray] = None,
        agent_heading: Optional[float] = None,
        trajectory: Optional[np.ndarray] = None,
        map_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        # Decision history for conflict visualization
        decision_history: Optional[Dict[int, list]] = None,
        # Target object for highlighting
        target_object_id: Optional[int] = None,
        # Accept and ignore extra keyword arguments from caller
        **kwargs: Any,
    ):
        """Save graph jsons + point clouds + visualizations to disk.

        Produces the following layout (compatible with 01fcc568 reference format):
          explicit_graph/
            meta.json
            graph/
              rooms/   <floor_id>_<rid>.json  (includes snapshots list)
              objects/ <obj_id>.json + .ply   (includes observed_by_snapshots, semantic_entropy)
              snapshots/ <img_name>.json      (evidence_data, full_obj_list, room_id, position)
              visualizations/ scene_graph_topology.png

        Note: floors/ directory is intentionally omitted per user request.

        Args:
            scene_objects: Scene.objects (MapObjectDict-like)
            scene_snapshots: Scene.snapshots dict {img_path -> SnapShot} for snapshot JSON output.
                             If None, the snapshots/ directory is skipped.
            visualization_mode: "textured" | "topology" | None (all)
            top_down_map: RGB top-down map array (H, W, 3)
            agent_position: Agent 3D position (3,) [x, y, z]
            agent_heading: Agent heading angle (radians)
            trajectory: Agent trajectory (N, 3)
            map_bounds: (min_bound, max_bound) for world_to_pixel conversion.
            decision_history: {obj_id: [decisions]} from ConflictResolver
            target_object_id: Target object ID for visualization highlighting.
            **kwargs: Extra keyword args accepted silently for forward-compat.
        """
        graph_root = os.path.join(self.save_root, "explicit_graph", "graph")
        rooms_dir = os.path.join(graph_root, "rooms")
        objects_dir = os.path.join(graph_root, "objects")
        snapshots_dir = os.path.join(graph_root, "snapshots")
        os.makedirs(rooms_dir, exist_ok=True)
        os.makedirs(objects_dir, exist_ok=True)
        os.makedirs(snapshots_dir, exist_ok=True)

        # ── Build helper maps from snapshots ────────────────────────────────
        # obj_id -> list of snapshot image names that observed it
        obj_to_snapshot_names: Dict[int, List[str]] = {}
        # room_id -> list of snapshot image names assigned to that room
        room_to_snapshot_names: Dict[int, List[str]] = {}

        if scene_snapshots:
            for img_name, snapshot in scene_snapshots.items():
                cluster = getattr(snapshot, "cluster", [])
                for obj_id in cluster:
                    obj_to_snapshot_names.setdefault(obj_id, []).append(img_name)

                # Determine which room this snapshot belongs to (via objects in cluster)
                room_id = None
                for obj_id in cluster:
                    r = self._obj_to_region.get(obj_id)
                    if r is not None:
                        room_id = r
                        break
                if room_id is not None:
                    room_to_snapshot_names.setdefault(room_id, []).append(img_name)

        # ── Snapshots/ ───────────────────────────────────────────────────────
        if scene_snapshots:
            for img_name, snapshot in scene_snapshots.items():
                cluster = getattr(snapshot, "cluster", [])
                full_obj_list = getattr(snapshot, "full_obj_list", {})
                obs_point = getattr(snapshot, "obs_point", None)

                # Convert obs_point (voxel coords) to world position
                position = None
                if obs_point is not None:
                    try:
                        op = np.asarray(obs_point)
                        position = _safe_list(op * self.voxel_size)
                    except Exception:
                        position = None

                # Determine room_id for this snapshot
                snap_room_id = None
                for obj_id in cluster:
                    r = self._obj_to_region.get(obj_id)
                    if r is not None:
                        snap_room_id = f"{self.floor_id}_{r}"
                        break

                # evidence_data per object
                evidence_data: Dict[str, Any] = {}
                for obj_id in cluster:
                    obj = scene_objects.get(obj_id, {})
                    conf = full_obj_list.get(obj_id, obj.get("conf", None))
                    evidence_data[str(obj_id)] = {
                        "confidence": _safe_float(conf),
                        "view_angle": None,
                        "bbox_area_ratio": None,
                        "timestamp": None,
                    }

                # full_obj_list as str keys with float values
                full_obj_list_json = {
                    str(k): _safe_float(v) for k, v in full_obj_list.items()
                    if k in scene_objects
                }

                snap_json = {
                    "image": img_name,
                    "obs_point": _safe_list(obs_point) if obs_point is not None else None,
                    "position": position,
                    "target_object_ids": [int(oid) for oid in cluster],
                    "room_id": snap_room_id,
                    "evidence_data": evidence_data,
                    "full_obj_list": full_obj_list_json,
                }
                snap_path = os.path.join(snapshots_dir, f"{img_name}.json")
                with open(snap_path, "w", encoding="utf-8") as f:
                    json.dump(_safe_list(snap_json), f, ensure_ascii=False, indent=2)

        # ── Rooms/ ──────────────────────────────────────────────────────────
        for rid in sorted(self._regions.keys()):
            snap_keys = room_to_snapshot_names.get(rid, [])
            with open(
                os.path.join(rooms_dir, f"{self.floor_id}_{rid}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(_safe_list(self._room_json(rid, snap_keys)), f, ensure_ascii=False, indent=2)

        # ── Objects/ ────────────────────────────────────────────────────────
        for obj_id, obj in scene_objects.items():
            obs_snaps = obj_to_snapshot_names.get(obj_id, [])
            obj_json_path = os.path.join(objects_dir, f"{obj_id}.json")
            with open(obj_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    _safe_list(self._object_json(obj_id, obj, obs_snaps)),
                    f, ensure_ascii=False, indent=2
                )

            # save object point cloud if open3d available
            if o3d is not None:
                pcd = obj.get("pcd", None)
                if pcd is not None and hasattr(pcd, "points") and len(pcd.points) > 0:
                    try:
                        o3d.io.write_point_cloud(os.path.join(objects_dir, f"{obj_id}.ply"), pcd)
                    except Exception:
                        pass

        # ── meta.json ───────────────────────────────────────────────────────
        meta = {
            "episode": self._episode_info,
            "num_regions": len(self._regions),
            "num_objects": len(scene_objects),
            "num_snapshots": len(scene_snapshots) if scene_snapshots else 0,
            "floor_id": self.floor_id,
            "voxel_size": self.voxel_size,
        }
        meta_path = os.path.join(self.save_root, "explicit_graph", "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(_safe_list(meta), f, ensure_ascii=False, indent=2)

        # ── Visualizations/ ─────────────────────────────────────────────────
        computed_map_bounds = map_bounds
        if computed_map_bounds is None:
            computed_map_bounds = self._compute_dynamic_map_bounds(scene_objects)

        self._save_visualizations(
            scene_objects,
            graph_root,
            visualization_mode=visualization_mode,
            top_down_map=top_down_map,
            agent_position=agent_position,
            agent_heading=agent_heading,
            trajectory=trajectory,
            map_bounds=computed_map_bounds,
            decision_history=decision_history,
            target_object_id=target_object_id,
        )

    def _save_visualizations(
        self,
        scene_objects: Dict[int, Dict[str, Any]],
        graph_root: str,
        *,
        visualization_mode: Optional[str] = None,
        top_down_map: Optional[np.ndarray] = None,
        agent_position: Optional[np.ndarray] = None,
        agent_heading: Optional[float] = None,
        trajectory: Optional[np.ndarray] = None,
        map_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        decision_history: Optional[Dict[int, list]] = None,
        target_object_id: Optional[int] = None,
    ):
        """Generate visualization images based on the specified mode.
        
        Args:
            scene_objects: Scene objects dictionary
            graph_root: Root directory for graph output
            visualization_mode: "textured", "topology", or None (all)
            top_down_map: Habitat-rendered RGB top-down map (H, W, 3)
            agent_position: Agent 3D position (3,)
            agent_heading: Agent heading angle (radians)
            trajectory: Agent trajectory (N, 3)
            map_bounds: Map bounds for coordinate transformation
            decision_history: Decision history from ConflictResolver for conflict highlighting
            target_object_id: Target object ID for highlighting
        """
        viz_dir = os.path.join(graph_root, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Use SceneGraphVisualizer if available
        if SceneGraphVisualizer is not None and VisualizationMode is not None:
            visualizer = SceneGraphVisualizer(voxel_size=self.voxel_size, style="academic")
            
            # Set map bounds if provided
            if map_bounds is not None:
                visualizer.set_map_bounds(map_bounds[0], map_bounds[1])
            
            # Collect object positions and classes for textured mode
            object_positions = {}
            object_classes = {}
            for obj_id, obj in scene_objects.items():
                bbox = obj.get("bbox", None)
                if bbox is not None and hasattr(bbox, "center"):
                    object_positions[obj_id] = np.asarray(bbox.center)
                class_name = obj.get("class_name", None)
                if class_name:
                    object_classes[obj_id] = class_name
            
            # Build snapshot positions from trajectory (subsample for clarity)
            snapshot_positions = None
            snapshot_connections = None
            object_to_snapshot = None
            
            if trajectory is not None and len(trajectory) > 0:
                # Subsample trajectory to create snapshot positions
                # Use every N-th point or distance-based sampling
                step = max(1, len(trajectory) // 20)  # Max ~20 snapshots
                snapshot_positions = [trajectory[i] for i in range(0, len(trajectory), step)]
                
                # Build sequential connections
                snapshot_connections = [(i, i+1) for i in range(len(snapshot_positions) - 1)]
                
                # Associate objects to nearest snapshot
                if object_positions:
                    object_to_snapshot = {}
                    for obj_id, obj_pos in object_positions.items():
                        min_dist = float('inf')
                        nearest_snap = 0
                        for snap_idx, snap_pos in enumerate(snapshot_positions):
                            dist = np.linalg.norm(obj_pos[[0,2]] - snap_pos[[0,2]])
                            if dist < min_dist:
                                min_dist = dist
                                nearest_snap = snap_idx
                        object_to_snapshot[obj_id] = nearest_snap
            
            # Determine which modes to generate
            modes_to_generate = []
            if visualization_mode is None:
                # Generate textured and topology modes (removed abstract mode)
                modes_to_generate = ["topology"]
                if top_down_map is not None:
                    modes_to_generate.append("textured")
            else:
                modes_to_generate = [visualization_mode]
            
            for mode_str in modes_to_generate:
                try:
                    if mode_str == "textured" and top_down_map is not None:
                        # Textured mode: High-quality academic visualization
                        output_path = os.path.join(viz_dir, "topdown_textured.png")
                        visualizer.visualize_textured(
                            top_down_map=top_down_map,
                            output_path=output_path,
                            agent_position=agent_position,
                            agent_heading=agent_heading,
                            object_positions=object_positions if object_positions else None,
                            object_classes=object_classes if object_classes else None,
                            trajectory=trajectory,
                            snapshot_positions=snapshot_positions,
                            snapshot_connections=snapshot_connections,
                            object_to_snapshot=object_to_snapshot,
                            target_object_id=target_object_id,  # Pass target object for highlighting
                            title=f"Scene Graph - Floor {self.floor_id}",
                            show_object_labels=True,  # Show class labels
                            show_legend=True,         # Show legend
                            figsize=(20, 16),         # High-resolution output
                            dpi=300,                  # Publication quality
                        )
                    
                    elif mode_str == "topology":
                        # Topology mode: Hierarchical graph with room->object connections
                        output_path = os.path.join(viz_dir, "scene_graph_topology.png")
                        visualizer.visualize_hierarchical_graph(
                            regions=self._regions,
                            scene_objects=scene_objects,
                            obj_to_region=self._obj_to_region,
                            floor_id=self.floor_id,
                            output_path=output_path,
                            decision_history=decision_history,
                            bg_image=top_down_map,  # Pass top_down_map as background image
                        )
                
                except Exception as e:
                    import logging
                    logging.warning(f"Visualization mode '{mode_str}' failed: {e}")
        


