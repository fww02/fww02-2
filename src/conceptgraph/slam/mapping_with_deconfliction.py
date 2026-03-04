"""
Modified mapping.py with integrated Conflict Resolution.

Key changes:
1. Import ConflictResolver
2. merge_detections_to_objects_v2: new version with conflict handling
3. Four decision paths: KEEP / REPLACE / SPLIT / MERGE

NOTE on SPLIT:
  "SPLIT" is a *conflict-detection label* only.  It signals that the semantic
  conflict between an existing object and a new detection is unresolvable.
  No physical object is created or mutated.  The caller treats SPLIT as KEEP
  (existing object is preserved, conflicting detection is discarded) and logs
  the event in ConflictResolver.decision_history for visualization purposes.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from src.conceptgraph.slam.slam_classes import (
    MapObjectDict,
    DetectionDict,
    MapObjectList,
    DetectionList,
)
from src.conceptgraph.slam.utils import (
    merge_obj2_into_obj1,
)
from src.conceptgraph.slam.conflict_resolver import ConflictResolver, ConflictResolverFactory


def merge_detections_to_objects_v2(
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
    match_method: str,
    phys_bias: float,
    detection_list: DetectionList,
    objects: MapObjectList,
    agg_sim: torch.Tensor,
    conflict_resolver: Optional[ConflictResolver] = None,
    enable_deconfliction: bool = True,
) -> MapObjectList:
    """
    Enhanced version of merge_detections_to_objects with conflict resolution.
    
    Args:
        (same as original function)
        conflict_resolver: ConflictResolver instance for handling conflicts
        enable_deconfliction: whether to enable conflict resolution
    
    Returns:
        Updated MapObjectList
    """
    
    # Fallback to original behavior if deconfliction disabled
    if not enable_deconfliction or conflict_resolver is None:
        # Original argmax + merge logic
        for detected_obj_idx in range(agg_sim.shape[0]):
            if agg_sim[detected_obj_idx].max() == float("-inf"):
                objects.append(detection_list[detected_obj_idx])
            else:
                existing_obj_match_idx = agg_sim[detected_obj_idx].argmax().item()
                detected_obj = detection_list[detected_obj_idx]
                matched_obj = objects[existing_obj_match_idx]
                merged_obj = merge_obj2_into_obj1(
                    obj1=matched_obj,
                    obj2=detected_obj,
                    downsample_voxel_size=downsample_voxel_size,
                    dbscan_remove_noise=dbscan_remove_noise,
                    dbscan_eps=dbscan_eps,
                    dbscan_min_points=dbscan_min_points,
                    spatial_sim_type=spatial_sim_type,
                    device=device,
                    run_dbscan=False,
                )
                objects[existing_obj_match_idx] = merged_obj
        return objects
    
    # Initialize conflict resolver with default if None provided
    if conflict_resolver is None:
        conflict_resolver = ConflictResolverFactory.create_default()
    
    # Track statistics
    stats = {
        "merged": 0,
        "split": 0,
        "replaced": 0,
        "kept": 0,
        "new": 0,
    }
    
    for detected_obj_idx in range(agg_sim.shape[0]):
        # No match found -> create new object
        if agg_sim[detected_obj_idx].max() == float("-inf"):
            objects.append(detection_list[detected_obj_idx])
            stats["new"] += 1
            continue
        
        # Find best matching existing object
        existing_obj_match_idx = agg_sim[detected_obj_idx].argmax().item()
        similarity_score = agg_sim[detected_obj_idx, existing_obj_match_idx].item()
        
        detected_obj = detection_list[detected_obj_idx]
        matched_obj = objects[existing_obj_match_idx]
        
        # === Conflict Resolution Decision Point ===
        decision, exclusion_pair = conflict_resolver.resolve(matched_obj, detected_obj)
        
        # === Execute decision ===
        if decision == "MERGE":
            # Normal merge: obj2 -> obj1
            merged_obj = merge_obj2_into_obj1(
                obj1=matched_obj,
                obj2=detected_obj,
                downsample_voxel_size=downsample_voxel_size,
                dbscan_remove_noise=dbscan_remove_noise,
                dbscan_eps=dbscan_eps,
                dbscan_min_points=dbscan_min_points,
                spatial_sim_type=spatial_sim_type,
                device=device,
                run_dbscan=False,
            )
            objects[existing_obj_match_idx] = merged_obj
            stats["merged"] += 1
        
        elif decision == "REPLACE":
            # Replace existing with new (existing is deemed unreliable)
            # Preserve ID but take new object's data
            new_obj = detected_obj.copy() if hasattr(detected_obj, 'copy') else dict(detected_obj)
            if "id" in matched_obj:
                new_obj["id"] = matched_obj["id"]
            objects[existing_obj_match_idx] = new_obj
            stats["replaced"] += 1
        
        elif decision == "KEEP":
            # Keep existing, discard new detection (new is unreliable)
            # No merge occurs
            stats["kept"] += 1
            pass
        
        elif decision == "SPLIT":
            # Conflict detected but unresolvable: do NOT physically create a new object.
            # Instead, treat as KEEP (preserve existing object, discard conflicting detection).
            # The conflict is already recorded in conflict_resolver.decision_history for
            # downstream logging / visualization; no object is mutated or duplicated.
            stats["split"] += 1  # still count for statistics
        
        else:
            raise ValueError(f"Unknown decision: {decision}")
    
    # Optional: log statistics
    # print(f"Merge stats: {stats}")
    
    return objects


def merge_detections_to_objects_with_filter(
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
    match_method: str,
    phys_bias: float,
    detection_list: DetectionList,
    objects: MapObjectList,
    agg_sim: torch.Tensor,
    visual_sim: torch.Tensor,
    conflict_resolver: Optional[ConflictResolver] = None,
) -> MapObjectList:
    """
    Alternative version: filter before merge using should_merge() quick check.
    
    This version is more efficient as it pre-filters conflicts.
    """
    
    if conflict_resolver is None:
        conflict_resolver = ConflictResolver()
    
    for detected_obj_idx in range(agg_sim.shape[0]):
        if agg_sim[detected_obj_idx].max() == float("-inf"):
            objects.append(detection_list[detected_obj_idx])
            continue
        
        existing_obj_match_idx = agg_sim[detected_obj_idx].argmax().item()
        detected_obj = detection_list[detected_obj_idx]
        matched_obj = objects[existing_obj_match_idx]
        
        # Quick filter using visual similarity
        vis_sim = visual_sim[detected_obj_idx, existing_obj_match_idx].item()
        
        if conflict_resolver.should_merge(matched_obj, detected_obj, vis_sim):
            # Safe to merge
            merged_obj = merge_obj2_into_obj1(
                obj1=matched_obj,
                obj2=detected_obj,
                downsample_voxel_size=downsample_voxel_size,
                dbscan_remove_noise=dbscan_remove_noise,
                dbscan_eps=dbscan_eps,
                dbscan_min_points=dbscan_min_points,
                spatial_sim_type=spatial_sim_type,
                device=device,
                run_dbscan=False,
            )
            objects[existing_obj_match_idx] = merged_obj
        else:
            # Conflict detected -> create as new object
            objects.append(detected_obj)
    
    return objects


# ==================== Integration Example ====================

def example_integration_in_slam_pipeline():
    """
    Example showing how to integrate ConflictResolver into existing pipeline.
    
    Replace the original merge_detections_to_objects call with:
    """
    
    # In your SLAM pipeline (e.g., cfslam_pipeline_batch.py):
    from src.conceptgraph.slam.conflict_resolver import ConflictResolver, ConflictResolverFactory
    from src.conceptgraph.slam.mapping_with_deconfliction import merge_detections_to_objects_v2
    
    # Initialize resolver (do this once at pipeline start)
    conflict_resolver = ConflictResolverFactory.create_default()
    # Or for conservative behavior:
    # conflict_resolver = ConflictResolverFactory.create_conservative()
    
    # When merging detections:
    # OLD CODE:
    # objects = merge_detections_to_objects(
    #     cfg, fg_detection_list, objects, agg_sim
    # )
    
    # NEW CODE:
    # objects = merge_detections_to_objects_v2(
    #     downsample_voxel_size=cfg.downsample_voxel_size,
    #     dbscan_remove_noise=cfg.dbscan_remove_noise,
    #     dbscan_eps=cfg.dbscan_eps,
    #     dbscan_min_points=cfg.dbscan_min_points,
    #     spatial_sim_type=cfg.spatial_sim_type,
    #     device=cfg.device,
    #     match_method=cfg.match_method,
    #     phys_bias=cfg.phys_bias,
    #     detection_list=fg_detection_list,
    #     objects=objects,
    #     agg_sim=agg_sim,
    #     conflict_resolver=conflict_resolver,
    #     enable_deconfliction=True,
    # )
    
    pass


# ==================== Drop-in Replacement ====================

def create_drop_in_replacement_for_mapping_py():
    """
    To use this in your existing code without changing imports:
    
    1. Backup original mapping.py
    2. Add this to mapping.py:
    """
    
    replacement_code = '''
# Add at top of mapping.py
from src.conceptgraph.slam.conflict_resolver import ConflictResolver

# Add module-level resolver
_global_conflict_resolver = ConflictResolver()

# Replace merge_detections_to_objects with this:
def merge_detections_to_objects(
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
    match_method: str,
    phys_bias: float,
    detection_list: DetectionList,
    objects: MapObjectList,
    agg_sim: torch.Tensor,
) -> MapObjectList:
    """Original function signature maintained for backward compatibility."""
    
    for detected_obj_idx in range(agg_sim.shape[0]):
        if agg_sim[detected_obj_idx].max() == float("-inf"):
            objects.append(detection_list[detected_obj_idx])
        else:
            existing_obj_match_idx = agg_sim[detected_obj_idx].argmax().item()
            detected_obj = detection_list[detected_obj_idx]
            matched_obj = objects[existing_obj_match_idx]
            
            # === CONFLICT RESOLUTION INSERTED HERE ===
            decision = _global_conflict_resolver.resolve(matched_obj, detected_obj)
            
            if decision == "MERGE":
                merged_obj = merge_obj2_into_obj1(
                    obj1=matched_obj,
                    obj2=detected_obj,
                    downsample_voxel_size=downsample_voxel_size,
                    dbscan_remove_noise=dbscan_remove_noise,
                    dbscan_eps=dbscan_eps,
                    dbscan_min_points=dbscan_min_points,
                    spatial_sim_type=spatial_sim_type,
                    device=device,
                    run_dbscan=False,
                )
                objects[existing_obj_match_idx] = merged_obj
            
            elif decision == "REPLACE":
                new_obj = dict(detected_obj)
                if "id" in matched_obj:
                    new_obj["id"] = matched_obj["id"]
                objects[existing_obj_match_idx] = new_obj
            
            elif decision == "KEEP":
                pass  # Discard new detection
            
            elif decision == "SPLIT":
                # Conflict detected: do NOT create a new object.
                # Treat as KEEP - preserve existing, discard conflicting detection.
                pass  # log via conflict_resolver.decision_history
    
    return objects
'''
    
    return replacement_code
