"""
Incremental Deconfliction Module for 3D-MEM
Handles semantic conflicts during object merging in incremental mapping.
"""

import numpy as np
from typing import Dict, Literal, Tuple, Optional, Set, Any
import torch


Decision = Literal["KEEP", "REPLACE", "SPLIT", "MERGE"]


class ConflictResolver:
    """
    Resolves semantic conflicts during incremental object fusion.
    
    Core Logic:
    - Maintains semantic belief distribution per object
    - Tracks observation counts and confidence history
    - Makes merge/split decisions based on feature consistency
    - Manages symmetric mutual exclusion groups to prevent SPLIT loops
    """
    
    def __init__(
        self,
        semantic_threshold: float = 0.7,
        confidence_weight: float = 0.3,
        belief_update_rate: float = 0.1,
        min_detections_for_trust: int = 3,
        spatial_overlap_threshold: float = 0.3,
    ):
        """
        Args:
            semantic_threshold: cosine similarity threshold for semantic consistency
            confidence_weight: weight for confidence vs detection count
            belief_update_rate: rate for belief distribution update
            min_detections_for_trust: minimum observations to trust an object
            spatial_overlap_threshold: spatial IoU threshold for exclusion filtering
        """
        self.semantic_threshold = semantic_threshold
        self.confidence_weight = confidence_weight
        self.belief_update_rate = belief_update_rate
        self.min_detections_for_trust = min_detections_for_trust
        self.spatial_overlap_threshold = spatial_overlap_threshold
        
        # Store belief distributions per object ID
        self.belief_cache: Dict[int, Dict[str, Any]] = {}
        
        # Symmetric mutual exclusion: obj_id -> set of conflicting obj_ids
        self.exclusion_map: Dict[int, Set[int]] = {}
        
        # Decision history: obj_id -> list of decisions
        self.decision_history: Dict[int, list] = {}
        
        # Decision events: (obj_id, decision, position_3d, robot_position_3d)
        self.decision_events: list = []
    
    def _to_numpy(self, feat: Any) -> np.ndarray:
        """Convert feature to numpy array (handles Torch Tensor and ndarray)."""
        if isinstance(feat, torch.Tensor):
            return feat.detach().cpu().numpy()
        elif isinstance(feat, np.ndarray):
            return feat
        else:
            return np.array(feat)
    
    def _compute_cosine_similarity(
        self, 
        feat1: Any, 
        feat2: Any
    ) -> float:
        """
        Compute cosine similarity between two CLIP features.
        Pure NumPy implementation for type safety.
        
        Args:
            feat1, feat2: Can be torch.Tensor or np.ndarray
            
        Returns:
            Cosine similarity in [0, 1] range
        """
        # Convert to numpy
        feat1 = self._to_numpy(feat1).flatten()
        feat2 = self._to_numpy(feat2).flatten()
        
        # Compute norms
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        # Handle zero vectors
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        # Cosine similarity: dot(a, b) / (||a|| * ||b||)
        similarity = np.dot(feat1, feat2) / (norm1 * norm2)
        
        # Clip to [0, 1] range
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _compute_spatial_iou(
        self,
        obj1: dict,
        obj2: dict,
    ) -> float:
        """
        Compute spatial IoU between two objects (bbox or pcd).
        
        Args:
            obj1, obj2: Objects with bbox or pcd
            
        Returns:
            IoU in [0, 1] range
        """
        bbox1 = obj1.get("bbox")
        bbox2 = obj2.get("bbox")
        
        if bbox1 is None or bbox2 is None:
            return 0.0
        
        # Get bbox extents (min, max)
        try:
            if hasattr(bbox1, "get_min_bound") and hasattr(bbox1, "get_max_bound"):
                min1 = np.asarray(bbox1.get_min_bound())
                max1 = np.asarray(bbox1.get_max_bound())
            else:
                return 0.0
            
            if hasattr(bbox2, "get_min_bound") and hasattr(bbox2, "get_max_bound"):
                min2 = np.asarray(bbox2.get_min_bound())
                max2 = np.asarray(bbox2.get_max_bound())
            else:
                return 0.0
        except:
            return 0.0
        
        # Compute intersection
        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)
        inter_dims = np.maximum(0, inter_max - inter_min)
        inter_volume = np.prod(inter_dims)
        
        # Compute union
        vol1 = np.prod(max1 - min1)
        vol2 = np.prod(max2 - min2)
        union_volume = vol1 + vol2 - inter_volume
        
        if union_volume < 1e-8:
            return 0.0
        
        return float(inter_volume / union_volume)
    
    def _compute_trustworthiness(
        self, 
        obj: dict,
    ) -> float:
        """
        Compute trustworthiness score based on detection count and confidence.
        
        Trust = w * normalized_conf + (1-w) * normalized_det_count
        """
        num_det = obj.get("num_detections", 1)
        conf = obj.get("conf", 0.5)
        
        # Normalize detection count (saturates at min_detections_for_trust)
        det_score = min(1.0, num_det / self.min_detections_for_trust)
        
        # Combine confidence and detection count
        trust = self.confidence_weight * conf + (1 - self.confidence_weight) * det_score
        
        return trust
    
    def _check_semantic_conflict(
        self,
        obj_existing: dict,
        obj_new: dict,
    ) -> Tuple[bool, float]:
        """
        Check if there's a semantic conflict between two objects.
        
        Returns:
            (is_conflict, similarity)
        """
        feat_existing = obj_existing.get("clip_ft")
        feat_new = obj_new.get("clip_ft")
        
        if feat_existing is None or feat_new is None:
            return False, 1.0  # No conflict if features missing
        
        similarity = self._compute_cosine_similarity(feat_existing, feat_new)
        
        is_conflict = similarity < self.semantic_threshold
        
        return is_conflict, similarity
    
    def _update_belief(
        self,
        obj_id: int,
        new_feature: Any,
    ) -> None:
        """
        Update semantic belief distribution for an object.
        
        In this simplified implementation, we track feature centroid.
        """
        new_feature = self._to_numpy(new_feature)
        
        if obj_id not in self.belief_cache:
            self.belief_cache[obj_id] = {
                "feature_centroid": new_feature.flatten().copy(),
                "update_count": 1,
            }
        else:
            cached = self.belief_cache[obj_id]
            centroid = cached["feature_centroid"]
            count = cached["update_count"]
            
            # Exponential moving average
            alpha = self.belief_update_rate
            updated_centroid = (1 - alpha) * centroid + alpha * new_feature.flatten()
            
            # Normalize
            norm = np.linalg.norm(updated_centroid)
            if norm > 1e-8:
                updated_centroid = updated_centroid / norm
            
            cached["feature_centroid"] = updated_centroid
            cached["update_count"] = count + 1
    
    def _register_mutual_exclusion(
        self,
        obj_id_1: int,
        obj_id_2: int,
    ) -> None:
        """
        Register SYMMETRIC mutual exclusion between two objects.
        Used after SPLIT to prevent immediate re-matching.
        
        Args:
            obj_id_1, obj_id_2: Object IDs to mark as mutually exclusive
        """
        if obj_id_1 < 0 or obj_id_2 < 0:
            return  # Skip if no valid IDs
        
        # Symmetric registration (BOTH directions)
        if obj_id_1 not in self.exclusion_map:
            self.exclusion_map[obj_id_1] = set()
        if obj_id_2 not in self.exclusion_map:
            self.exclusion_map[obj_id_2] = set()
        
        self.exclusion_map[obj_id_1].add(obj_id_2)
        self.exclusion_map[obj_id_2].add(obj_id_1)
    
    def is_excluded(
        self,
        obj_id_1: int,
        obj_id_2: int,
        spatial_iou: Optional[float] = None,
    ) -> bool:
        """
        Check if two objects are mutually exclusive.
        Spatial-guided filtering: only exclude if spatial overlap is HIGH.
        
        Args:
            obj_id_1, obj_id_2: Object IDs to check
            spatial_iou: Precomputed spatial IoU (optional)
            
        Returns:
            True if they are in exclusion relationship AND spatially overlapping
        """
        if obj_id_1 < 0 or obj_id_2 < 0:
            return False
        
        # Check if in exclusion map
        in_exclusion = (obj_id_1 in self.exclusion_map and 
                       obj_id_2 in self.exclusion_map[obj_id_1])
        
        if not in_exclusion:
            return False
        
        # If spatial IoU provided, only exclude if overlapping
        if spatial_iou is not None:
            return spatial_iou > self.spatial_overlap_threshold
        
        # Default: assume exclusion if in map
        return True
    
    def _record_decision(
        self,
        obj_id: int,
        decision: Decision,
        position_3d: Optional[np.ndarray] = None,
        robot_position_3d: Optional[np.ndarray] = None,
    ) -> None:
        """
        记录决策历史（用于可视化）。
        
        Args:
            obj_id: 物体 ID
            decision: 决策类型
            position_3d: 物体的 3D 位置
            robot_position_3d: 机器人的 3D 位置
        """
        # 记录到物体的决策历史
        if obj_id not in self.decision_history:
            self.decision_history[obj_id] = []
        self.decision_history[obj_id].append(decision)
        
        # 记录决策事件（用于热力图）
        if position_3d is not None:
            self.decision_events.append((obj_id, decision, position_3d, robot_position_3d))
    
    def get_decision_history(self) -> Dict[int, list]:
        """获取所有物体的决策历史。"""
        return self.decision_history
    
    def get_decision_events(self) -> list:
        """获取所有决策事件（用于热力图）。"""
        return self.decision_events
    
    def clear_history(self) -> None:
        """清空决策历史（新场景时调用）。"""
        self.decision_history.clear()
        self.decision_events.clear()
    
    def resolve(
        self,
        obj_existing: dict,
        obj_new: dict,
    ) -> Tuple[Decision, Optional[Tuple[int, int]]]:
        """
        Main decision interface for conflict resolution.
        
        Decision Logic:
        1. Check semantic consistency via CLIP feature similarity
        2. If consistent (>= threshold) -> MERGE
        3. If conflict detected:
           a. Compare trustworthiness
           b. If new object more trustworthy -> REPLACE
           c. If existing more trustworthy -> KEEP (don't merge)
           d. If comparable but irreconcilable -> SPLIT (create new object)
        4. For SPLIT: register SYMMETRIC mutual exclusion
        
        Args:
            obj_existing: existing map object (with clip_ft, conf, num_detections, id)
            obj_new: newly detected object
            
        Returns:
            (decision, exclusion_pair)
            - decision: one of {"KEEP", "REPLACE", "SPLIT", "MERGE"}
            - exclusion_pair: (obj_id_1, obj_id_2) if SPLIT, else None
        """
        
        # Check semantic conflict
        is_conflict, similarity = self._check_semantic_conflict(obj_existing, obj_new)
        
        # No conflict -> safe to merge
        if not is_conflict:
            obj_id = obj_existing.get("id", -1)
            if obj_id >= 0:
                self._update_belief(obj_id, obj_existing["clip_ft"])
            return "MERGE", None
        
        # Conflict detected -> compare trustworthiness
        trust_existing = self._compute_trustworthiness(obj_existing)
        trust_new = self._compute_trustworthiness(obj_new)
        
        trust_diff = abs(trust_existing - trust_new)
        
        # Clear winner -> REPLACE or KEEP
        if trust_diff > 0.2:
            if trust_new > trust_existing:
                return "REPLACE", None
            else:
                return "KEEP", None
        
        # Comparable trust but conflicting semantics -> SPLIT
        # Register SYMMETRIC mutual exclusion to prevent re-matching
        obj_id_existing = obj_existing.get("id", -1)
        obj_id_new = obj_new.get("id", -1)
        
        if obj_id_existing >= 0 and obj_id_new >= 0:
            exclusion_pair = (obj_id_existing, obj_id_new)
            self._register_mutual_exclusion(obj_id_existing, obj_id_new)
        else:
            exclusion_pair = None
        
        return "SPLIT", exclusion_pair
    
    def should_merge(
        self,
        obj_existing: dict,
        obj_new: dict,
        similarity: float,
    ) -> bool:
        """
        Quick check if merge should proceed (used before full resolve).
        Includes spatial-guided exclusion filtering.
        
        Args:
            obj_existing: existing map object
            obj_new: newly detected object
            similarity: precomputed similarity score
            
        Returns:
            True if merge is safe, False otherwise
        """
        # Check mutual exclusion with spatial filtering
        obj_id_existing = obj_existing.get("id", -1)
        obj_id_new = obj_new.get("id", -1)
        
        if obj_id_existing >= 0 and obj_id_new >= 0:
            spatial_iou = self._compute_spatial_iou(obj_existing, obj_new)
            if self.is_excluded(obj_id_existing, obj_id_new, spatial_iou):
                return False  # Skip if spatially overlapping and in exclusion list
        
        # Fast path: if similarity is very high, always merge
        if similarity > 0.85:
            return True
        
        # If below semantic threshold, need careful resolution
        if similarity < self.semantic_threshold:
            return False
        
        # In-between zone: check trustworthiness
        trust_existing = self._compute_trustworthiness(obj_existing)
        trust_new = self._compute_trustworthiness(obj_new)
        
        # If new detection is much less trustworthy, reject
        if trust_new < 0.3 and trust_existing > 0.6:
            return False
        
        return True


class ConflictResolverFactory:
    """Factory for creating pre-configured ConflictResolver instances."""
    
    @staticmethod
    def create_conservative() -> ConflictResolver:
        """Conservative resolver (high threshold, prefers SPLIT)."""
        return ConflictResolver(
            semantic_threshold=0.75,
            confidence_weight=0.4,
            belief_update_rate=0.05,
            min_detections_for_trust=5,
            spatial_overlap_threshold=0.3,
        )
    
    @staticmethod
    def create_aggressive() -> ConflictResolver:
        """Aggressive resolver (low threshold, prefers MERGE)."""
        return ConflictResolver(
            semantic_threshold=0.6,
            confidence_weight=0.2,
            belief_update_rate=0.15,
            min_detections_for_trust=2,
            spatial_overlap_threshold=0.3,
        )
    
    @staticmethod
    def create_default() -> ConflictResolver:
        """Balanced default resolver."""
        return ConflictResolver()
