import logging
from typing import Tuple, Optional, Union

from src.eval_utils_gpt_aeqa import explore_step
from src.tsdf_planner import TSDFPlanner, SnapShot, Frontier
from src.scene_aeqa import Scene


def query_vlm_for_response(
    question: str,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    rgb_egocentric_views: list,
    cfg,
    verbose: bool = False,
) -> Optional[Tuple[Union[SnapShot, Frontier], str, int]]:
    # prepare input for vlm
    step_dict = {}

    # prepare room exploration status from builder (if available)
    if hasattr(scene, 'builder') and scene.builder is not None and hasattr(scene.builder, 'get_room_exploration_status'):
        try:
            step_dict["room_exploration_status"] = scene.builder.get_room_exploration_status()
        except Exception:
            step_dict["room_exploration_status"] = None
    else:
        step_dict["room_exploration_status"] = None

    # prepare snapshots
    object_id_to_name = {
        obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()
    }
    step_dict["obj_map"] = object_id_to_name

    # Build indicator reverse index from builder for semantic conflict annotation
    indicator_map = {}
    class_to_room_type = {}
    if hasattr(scene, 'builder') and scene.builder is not None:
        indicator_map = getattr(scene.builder, '_indicator_map', {})
        class_to_room_type = getattr(scene.builder, '_class_to_room_type', {})

    step_dict["snapshot_objects"] = {}
    step_dict["snapshot_imgs"] = {}
    step_dict["snapshot_room_ids"] = {}
    step_dict["snapshot_room_labels"] = {}
    for rgb_id, snapshot in scene.snapshots.items():
        step_dict["snapshot_objects"][rgb_id] = snapshot.cluster
        step_dict["snapshot_imgs"][rgb_id] = scene.all_observations[rgb_id]
        step_dict["snapshot_room_ids"][rgb_id] = snapshot.room_id

        # Build base room label (never None)
        if snapshot.room_name is not None:
            base_label = str(snapshot.room_name)
        elif snapshot.room_id is not None:
            base_label = f"Room_{snapshot.room_id}"
        else:
            base_label = "unknown area"

        # Detect semantic conflict: objects in this snapshot that belong to a different room type
        if class_to_room_type and snapshot.room_name is not None:
            current_type = snapshot.room_name.replace(" ", "_").lower()
            conflict_types = set()
            for obj_id in snapshot.cluster:
                obj = scene.objects.get(obj_id, {})
                cn = obj.get("class_name", None) if obj else None
                if cn is None:
                    continue
                assigned_type = class_to_room_type.get(cn)
                if assigned_type is not None and assigned_type != current_type:
                    conflict_types.add(assigned_type.replace("_", " "))
            if conflict_types:
                conflict_str = ", ".join(sorted(conflict_types))
                base_label = f"{base_label}, contains {conflict_str} objects"

        step_dict["snapshot_room_labels"][rgb_id] = base_label

    # prepare frontier
    step_dict["frontier_imgs"] = [
        frontier.feature for frontier in tsdf_planner.frontiers
    ]

    # prepare egocentric views
    if cfg.egocentric_views:
        step_dict["egocentric_views"] = rgb_egocentric_views
        step_dict["use_egocentric_views"] = True

    # prepare question
    step_dict["question"] = question

    # query vlm
    outputs, snapshot_id_mapping, reason, n_filtered_snapshots = explore_step(
        step_dict, cfg, verbose=verbose
    )
    if outputs is None:
        logging.error(f"explore_step failed and returned None")
        return None
    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")

    # parse returned results
    try:
        target_type, target_index = outputs.split(" ")[0], outputs.split(" ")[1]
        logging.info(f"Prediction: {target_type}, {target_index}")
    except:
        logging.info(f"Wrong output format, failed!")
        return None

    if target_type not in ["snapshot", "frontier"]:
        logging.info(f"Wrong target type: {target_type}, failed!")
        return None

    if target_type == "snapshot":
        if int(target_index) < 0 or int(target_index) >= len(snapshot_id_mapping):
            logging.info(
                f"Target index can not match real objects: {target_index}, failed!"
            )
            return None
        target_index = snapshot_id_mapping[int(target_index)]
        logging.info(f"The index of target snapshot {target_index}")

        # get the target snapshot
        if target_index < 0 or target_index >= len(scene.snapshots):
            logging.info(
                f"Predicted snapshot target index out of range: {target_index}, failed!"
            )
            return None

        pred_target_snapshot = list(scene.snapshots.values())[target_index]
        logging.info(
            "Pred_target_class: "
            + str(
                " ".join(
                    [
                        object_id_to_name[obj_id]
                        for obj_id in pred_target_snapshot.cluster
                    ]
                )
            )
        )
        logging.info(f"Next choice Snapshot of {pred_target_snapshot.image}")

        return pred_target_snapshot, reason, n_filtered_snapshots
    else:  # target_type == "frontier"
        target_index = int(target_index)
        if target_index < 0 or target_index >= len(tsdf_planner.frontiers):
            logging.info(
                f"Predicted frontier target index out of range: {target_index}, failed!"
            )
            return None
        target_point = tsdf_planner.frontiers[target_index].position
        logging.info(f"Next choice: Frontier at {target_point}")
        pred_target_frontier = tsdf_planner.frontiers[target_index]

        return pred_target_frontier, reason, n_filtered_snapshots
