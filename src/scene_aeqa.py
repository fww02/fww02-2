import os
import numpy as np
import logging
import random
import torch
import habitat_sim
import quaternion
from quaternion import as_float_array
import supervision as sv
import logging
from collections import Counter
from typing import List, Optional, Tuple, Dict, Union
import copy

from habitat_sim.utils.common import (
    quat_to_coeffs,
    quat_from_angle_axis,
    quat_from_two_vectors,
)
from src.habitat import (
    make_semantic_cfg,
    make_simple_cfg,
    get_quaternion,
    get_navigable_point_to,
)
from src.geom import get_cam_intr, IoU
from src.utils import rgba2rgb
from src.tsdf_planner import SnapShot
from src.hierarchy_clustering import SceneHierarchicalClustering

# Local application/library specific imports
from src.conceptgraph.utils.ious import mask_subtract_contained
from src.conceptgraph.utils.general_utils import (
    ObjectClasses,
    measure_time,
    filter_detections,
)
from src.conceptgraph.slam.slam_classes import MapObjectDict, DetectionDict, to_tensor
from src.conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    denoise_objects,
    merge_objects,
    detections_to_obj_pcd_and_bbox,
    processing_needed,
    resize_gobs,
    merge_obj2_into_obj1,
)
from src.conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
)
from src.conceptgraph.utils.model_utils import compute_clip_features_batched


class Scene:
    def __init__(
        self,
        scene_id,
        cfg,
        graph_cfg,
        detection_model,
        sam_predictor,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
    ):
        self.cfg = cfg
        # concept graph configuration
        self.cfg_cg = graph_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # about the loading the scene
        split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
        scene_mesh_path = os.path.join(
            cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.glb"
        )
        navmesh_path = os.path.join(
            cfg.scene_data_path,
            split,
            scene_id,
            scene_id.split("-")[1] + ".basis.navmesh",
        )
        semantic_texture_path = os.path.join(
            cfg.scene_data_path,
            split,
            scene_id,
            scene_id.split("-")[1] + ".semantic.glb",
        )
        scene_semantic_annotation_path = os.path.join(
            cfg.scene_data_path,
            split,
            scene_id,
            scene_id.split("-")[1] + ".semantic.txt",
        )
        assert os.path.exists(
            scene_mesh_path
        ), f"scene_mesh_path: {scene_mesh_path} does not exist"
        assert os.path.exists(
            navmesh_path
        ), f"navmesh_path: {navmesh_path} does not exist"
        if not os.path.exists(semantic_texture_path) or not os.path.exists(
            scene_semantic_annotation_path
        ):
            logging.warning(
                f"semantic_texture_path: {semantic_texture_path} or scene_semantic_annotation_path: {scene_semantic_annotation_path} does not exist"
            )

        sim_settings = {
            "scene": scene_mesh_path,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": cfg.img_width,
            "height": cfg.img_height,
            "hfov": cfg.hfov,
            "scene_dataset_config_file": cfg.scene_dataset_config_path,
            "camera_tilt": cfg.camera_tilt_deg * np.pi / 180,
        }
        if os.path.exists(semantic_texture_path) and os.path.exists(
            scene_semantic_annotation_path
        ):
            sim_cfg = make_semantic_cfg(sim_settings)
        else:
            sim_cfg = make_simple_cfg(sim_settings)
        self.simulator = habitat_sim.Simulator(sim_cfg)
        self.pathfinder = self.simulator.pathfinder
        self.pathfinder.seed(cfg.seed)
        self.pathfinder.load_nav_mesh(navmesh_path)

        # load object classes
        # maintain a list of object classes
        self.obj_classes = ObjectClasses(
            classes_file_path=scene_semantic_annotation_path,
            bg_classes=self.cfg_cg["bg_classes"],
            skip_bg=self.cfg_cg["skip_bg"],
            class_set=self.cfg["class_set"],
        )

        if os.path.exists(semantic_texture_path) and os.path.exists(
            scene_semantic_annotation_path
        ):
            logging.info(f"Load scene {scene_id} successfully with semantic texture")
        else:
            logging.info(f"Load scene {scene_id} successfully without semantic texture")

        # set agent
        self.agent = self.simulator.initialize_agent(sim_settings["default_agent"])

        self.cam_intrinsic = get_cam_intr(cfg.img_width, cfg.img_height, cfg.hfov)

        # about scene graph
        self.objects: MapObjectDict[int, Dict] = (
            MapObjectDict()
        )  # object_id -> object item
        self.object_id_counter = 1

        self.snapshots: Dict[str, SnapShot] = {}  # image_path -> snapshot
        self.frames: Dict[str, SnapShot] = {}  # image_path -> all frames
        self.all_observations: Dict[str, np.ndarray] = (
            {}
        )  # image_path -> image, stores all actual observations at each step, used for querying vlm

        self.clustering = SceneHierarchicalClustering(
            min_sample_split=0,
            random_state=66,
        )

        # reference to ExplicitMemoryGraphBuilder (set externally after creation)
        self.builder = None

        # setup detection and segmentation models
        self.detection_model = detection_model
        self.detection_model.set_classes(self.obj_classes.get_classes_arr())

        self.sam_predictor = sam_predictor

        self.clip_model = clip_model.to(self.device)
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer

    def __del__(self):
        try:
            self.simulator.close()
        except:
            pass

    def clear_up_detections(self):
        self.objects = MapObjectDict()
        self.object_id_counter = 1

        self.snapshots = {}
        self.frames = {}
        self.all_observations = {}

    def update_snapshot_rooms(self, obj_to_region: dict, agent_pts: Optional[np.ndarray] = None):
        """Assign room_id and room_name to each snapshot via distance-weighted voting.

        Uses object-to-agent distance to weight each object's region vote.
        Weight formula: w = max(0, 1 - d / 5.0), so nearby objects have higher influence.
        Falls back to equal-weight voting when agent_pts is unavailable.

        Also registers each snapshot with its room in the builder (if available)
        and attempts to infer/refresh a human-readable room name.

        Args:
            obj_to_region: dict mapping object_id (int) -> region_id (int),
                           from ExplicitMemoryGraphBuilder.obj_to_region.
            agent_pts: (3,) habitat coordinate of agent, used for distance weighting.
        """
        for snap_idx, snapshot in enumerate(self.snapshots.values()):
            if not snapshot.cluster:
                snapshot.room_id = None
                snapshot.room_name = "unknown area"
                continue

            # Distance-weighted vote: region_id -> accumulated weight
            weighted_scores: dict = {}
            has_any_region = False

            for obj_id in snapshot.cluster:
                rid = obj_to_region.get(obj_id)
                if rid is None:
                    continue
                has_any_region = True

                # Compute distance weight
                weight = 1.0
                if agent_pts is not None:
                    obj = self.objects.get(obj_id, {})
                    bbox = obj.get("bbox", None)
                    if bbox is not None and hasattr(bbox, "center"):
                        center = np.asarray(bbox.center)
                        dist = float(np.linalg.norm(center[[0, 2]] - np.asarray(agent_pts)[[0, 2]]))
                        weight = max(0.0, 1.0 - dist / 5.0)

                weighted_scores[rid] = weighted_scores.get(rid, 0.0) + weight

            if not has_any_region:
                snapshot.room_id = None
                snapshot.room_name = "unknown area"
                continue

            # Pick highest-scoring region
            best_room_id = max(weighted_scores, key=weighted_scores.get)
            snapshot.room_id = best_room_id

            # Try to get/infer room name via builder
            if self.builder is not None:
                try:
                    name = self.builder.get_room_name(snapshot.room_id, scene_objects=self.objects)
                    # get_room_name always returns non-None (falls back to "Room_<id>")
                    snapshot.room_name = name if name is not None else f"Room_{snapshot.room_id}"
                except Exception:
                    snapshot.room_name = f"Room_{snapshot.room_id}"
                # Register snapshot → room association
                try:
                    self.builder.add_snapshot_to_room(snapshot.image, snapshot.room_id)
                except Exception:
                    pass
            else:
                snapshot.room_name = f"Room_{snapshot.room_id}"

            logging.info(
                f"Snapshot {snap_idx} assigned to '{snapshot.room_name}' "
                f"(Room {snapshot.room_id}) via distance-weighted voting."
            )

    def get_observation(self, pts, angle):
        agent_state = habitat_sim.AgentState()
        agent_state.position = pts
        agent_state.rotation = get_quaternion(angle, 0)
        self.agent.set_state(agent_state)

        obs = self.simulator.get_sensor_observations()

        # get camera extrinsic matrix
        sensor = self.agent.get_state().sensor_states["depth_sensor"]
        quaternion_0 = sensor.rotation
        translation_0 = sensor.position
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
        cam_pose[:3, 3] = translation_0

        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])

        return obs, cam_pose

    def get_frontier_observation(self, pts, view_dir, camera_tilt=0.0):
        agent_state = habitat_sim.AgentState()

        # solve edge cases of viewing direction
        default_view_dir = np.asarray([0.0, 0.0, -1.0])
        if np.linalg.norm(view_dir) < 1e-3:
            view_dir = default_view_dir
        view_dir = view_dir / np.linalg.norm(view_dir)

        agent_state.position = pts
        # set agent observation direction
        if np.dot(view_dir, default_view_dir) / np.linalg.norm(view_dir) < -1 + 1e-3:
            # if the rotation is to rotate 180 degree, then the quaternion is not unique
            # we need to specify rotating along y-axis
            agent_state.rotation = quat_to_coeffs(
                quaternion.quaternion(0, 0, 1, 0)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()
        else:
            agent_state.rotation = quat_to_coeffs(
                quat_from_two_vectors(default_view_dir, view_dir)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()

        self.agent.set_state(agent_state)
        obs = self.simulator.get_sensor_observations()

        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])

        return obs

    def get_frontier_observation_and_detect_target(
        self,
        pts,
        view_dir,
        detection_model,
        target_obj_id,
        target_obj_class,
        camera_tilt=0.0,
    ):
        obs = self.get_frontier_observation(pts, view_dir, camera_tilt)

        # detect target object
        rgb = obs["color_sensor"]
        semantic_obs = obs["semantic_sensor"]

        detection_model.set_classes([target_obj_class])
        results = detection_model.infer(
            rgb[..., :3], confidence=self.cfg.scene_graph.confidence
        )
        detections = sv.Detections.from_inference(results).with_nms(
            threshold=self.cfg.scene_graph.nms_threshold
        )

        target_detected = False
        if target_obj_id in np.unique(semantic_obs):
            for i in range(len(detections)):
                x_start, y_start, x_end, y_end = detections.xyxy[i].astype(int)
                bbox_mask = np.zeros(semantic_obs.shape, dtype=bool)
                bbox_mask[y_start:y_end, x_start:x_end] = True

                target_x_start, target_y_start = np.argwhere(
                    semantic_obs == target_obj_id
                ).min(axis=0)
                target_x_end, target_y_end = np.argwhere(
                    semantic_obs == target_obj_id
                ).max(axis=0)
                obj_mask = np.zeros(semantic_obs.shape, dtype=bool)
                obj_mask[target_x_start:target_x_end, target_y_start:target_y_end] = (
                    True
                )
                if IoU(bbox_mask, obj_mask) > self.cfg.scene_graph.iou_threshold:
                    target_detected = True
                    break

        return obs, target_detected

    def get_navigable_point_to(
        self,
        target_position,
        max_search=1000,
        min_dist=6.0,
        max_dist=999.0,
        prev_start_positions=None,
    ):
        self.pathfinder.seed(random.randint(0, 1000000))
        return get_navigable_point_to(
            target_position,
            self.pathfinder,
            max_search,
            min_dist,
            max_dist,
            prev_start_positions,
        )

    def update_scene_graph(
        self,
        image_rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics,
        cam_pos,
        pts,
        pts_voxel,
        img_path,
        frame_idx,
        target_obj_mask=None,  # the boolean mask of target object generated from the semantic sensor. If given, return the object id of the target object
    ) -> Tuple[np.ndarray, List[int], Optional[int]]:
        # return annotated image; the detected object ids in current frame; the object id of the target object (if detected)

        # set up object_classes first
        obj_classes = self.obj_classes

        # Detect objects
        results = self.detection_model.predict(image_rgb, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detection_class_labels = [
            f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(detection_class_ids)
        ]
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        # if there are detections,
        # Get Masks Using SAM or MobileSAM
        # UltraLytics SAM
        if xyxy_tensor.numel() != 0:
            sam_out = self.sam_predictor.predict(
                image_rgb, bboxes=xyxy_tensor, verbose=False
            )
            masks_tensor = sam_out[0].masks.data

            masks_np = masks_tensor.cpu().numpy()
        else:
            masks_np = np.empty((0, *image_rgb.shape[:2]), dtype=np.float64)

        # Create a detections object that we will save later
        curr_det = sv.Detections(
            xyxy=xyxy_np,
            confidence=confidences,
            class_id=detection_class_ids,
            mask=masks_np,
        )

        if len(curr_det) == 0:  # no detections, skip
            logging.debug("No detections in this frame")
            return image_rgb, [], None

        # filter the detection by removing overlapping detections
        curr_det, labels = filter_detections(
            image=image_rgb,
            detections=curr_det,
            classes=obj_classes,
            given_labels=detection_class_labels,
            iou_threshold=self.cfg_cg.object_detection_iou_threshold,
            min_mask_size_ratio=self.cfg_cg.min_mask_size_ratio,
            confidence_threshold=self.cfg_cg.object_detection_confidence_threshold,
        )
        if curr_det is None:
            logging.debug("No detections left after filter_detections")
            return image_rgb, [], None

        image_crops, image_feats, text_feats = compute_clip_features_batched(
            image_rgb,
            curr_det,
            self.clip_model,
            self.clip_preprocess,
            self.clip_tokenizer,
            obj_classes.get_classes_arr(),
            self.device,
        )

        raw_gobs = {
            # add new uuid for each detection
            "xyxy": curr_det.xyxy,
            "confidence": curr_det.confidence,
            "class_id": curr_det.class_id,
            "mask": curr_det.mask,
            "classes": obj_classes.get_classes_arr(),
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "detection_class_labels": detection_class_labels,
        }

        # resize the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # filter the observations
        filtered_gobs = filter_gobs(
            resized_gobs,
            image_rgb,
            skip_bg=self.cfg_cg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=self.cfg_cg.mask_area_threshold,
            max_bbox_area_ratio=self.cfg_cg.max_bbox_area_ratio,
            mask_conf_threshold=self.cfg_cg.mask_conf_threshold,
        )

        gobs = filtered_gobs

        if len(gobs["mask"]) == 0:  # no detections in this frame
            logging.debug("No detections left after filter_gobs")
            return image_rgb, [], None

        # this helps make sure things like pillows on couches are separate objects
        gobs["mask"] = mask_subtract_contained(gobs["xyxy"], gobs["mask"])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth,
            masks=gobs["mask"],
            cam_K=intrinsics[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=cam_pos,
            min_points_threshold=self.cfg_cg.min_points_threshold,
            spatial_sim_type=self.cfg_cg.spatial_sim_type,
            obj_pcd_max_points=self.cfg_cg.obj_pcd_max_points,
            device=self.device,
        )

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg_cg["dbscan_eps"],
                    dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                    pcd=obj["pcd"],
                )
        # if the list is all None, then skip
        if all([obj is None for obj in obj_pcds_and_bboxes]):
            logging.debug("All objects are None in obj_pcds_and_bboxes")
            return image_rgb, [], None

        # add pcds and bboxes to gobs
        gobs["bbox"] = [
            obj["bbox"] if obj is not None else None for obj in obj_pcds_and_bboxes
        ]
        gobs["pcd"] = [
            obj["pcd"] if obj is not None else None for obj in obj_pcds_and_bboxes
        ]

        # filter out objects that are far away
        gobs = self.filter_gobs_with_distance(pts, gobs)

        detection_list = self.make_detection_list_from_pcd_and_gobs(
            gobs, img_path, obj_classes
        )

        if len(detection_list) == 0:  # no detections, skip
            logging.debug(
                "No detections left after make_detection_list_from_pcd_and_gobs"
            )
            return image_rgb, [], None

        # compare the detections with the target object mask to see whether the target object is detected
        target_obj_id = None
        if (
            target_obj_mask is not None
            and np.sum(target_obj_mask)
            / (target_obj_mask.shape[0] * target_obj_mask.shape[1])
            > 0.0001
        ):
            assert len(detection_list) == len(
                gobs["mask"]
            ), f"Error in update_scene_graph: {len(detection_list)} != {len(gobs['mask'])}"  # sanity check

            # loop through the detected objects to find the highest IoU with the target object
            max_iou = -1
            max_iou_obj_id = None
            for idx, obj_id in enumerate(detection_list.keys()):
                detected_mask = gobs["mask"][idx]
                iou_score = IoU(detected_mask, target_obj_mask)
                if iou_score > max_iou:
                    max_iou = iou_score
                    max_iou_obj_id = obj_id
            if max_iou > self.cfg.scene_graph.target_obj_iou_threshold:
                target_obj_id = max_iou_obj_id
                logging.info(
                    f"Target object {target_obj_id} {detection_list[target_obj_id]['class_name']} detected with IoU {max_iou} in {img_path}!!!"
                )

        # if there exists object detected in this frame, create a snapshot
        frame = SnapShot(
            image=img_path,
            color=(random.random(), random.random(), random.random()),
            obs_point=pts_voxel,
        )
        # add all detected objects into the snapshot
        frame.full_obj_list = {
            obj_id: detection_list[obj_id]["conf"] for obj_id in detection_list.keys()
        }

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(self.objects) == 0:
            logging.debug(
                f"No objects in the map yet, adding all detections of length {len(detection_list)}"
            )
            self.objects.update(detection_list)

            self.frames[img_path] = frame

            annotated_image = image_rgb
            added_obj_ids = list(detection_list.keys())
        else:
            ### compute similarities and then merge
            spatial_sim = compute_spatial_similarities(
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                detection_list=detection_list,
                objects=self.objects,
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
            )

            visual_sim = compute_visual_similarities(detection_list, self.objects)

            agg_sim = aggregate_similarities(
                match_method=self.cfg_cg["match_method"],
                phys_bias=self.cfg_cg["phys_bias"],
                spatial_sim=spatial_sim,
                visual_sim=visual_sim,
            )

            # Perform matching of detections to existing objects
            match_indices = match_detections_to_objects(
                agg_sim=agg_sim,
                detection_threshold=self.cfg_cg[
                    "sim_threshold"
                ],  # Use the sim_threshold from the configuration
                existing_obj_ids=list(self.objects.keys()),
                detected_obj_ids=list(detection_list.keys()),
            )

            # Now merge the detected objects into the existing objects based on the match indices
            visualize_captions, target_obj_id, added_obj_ids = self.merge_obj_matches(
                detection_list=detection_list,
                match_indices=match_indices,
                obj_classes=obj_classes,
                snapshot=frame,
                target_obj_id=target_obj_id,
            )

            # add the snapshot into the snapshot list
            self.frames[img_path] = frame

            # create a Detection object for visualization
            det_visualize = sv.Detections(
                xyxy=gobs["xyxy"],
                confidence=gobs["confidence"],
                class_id=gobs["class_id"],
            )
            det_visualize.data["class_name"] = visualize_captions
            annotated_image = image_rgb.copy()
            BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
            LABEL_ANNOTATOR = sv.LabelAnnotator(
                text_thickness=1, text_scale=0.25, text_color=sv.Color.BLACK
            )
            annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(
                annotated_image, det_visualize
            )
            annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, det_visualize)

        return annotated_image, added_obj_ids, target_obj_id

    def filter_gobs_with_distance(self, pts, gobs):
        idx_to_keep = []
        for idx in range(len(gobs["bbox"])):
            if gobs["bbox"][idx] is None:  # point cloud was discarded
                continue

            # get the distance between the object and the current observation point
            if (
                np.linalg.norm(gobs["bbox"][idx].center[[0, 2]] - pts[[0, 2]])
                > self.cfg.scene_graph.obj_include_dist
            ):
                logging.debug(
                    f"Object {gobs['detection_class_labels'][idx]} is too far away, skipping"
                )
                continue
            idx_to_keep.append(idx)

        for attribute in gobs.keys():
            if isinstance(gobs[attribute], str) or attribute == "classes":  # Captions
                continue
            if attribute in ["labels", "edges", "text_feats", "captions"]:
                # Note: this statement was used to also exempt 'detection_class_labels' but that causes a bug. It causes the edges to be misalgined with the objects.
                continue
            elif isinstance(gobs[attribute], list):
                gobs[attribute] = [gobs[attribute][i] for i in idx_to_keep]
            elif isinstance(gobs[attribute], np.ndarray):
                gobs[attribute] = gobs[attribute][idx_to_keep]
            else:
                raise NotImplementedError(f"Unhandled type {type(gobs[attribute])}")

        return gobs

    def merge_obj_matches(
        self,
        detection_list: DetectionDict,
        match_indices: List[Tuple[int, Optional[int]]],
        obj_classes: ObjectClasses,
        snapshot: SnapShot,
        target_obj_id: Optional[
            int
        ] = None,  # if given, then track whether the target object is merged into a previous object (so the id would change)
    ) -> Tuple[List[str], Optional[int], List[int]]:
        visualize_captions = []
        added_obj_ids = []
        for idx, (detected_obj_id, existing_obj_match_id) in enumerate(match_indices):
            if existing_obj_match_id is None:
                self.objects[detected_obj_id] = detection_list[detected_obj_id]
                visualize_captions.append(
                    f"{detected_obj_id} {self.objects[detected_obj_id]['class_name']} {self.objects[detected_obj_id]['conf']:.3f} N"
                )
                added_obj_ids.append(detected_obj_id)
            else:
                # merge detected object into existing object
                detected_obj = detection_list[detected_obj_id]
                matched_obj = self.objects[existing_obj_match_id]

                merged_obj = merge_obj2_into_obj1(
                    obj1=matched_obj,
                    obj2=detected_obj,
                    downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg_cg["dbscan_eps"],
                    dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                    spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                    device=self.device,
                    run_dbscan=False,
                )
                # fix the class name by adopting the most popular class name
                class_id_counter = Counter(merged_obj["class_id"])
                most_common_class_id = class_id_counter.most_common(1)[0][0]
                most_common_class_name = obj_classes.get_classes_arr()[
                    most_common_class_id
                ]
                merged_obj["class_name"] = most_common_class_name

                # adjust the full detected list of the current snapshot: remove the detected object and add the merged object
                snapshot.full_obj_list[existing_obj_match_id] = detected_obj["conf"]
                snapshot.full_obj_list.pop(detected_obj_id)

                self.objects[existing_obj_match_id] = merged_obj
                visualize_captions.append(
                    f"{existing_obj_match_id} {self.objects[existing_obj_match_id]['class_name']} {detected_obj['conf']:.3f} {merged_obj['num_detections']}"
                )

                # if target object is merged, update target_obj_id
                if target_obj_id == detected_obj_id:
                    target_obj_id = existing_obj_match_id

        return visualize_captions, target_obj_id, added_obj_ids

    def make_detection_list_from_pcd_and_gobs(
        self, gobs, image_path, obj_classes
    ) -> DetectionDict:
        detection_list = DetectionDict()
        for mask_idx in range(len(gobs["mask"])):
            if gobs["pcd"][mask_idx] is None:  # point cloud was discarded
                continue

            curr_class_name = gobs["classes"][gobs["class_id"][mask_idx]]
            curr_class_idx = obj_classes.get_classes_arr().index(curr_class_name)

            detected_object = {
                "id": self.object_id_counter,  # unique id for this object
                "class_name": curr_class_name,  # global class id for this detection
                "class_id": [curr_class_idx],  # global class id for this detection
                "num_detections": 1,  # number of detections in this object
                "conf": gobs["confidence"][mask_idx],
                # These are for the entire 3D object
                "pcd": gobs["pcd"][mask_idx],
                "bbox": gobs["bbox"][mask_idx],
                "clip_ft": to_tensor(gobs["image_feats"][mask_idx]),
                # the snapshot name it belongs to
                "image": None,
            }

            detection_list[self.object_id_counter] = detected_object
            self.object_id_counter += 1

        return detection_list

    def cleanup_empty_frames_snapshots(self):
        # remove the frame that have empty detected objects
        filtered_frames = {}
        for file_name, frame in self.frames.items():
            if len(frame.full_obj_list) > 0:
                filtered_frames[file_name] = frame
        self.frames = filtered_frames

        # remove the snapshots that have no cluster
        filtered_snapshots = {}
        for file_name, snapshot in self.snapshots.items():
            if len(snapshot.cluster) > 0:
                filtered_snapshots[file_name] = snapshot
        self.snapshots = filtered_snapshots

    def update_snapshots(
        self,
        obj_ids,
        min_detection=2,
    ):
        self.cleanup_empty_frames_snapshots()

        prev_snapshots = copy.deepcopy(self.snapshots)

        obj_ids_temp = obj_ids.copy()
        for filename, snapshot in self.snapshots.items():
            cluster = snapshot.cluster
            if any([obj_id in obj_ids_temp for obj_id in cluster]):
                obj_ids = obj_ids.union(set(cluster))
                prev_snapshots.pop(filename)
        obj_ids = list(set(obj_ids))

        # find and exclude the objects that have only one observation
        obj_exclude = [
            obj_id
            for obj_id in self.objects.keys()
            if self.objects[obj_id]["num_detections"] < min_detection
        ]
        obj_ids = [obj_id for obj_id in obj_ids if obj_id not in obj_exclude]

        obj_centers = np.zeros((len(obj_ids), 2))
        for i, obj_id in enumerate(obj_ids):
            obj_centers[i] = self.objects[obj_id]["bbox"].center[[0, 2]]

        if len(obj_centers) == 0:
            return

        new_snapshots = self.clustering.fit(obj_centers, obj_ids, self.frames)

        # Objects that clustering could not assign to any snapshot (no observed frame found).
        # This can happen when periodic_cleanup_objects removed the frames that originally
        # observed these objects.  Demote them so they are treated like low-detection objects
        # and will be re-clustered once observed again.
        new_snapshot_obj_ids = set(
            obj_id
            for snapshot in new_snapshots.values()
            for obj_id in snapshot.cluster
        )
        unassigned = [oid for oid in obj_ids if oid not in new_snapshot_obj_ids]
        for oid in unassigned:
            logging.warning(
                f"update_snapshots: object {oid} ({self.objects[oid]['class_name']}) "
                f"could not be assigned to any snapshot frame — demoting num_detections to 1."
            )
            self.objects[oid]["num_detections"] = 1
            self.objects[oid]["image"] = None
        # Re-derive obj_ids / obj_exclude after demotion
        obj_ids = [oid for oid in obj_ids if oid in new_snapshot_obj_ids]
        obj_exclude = [
            obj_id
            for obj_id in self.objects.keys()
            if self.objects[obj_id]["num_detections"] < min_detection
        ]

        prev_snapshot_objs = [
            obj_id
            for snapshot in prev_snapshots.values()
            for obj_id in snapshot.cluster
        ]
        assert set(
            [
                obj_id
                for snapshot in new_snapshots.values()
                for obj_id in snapshot.cluster
            ]
        ) == set(
            obj_ids
        ), f"{set([obj_id for snapshot in new_snapshots.values() for obj_id in snapshot.cluster])} != {set(obj_ids)}"
        assert (
            set(obj_ids) & set(prev_snapshot_objs)
        ) == set(), f"{set(obj_ids)} & {set(prev_snapshot_objs)} != empty"
        assert (set(obj_ids) | set(prev_snapshot_objs) | set(obj_exclude)) == set(
            self.objects.keys()
        ), f"{set(obj_ids)} | {set(prev_snapshot_objs)} | {set(obj_exclude)} != {set(self.objects.keys())}"

        for key, snapshot in new_snapshots.items():
            if key in prev_snapshots.keys():
                prev_snapshots[key].cluster += snapshot.cluster
            else:
                prev_snapshots[key] = snapshot
        self.snapshots = prev_snapshots

        # update the snapshot belonging of each object
        for file_name, snapshot in self.snapshots.items():
            for obj_id in snapshot.cluster:
                self.objects[obj_id]["image"] = file_name

        # remove the duplicates caused by copying snapshots: self.frames and self.snapshots should point to the same object
        for file_name, snapshot in self.snapshots.items():
            self.frames[file_name] = snapshot

        # sanity check
        for obj_id, obj in self.objects.items():
            if obj["num_detections"] < min_detection:
                assert (
                    obj["image"] is None
                ), f"{obj_id} has only one detection but has image"
            else:
                assert obj["image"] is not None, f"{obj_id} has no image"

    def periodic_cleanup_objects(self, frame_idx, pts):
        ### Perform post-processing periodically if told so

        # Denoising
        if processing_needed(
            self.cfg_cg["denoise_interval"],
            self.cfg_cg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame=False,
        ):
            self.objects = measure_time(denoise_objects)(
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=self.cfg_cg["dbscan_eps"],
                dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                device=self.device,
                objects=self.objects,
            )

        # Filtering
        if processing_needed(
            self.cfg_cg["filter_interval"],
            self.cfg_cg["run_filter_final_frame"],
            frame_idx,
            is_final_frame=False,
        ):
            self.objects = filter_objects(
                obj_min_points=self.cfg_cg["obj_min_points"],
                obj_min_detections=self.cfg_cg["obj_min_detections"],
                min_distance=self.cfg.scene_graph.obj_include_dist,
                objects=self.objects,
                pts=pts,
            )

        # temporarily we do not merge close objects, since handling which snapshot the merged object belongs to is a bit tricky

        # Merging
        if processing_needed(
            self.cfg_cg["merge_interval"],
            self.cfg_cg["run_merge_final_frame"],
            frame_idx,
            is_final_frame=False,
        ):
            self.objects = measure_time(merge_objects)(
                merge_overlap_thresh=self.cfg_cg["merge_overlap_thresh"],
                merge_visual_sim_thresh=self.cfg_cg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=self.cfg_cg["merge_text_sim_thresh"],
                objects=self.objects,
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=self.cfg_cg["dbscan_eps"],
                dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                device=self.device,
            )

        # update the object list in snapshots, since some objects may have been removed
        frame_to_pop = []
        for (
            filename,
            ss,
        ) in (
            self.frames.items()
        ):  # TODO: check whether content in snapshots are also changed, and see whether need to remove snapshot that have empty cluster
            ss.cluster = [
                obj_id for obj_id in ss.cluster if obj_id in self.objects.keys()
            ]
            ss.full_obj_list = {
                obj_id: conf
                for obj_id, conf in ss.full_obj_list.items()
                if obj_id in self.objects.keys()
            }
            if len(ss.full_obj_list) == 0:
                frame_to_pop.append(filename)
        for filename in frame_to_pop:
            self.frames.pop(filename)

    def sanity_check(self, cfg):
        obj_exclude_count = sum(
            [
                1 if obj["num_detections"] < cfg.min_detection else 0
                for obj in self.objects.values()
            ]
        )
        total_objs_count = sum(
            [len(snapshot.cluster) for snapshot in self.snapshots.values()]
        )
        assert (
            len(self.objects) == total_objs_count + obj_exclude_count
        ), f"{len(self.objects)} != {total_objs_count} + {obj_exclude_count}"
        total_objs_count = sum(
            [len(set(snapshot.cluster)) for snapshot in self.snapshots.values()]
        )
        assert (
            len(self.objects) == total_objs_count + obj_exclude_count
        ), f"{len(self.objects)} != {total_objs_count} + {obj_exclude_count}"
        for obj_id in self.objects.keys():
            exist_count = 0
            for ss in self.snapshots.values():
                if obj_id in ss.cluster:
                    exist_count += 1
            if self.objects[obj_id]["num_detections"] < cfg.min_detection:
                assert (
                    exist_count == 0
                ), f"{exist_count} != 0 for obj_id {obj_id}, {self.objects[obj_id]['class_name']}"
            else:
                assert (
                    exist_count == 1
                ), f"{exist_count} != 1 for obj_id {obj_id}, {self.objects[obj_id]['class_name']}"
        for ss in self.snapshots.values():
            assert len(ss.cluster) == len(
                set(ss.cluster)
            ), f"{ss.cluster} has duplicates"
            assert len(ss.full_obj_list.keys()) == len(
                set(ss.full_obj_list.keys())
            ), f"{ss.full_obj_list.keys()} has duplicates"
            for obj_id in ss.cluster:
                assert (
                    obj_id in ss.full_obj_list
                ), f"{obj_id} not in {ss.full_obj_list.keys()}"
            for obj_id in ss.full_obj_list.keys():
                assert obj_id in self.objects, f"{obj_id} not in scene objects"
        # check whether the snapshots in scene.snapshots and scene.frames are the same
        for file_name, ss in self.snapshots.items():
            assert (
                ss.cluster == self.frames[file_name].cluster
            ), f"{ss}\n!=\n{self.frames[file_name]}"
            assert (
                ss.full_obj_list == self.frames[file_name].full_obj_list
            ), f"{ss}\n==\n{self.frames[file_name]}"

    def print_scene_graph(self):
        snapshot_dict = {}
        for obj_id, obj in self.objects.items():
            if obj["image"] not in snapshot_dict:
                snapshot_dict[obj["image"]] = []
            snapshot_dict[obj["image"]].append(
                f"{obj_id}: {obj['class_name']} {obj['num_detections']}"
            )
        for snapshot_id, obj_list in snapshot_dict.items():
            logging.info(f"{snapshot_id}:")
            for obj_str in obj_list:
                logging.info(f"\t{obj_str}")

    def get_topdown_map(self, colorize: bool = True, meters_per_pixel: float = 0.025):
        """
        Return a top-down map for the current scene with real texture rendering.
        
        Uses Habitat's orthographic camera to render a true top-down textured view,
        showing actual furniture (sofas, beds, tables, etc.) from above.

        Args:
            colorize: If True, return RGB colorized map; if False, return binary occupancy
            meters_per_pixel: Resolution of the topdown map (default 0.025m = 2.5cm per pixel)

        Returns:
            (rgb_map, map_bounds)
            - rgb_map: HxWx3 uint8 RGB array with real scene textures
            - map_bounds: (min_bound, max_bound) as two (3,) numpy arrays, or None
        """
        import cv2
        
        # Try to get bounds from pathfinder
        map_bounds = None
        try:
            if hasattr(self, "pathfinder") and self.pathfinder is not None:
                bounds = self.pathfinder.get_bounds()
                if bounds is not None:
                    map_bounds = (np.array(bounds[0]), np.array(bounds[1]))
        except Exception:
            map_bounds = None

        if map_bounds is None:
            return None, None

        min_bound, max_bound = map_bounds
        
        # Compute map dimensions
        x_range = max_bound[0] - min_bound[0]
        z_range = max_bound[2] - min_bound[2]
        
        map_width = int(np.ceil(x_range / meters_per_pixel))
        map_height = int(np.ceil(z_range / meters_per_pixel))
        
        # Clamp to reasonable size
        max_dim = 2000
        if map_width > max_dim or map_height > max_dim:
            scale = max_dim / max(map_width, map_height)
            map_width = int(map_width * scale)
            map_height = int(map_height * scale)
            meters_per_pixel = max(x_range, z_range) / max_dim

        # ========== Method 1: Try to render using orthographic top-down camera ==========
        try:
            rgb_textured = self._render_topdown_textured(
                min_bound, max_bound, map_width, map_height, meters_per_pixel
            )
            if rgb_textured is not None:
                return rgb_textured, map_bounds
        except Exception as e:
            logging.debug(f"Textured topdown rendering failed: {e}")

        # ========== Method 2: Fallback to navmesh-based occupancy map ==========
        topdown = np.zeros((map_height, map_width), dtype=np.uint8)
        
        try:
            # Get navmesh vertices and create navigable area
            if hasattr(self, "pathfinder") and self.pathfinder is not None:
                navmesh_vertices = self.pathfinder.build_navmesh_vertices()
                navmesh_indices = self.pathfinder.build_navmesh_vertex_indices()
                
                if navmesh_vertices and navmesh_indices:
                    # Convert vertices to numpy array
                    vertices = np.array([np.array(v).flatten() for v in navmesh_vertices])
                    
                    # Draw triangles on topdown map
                    for i in range(0, len(navmesh_indices), 3):
                        if i + 2 < len(navmesh_indices):
                            idx0, idx1, idx2 = navmesh_indices[i], navmesh_indices[i+1], navmesh_indices[i+2]
                            if idx0 < len(vertices) and idx1 < len(vertices) and idx2 < len(vertices):
                                v0, v1, v2 = vertices[idx0], vertices[idx1], vertices[idx2]
                                
                                # Project to 2D (XZ plane)
                                pts = np.array([
                                    [(v0[0] - min_bound[0]) / meters_per_pixel, (v0[2] - min_bound[2]) / meters_per_pixel],
                                    [(v1[0] - min_bound[0]) / meters_per_pixel, (v1[2] - min_bound[2]) / meters_per_pixel],
                                    [(v2[0] - min_bound[0]) / meters_per_pixel, (v2[2] - min_bound[2]) / meters_per_pixel],
                                ], dtype=np.int32)
                                
                                # Clamp to map bounds
                                pts[:, 0] = np.clip(pts[:, 0], 0, map_width - 1)
                                pts[:, 1] = np.clip(pts[:, 1], 0, map_height - 1)
                                
                                # Fill triangle
                                cv2.fillPoly(topdown, [pts], 255)
        except Exception as e:
            logging.debug(f"Failed to build navmesh topdown: {e}")
            # If navmesh approach fails, try sampling-based approach
            try:
                if hasattr(self, "pathfinder") and self.pathfinder is not None:
                    # Sample points and mark navigable areas
                    for _ in range(10000):
                        pt = self.pathfinder.get_random_navigable_point()
                        if pt is not None:
                            px = int((pt[0] - min_bound[0]) / meters_per_pixel)
                            pz = int((pt[2] - min_bound[2]) / meters_per_pixel)
                            if 0 <= px < map_width and 0 <= pz < map_height:
                                # Mark a small area around the point
                                for dx in range(-2, 3):
                                    for dz in range(-2, 3):
                                        npx, npz = px + dx, pz + dz
                                        if 0 <= npx < map_width and 0 <= npz < map_height:
                                            topdown[npz, npx] = 255
            except Exception:
                pass

        # Colorize
        if colorize:
            # Create RGB: navigable=light gray, obstacle=dark
            rgb = np.zeros((map_height, map_width, 3), dtype=np.uint8)
            rgb[topdown > 0] = [200, 200, 200]  # navigable: light gray
            rgb[topdown == 0] = [50, 50, 50]     # obstacle: dark gray
        else:
            rgb = topdown

        return rgb, map_bounds

    def _render_topdown_textured(
        self,
        min_bound: np.ndarray,
        max_bound: np.ndarray,
        map_width: int,
        map_height: int,
        meters_per_pixel: float,
    ) -> Optional[np.ndarray]:
        """
        Render a top-down textured view by rendering from multiple low-altitude viewpoints.
        
        Instead of looking straight down (which only shows floor), this method 
        renders from multiple angled viewpoints at furniture height to capture
        the actual appearance of objects.
        
        Args:
            min_bound: Scene minimum bounds (3,)
            max_bound: Scene maximum bounds (3,)
            map_width: Output map width in pixels
            map_height: Output map height in pixels
            meters_per_pixel: Resolution
            
        Returns:
            RGB image (H, W, 3) uint8 or None if failed
        """
        import habitat_sim
        import cv2
        from scipy.spatial.transform import Rotation as R
        
        try:
            # Get the agent
            agent = self.simulator.get_agent(0)
            agent_state = agent.get_state()
            
            # Scene dimensions
            x_range = max_bound[0] - min_bound[0]
            z_range = max_bound[2] - min_bound[2]
            y_floor = min_bound[1]
            
            # Create output canvas
            output = np.ones((map_height, map_width, 3), dtype=np.uint8) * 240  # Light gray background
            
            # Rotation for looking straight down
            rot_down = R.from_euler('x', -90, degrees=True)
            cam_rotation_down = rot_down.as_quat()
            cam_rotation_habitat = np.quaternion(
                cam_rotation_down[3], cam_rotation_down[0], cam_rotation_down[1], cam_rotation_down[2]
            )
            
            # Get camera FOV
            sensor_hfov = self.cfg.hfov if hasattr(self.cfg, 'hfov') else 90
            fov_rad = np.radians(sensor_hfov)
            
            # Render from low height to see furniture tops (not just floor)
            # Camera at ~2m height looking down covers furniture
            cam_height = y_floor + 2.5  # 2.5m above floor - sees furniture tops
            
            # Calculate ground coverage at this height
            effective_height = cam_height - y_floor
            ground_coverage = 2 * effective_height * np.tan(fov_rad / 2)
            
            # Tile step with overlap
            step = ground_coverage * 0.4  # 60% overlap for smooth blending
            
            n_tiles_x = max(1, int(np.ceil(x_range / step)) + 2)
            n_tiles_z = max(1, int(np.ceil(z_range / step)) + 2)
            
            # Weight map for blending
            weight_map = np.zeros((map_height, map_width), dtype=np.float32)
            color_accum = np.zeros((map_height, map_width, 3), dtype=np.float32)
            
            for ti in range(n_tiles_z):
                for tj in range(n_tiles_x):
                    tile_center_x = min_bound[0] - step + tj * step
                    tile_center_z = min_bound[2] - step + ti * step
                    
                    # Camera position
                    cam_position = np.array([tile_center_x, cam_height, tile_center_z])
                    
                    # Set agent state
                    new_state = habitat_sim.AgentState()
                    new_state.position = cam_position
                    new_state.rotation = cam_rotation_habitat
                    agent.set_state(new_state)
                    
                    # Render
                    obs = self.simulator.get_sensor_observations()
                    if "color_sensor" not in obs:
                        continue
                    
                    rgb_tile = obs["color_sensor"]
                    if rgb_tile.shape[-1] == 4:
                        rgb_tile = rgb_tile[:, :, :3]
                    
                    h, w = rgb_tile.shape[:2]
                    
                    # Map each rendered pixel to output coordinates
                    half_cov = ground_coverage / 2
                    
                    # Crop center to reduce distortion
                    margin = 0.15
                    y1, y2 = int(h * margin), int(h * (1 - margin))
                    x1, x2 = int(w * margin), int(w * (1 - margin))
                    cropped = rgb_tile[y1:y2, x1:x2]
                    ch, cw = cropped.shape[:2]
                    
                    # Coverage after crop
                    crop_cov = ground_coverage * (1 - 2 * margin)
                    half_crop = crop_cov / 2
                    
                    # Output coordinates
                    px_x1 = int((tile_center_x - half_crop - min_bound[0]) / meters_per_pixel)
                    px_z1 = int((tile_center_z - half_crop - min_bound[2]) / meters_per_pixel)
                    px_x2 = int((tile_center_x + half_crop - min_bound[0]) / meters_per_pixel)
                    px_z2 = int((tile_center_z + half_crop - min_bound[2]) / meters_per_pixel)
                    
                    # Clamp
                    out_x1 = max(0, px_x1)
                    out_z1 = max(0, px_z1)
                    out_x2 = min(map_width, px_x2)
                    out_z2 = min(map_height, px_z2)
                    
                    if out_x2 <= out_x1 or out_z2 <= out_z1:
                        continue
                    
                    # Resize cropped tile to output region size
                    out_w = out_x2 - out_x1
                    out_h = out_z2 - out_z1
                    
                    # Calculate corresponding source region
                    if px_x2 != px_x1 and px_z2 != px_z1:
                        src_x1 = int((out_x1 - px_x1) / (px_x2 - px_x1) * cw)
                        src_z1 = int((out_z1 - px_z1) / (px_z2 - px_z1) * ch)
                        src_x2 = int((out_x2 - px_x1) / (px_x2 - px_x1) * cw)
                        src_z2 = int((out_z2 - px_z1) / (px_z2 - px_z1) * ch)
                    else:
                        continue
                    
                    src_x1, src_x2 = max(0, src_x1), min(cw, src_x2)
                    src_z1, src_z2 = max(0, src_z1), min(ch, src_z2)
                    
                    if src_x2 <= src_x1 or src_z2 <= src_z1:
                        continue
                    
                    src_region = cropped[src_z1:src_z2, src_x1:src_x2]
                    if src_region.shape[0] < 2 or src_region.shape[1] < 2:
                        continue
                    
                    resized = cv2.resize(src_region, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Create center-weighted mask for blending
                    y_coords = np.linspace(-1, 1, out_h)
                    x_coords = np.linspace(-1, 1, out_w)
                    xx, yy = np.meshgrid(x_coords, y_coords)
                    # Gaussian-like center weight
                    blend_mask = np.exp(-(xx**2 + yy**2) / 0.5)
                    
                    # Accumulate
                    color_accum[out_z1:out_z2, out_x1:out_x2] += resized.astype(np.float32) * blend_mask[:, :, np.newaxis]
                    weight_map[out_z1:out_z2, out_x1:out_x2] += blend_mask
            
            # Normalize by weights
            valid = weight_map > 0
            for c in range(3):
                output[:, :, c] = np.where(
                    valid,
                    (color_accum[:, :, c] / weight_map).astype(np.uint8),
                    output[:, :, c]
                )
            
            # Restore agent state
            agent.set_state(agent_state)
            
            # Check if we rendered anything
            if not valid.any():
                logging.warning("Top-down rendering produced empty image, using fallback")
                # 不返回 None，而是返回带有占用信息的备用图像
                return self._create_fallback_topdown_map(min_bound, max_bound, map_width, map_height, meters_per_pixel)
            
            # ==================== 学术抽象风格后处理 ====================
            # 将照片级拼接底图转换为学术论文风格的结构底图
            output = self._apply_academic_style_postprocess(output, valid)
            
            logging.info(f"[_render_topdown_textured] Successfully rendered textured map: shape={output.shape}")
            return output
            
        except Exception as e:
            logging.warning(f"Top-down textured rendering failed: {e}")
            import traceback
            traceback.print_exc()
            # 返回备用底图而不是 None
            return self._create_fallback_topdown_map(min_bound, max_bound, map_width, map_height, meters_per_pixel)

    def _create_fallback_topdown_map(
        self,
        min_bound: np.ndarray,
        max_bound: np.ndarray,
        map_width: int,
        map_height: int,
        meters_per_pixel: float,
    ) -> np.ndarray:
        """
        创建备用的学术风格俯视底图（基于 navmesh/pathfinder）
        
        当纹理渲染失败时调用，确保始终返回有效的底图。
        
        Returns:
            RGB 图像 (H, W, 3) uint8
        """
        import cv2
        
        logging.info("[_create_fallback_topdown_map] Creating fallback map from navmesh...")
        
        # 创建输出画布 - 浅灰色背景
        output = np.ones((map_height, map_width, 3), dtype=np.uint8) * 220
        
        try:
            # 从 pathfinder 获取导航网格
            if hasattr(self, 'pathfinder') and self.pathfinder is not None:
                # 尝试从 navmesh 顶点构建占用栅格
                navmesh_vertices = self.pathfinder.build_navmesh_vertices()
                navmesh_indices = self.pathfinder.build_navmesh_vertex_indices()
                
                if navmesh_vertices and navmesh_indices:
                    # 创建占用掩码
                    occupancy = np.zeros((map_height, map_width), dtype=np.uint8)
                    vertices = np.array([np.array(v).flatten() for v in navmesh_vertices])
                    
                    # 绘制导航三角形
                    for i in range(0, len(navmesh_indices), 3):
                        if i + 2 < len(navmesh_indices):
                            idx0, idx1, idx2 = navmesh_indices[i], navmesh_indices[i+1], navmesh_indices[i+2]
                            if idx0 < len(vertices) and idx1 < len(vertices) and idx2 < len(vertices):
                                v0, v1, v2 = vertices[idx0], vertices[idx1], vertices[idx2]
                                pts = np.array([
                                    [(v0[0] - min_bound[0]) / meters_per_pixel, 
                                     (v0[2] - min_bound[2]) / meters_per_pixel],
                                    [(v1[0] - min_bound[0]) / meters_per_pixel, 
                                     (v1[2] - min_bound[2]) / meters_per_pixel],
                                    [(v2[0] - min_bound[0]) / meters_per_pixel, 
                                     (v2[2] - min_bound[2]) / meters_per_pixel],
                                ], dtype=np.int32)
                                pts[:, 0] = np.clip(pts[:, 0], 0, map_width - 1)
                                pts[:, 1] = np.clip(pts[:, 1], 0, map_height - 1)
                                cv2.fillPoly(occupancy, [pts], 255)
                    
                    # 可导航区域用浅色
                    output[occupancy > 0] = [245, 245, 245]
                    
                    # 提取并绘制边界
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    boundary = cv2.morphologyEx(occupancy, cv2.MORPH_GRADIENT, kernel)
                    
                    # 加粗边界
                    boundary_thick = cv2.dilate(boundary, kernel, iterations=1)
                    
                    # 深灰色边界
                    output[boundary_thick > 0] = [80, 80, 80]
                    
                    logging.info(f"[_create_fallback_topdown_map] Created fallback map with navmesh, shape={output.shape}")
                    return output
        except Exception as e:
            logging.warning(f"[_create_fallback_topdown_map] Navmesh fallback failed: {e}")
        
        # 最终备用：纯色背景
        logging.warning("[_create_fallback_topdown_map] Returning plain gray background")
        return output

    def _apply_academic_style_postprocess(
        self, 
        image: np.ndarray, 
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        将照片级拼接底图转换为学术抽象风格的结构底图
        
        实现"去感知化（de-perceptualization）"：
        1. 灰度化与高斯去噪 - 消除纹理噪声，呈现云雾状空间暗示
        2. 对比度压缩与亮度提升 - 整体偏向浅灰色，不干扰上层节点
        3. 边缘检测增强轮廓 - Floorplan-like 效果，暗示墙壁和房间边界
        
        Args:
            image: RGB 图像 (H, W, 3) uint8
            valid_mask: 有效区域掩码 (H, W) bool
            
        Returns:
            处理后的 RGB 图像 (H, W, 3) uint8
        """
        import cv2
        
        # 保存原始图像用于边缘提取
        original = image.copy()
        
        # ========== Step 1: 灰度化与高斯去噪 ==========
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 使用较大核的高斯模糊，消除拼接锐利边缘和纹理噪声
        # 核大小根据图像尺寸动态调整
        blur_kernel_size = max(15, min(image.shape[0], image.shape[1]) // 30)
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1  # 确保是奇数
        
        gray_blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
        
        # 再进行一次更大核的模糊，增强"云雾状"效果
        cloud_kernel = blur_kernel_size * 2 + 1
        gray_cloud = cv2.GaussianBlur(gray_blurred, (cloud_kernel, cloud_kernel), 0)
        
        # 混合两个模糊层，保留一定的空间结构
        gray_final = cv2.addWeighted(gray_blurred, 0.4, gray_cloud, 0.6, 0)
        
        # ========== Step 2: 对比度压缩与亮度提升 ==========
        # 压缩动态范围，映射到 [180, 240] 区间（浅灰色）
        # 使低对比度区域更加统一，高对比度区域被压缩
        min_val = gray_final.min()
        max_val = gray_final.max()
        
        if max_val > min_val:
            # 归一化到 [0, 1]
            normalized = (gray_final.astype(np.float32) - min_val) / (max_val - min_val)
            # 应用 gamma 校正压缩对比度
            gamma = 0.6  # < 1 会压缩高亮区域，提升暗部
            normalized = np.power(normalized, gamma)
            # 映射到目标范围 [180, 240]
            target_min, target_max = 185, 235
            gray_toned = (normalized * (target_max - target_min) + target_min).astype(np.uint8)
        else:
            gray_toned = np.full_like(gray_final, 210)  # 统一浅灰色
        
        # ========== Step 3: 边缘检测增强轮廓 ==========
        # 使用 Canny 提取原始图像的边缘轮廓
        # 先对原图进行适度模糊以减少噪声边缘
        edge_blur = cv2.GaussianBlur(
            cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), 
            (5, 5), 0
        )
        
        # Canny 边缘检测（阈值设置较高，只提取显著边缘）
        edges = cv2.Canny(edge_blur, 30, 100)
        
        # 对边缘进行轻微膨胀，使线条更连续
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # 对边缘进行模糊，使其更柔和
        edges_soft = cv2.GaussianBlur(edges_dilated.astype(np.float32), (3, 3), 0)
        edges_soft = (edges_soft / edges_soft.max() * 255).astype(np.uint8) if edges_soft.max() > 0 else edges_soft
        
        # 将边缘以极浅的颜色叠加（边缘区域略微变暗）
        # 边缘强度映射到 [0, 25] 的减量
        edge_darkening = (edges_soft.astype(np.float32) / 255.0 * 20).astype(np.uint8)
        gray_with_edges = np.clip(gray_toned.astype(np.int16) - edge_darkening, 160, 240).astype(np.uint8)
        
        # ========== Step 4: 处理无效区域 ==========
        # 无效区域（未渲染）使用更深的灰色
        invalid_color = 140
        gray_with_edges = np.where(valid_mask, gray_with_edges, invalid_color)
        
        # ========== Step 5: 转换回 RGB 格式 ==========
        # 创建输出图像（保持 RGB 格式兼容性）
        output_rgb = np.stack([gray_with_edges, gray_with_edges, gray_with_edges], axis=-1)
        
        # 可选：给有效区域添加极微弱的暖色调，增加空间感
        # 这会使底图呈现非常淡的米色/暖灰色
        warmth = np.zeros_like(output_rgb, dtype=np.float32)
        warmth[:, :, 0] = 3   # R 通道轻微增加
        warmth[:, :, 1] = 1   # G 通道轻微增加
        warmth[:, :, 2] = -2  # B 通道轻微减少
        
        output_rgb = np.clip(
            output_rgb.astype(np.float32) + warmth * valid_mask[:, :, np.newaxis], 
            0, 255
        ).astype(np.uint8)
        
        return output_rgb

    # ==================== 新增：纯几何学术底图 ====================
    
    def get_structured_topdown_map(
        self, 
        meters_per_pixel: float = 0.025,
        style: str = "academic",
        enable_distance_shading: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]], float]:
        """
        生成纯几何结构的学术级俯视底图（Academic Structured Top-down Map）
        
        【核心设计理念】
        完全放弃 RGB 纹理拼接，仅从 navmesh/pathfinder 数据生成干净的结构底图：
        - Navigable Area（可导航区域）: 浅灰色连续区域
        - Obstacle/Wall Boundary（障碍/墙壁轮廓）: 深色边界线
        - Unexplored Region（未探索区域）: 统一背景色
        
        【视觉规范】
        - 低信息密度，高结构清晰度
        - 适合直接用于学术论文（非调试截图）
        - 所有叠加层（轨迹、节点）必须严格按 meters_per_pixel 对齐
        
        Args:
            meters_per_pixel: 分辨率（默认 0.025m = 2.5cm/pixel）
            style: 风格选择
                - "academic": 极简学术风格（白底 + 灰色导航区 + 深色边界）
                - "blueprint": 蓝图风格（深蓝背景 + 浅蓝导航区）
                - "grayscale": 纯灰度风格
            enable_distance_shading: 是否启用距离变换着色（为导航区域添加柔和空间梯度）
            
        Returns:
            (rgb_map, map_bounds)
            - rgb_map: (H, W, 3) np.uint8 RGB 图像
            - map_bounds: (min_bound, max_bound) 各为 (3,) numpy 数组，或 None
            - meters_per_pixel: 实际使用的分辨率（可能因尺寸限制被调整）
        """
        import cv2
        from scipy import ndimage
        
        # ==================== Step 1: 获取场景边界 ====================
        map_bounds = None
        try:
            if hasattr(self, "pathfinder") and self.pathfinder is not None:
                bounds = self.pathfinder.get_bounds()
                if bounds is not None:
                    map_bounds = (np.array(bounds[0]), np.array(bounds[1]))
        except Exception:
            map_bounds = None

        if map_bounds is None:
            logging.warning("Failed to get pathfinder bounds for structured map")
            return None, None, meters_per_pixel

        min_bound, max_bound = map_bounds
        
        # ==================== Step 2: 计算地图尺寸 ====================
        x_range = max_bound[0] - min_bound[0]
        z_range = max_bound[2] - min_bound[2]
        
        map_width = int(np.ceil(x_range / meters_per_pixel))
        map_height = int(np.ceil(z_range / meters_per_pixel))
        
        # 限制最大尺寸，避免内存爆炸
        max_dim = 3000
        if map_width > max_dim or map_height > max_dim:
            scale = max_dim / max(map_width, map_height)
            map_width = int(map_width * scale)
            map_height = int(map_height * scale)
            meters_per_pixel = max(x_range, z_range) / max_dim
            logging.info(f"Map size clamped to {map_width}x{map_height}, mpp={meters_per_pixel:.4f}")

        # ==================== Step 3: 从 Navmesh 构建占用栅格 ====================
        # occupancy: 0 = 障碍/未知, 255 = 可导航
        occupancy = np.zeros((map_height, map_width), dtype=np.uint8)
        
        navmesh_available = False
        try:
            if hasattr(self, "pathfinder") and self.pathfinder is not None:
                navmesh_vertices = self.pathfinder.build_navmesh_vertices()
                navmesh_indices = self.pathfinder.build_navmesh_vertex_indices()
                
                if navmesh_vertices and navmesh_indices:
                    navmesh_available = True
                    # 将顶点转换为 numpy 数组
                    vertices = np.array([np.array(v).flatten() for v in navmesh_vertices])
                    
                    # 绘制每个三角形
                    for i in range(0, len(navmesh_indices), 3):
                        if i + 2 < len(navmesh_indices):
                            idx0, idx1, idx2 = navmesh_indices[i], navmesh_indices[i+1], navmesh_indices[i+2]
                            if idx0 < len(vertices) and idx1 < len(vertices) and idx2 < len(vertices):
                                v0, v1, v2 = vertices[idx0], vertices[idx1], vertices[idx2]
                                
                                # 投影到 XZ 平面并转换为像素坐标
                                pts = np.array([
                                    [(v0[0] - min_bound[0]) / meters_per_pixel, 
                                     (v0[2] - min_bound[2]) / meters_per_pixel],
                                    [(v1[0] - min_bound[0]) / meters_per_pixel, 
                                     (v1[2] - min_bound[2]) / meters_per_pixel],
                                    [(v2[0] - min_bound[0]) / meters_per_pixel, 
                                     (v2[2] - min_bound[2]) / meters_per_pixel],
                                ], dtype=np.int32)
                                
                                # 裁剪到地图边界
                                pts[:, 0] = np.clip(pts[:, 0], 0, map_width - 1)
                                pts[:, 1] = np.clip(pts[:, 1], 0, map_height - 1)
                                
                                # 填充三角形
                                cv2.fillPoly(occupancy, [pts], 255)
                    
                    logging.info(f"Built navmesh occupancy: {np.sum(occupancy > 0)} navigable pixels")
        except Exception as e:
            logging.debug(f"Failed to build navmesh occupancy: {e}")
        
        # 如果 navmesh 方法失败，尝试采样方法
        if not navmesh_available or np.sum(occupancy > 0) < 100:
            logging.info("Navmesh unavailable, falling back to sampling-based occupancy")
            try:
                if hasattr(self, "pathfinder") and self.pathfinder is not None:
                    # 采样大量点以构建占用图
                    n_samples = min(50000, map_width * map_height // 4)
                    for _ in range(n_samples):
                        pt = self.pathfinder.get_random_navigable_point()
                        if pt is not None:
                            px = int((pt[0] - min_bound[0]) / meters_per_pixel)
                            pz = int((pt[2] - min_bound[2]) / meters_per_pixel)
                            if 0 <= px < map_width and 0 <= pz < map_height:
                                # 标记邻域（因为采样点可能稀疏）
                                radius = max(1, int(0.1 / meters_per_pixel))  # 10cm 半径
                                cv2.circle(occupancy, (px, pz), radius, 255, -1)
                    
                    # 使用形态学操作填充小空洞
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)
                    
                    logging.info(f"Built sampling-based occupancy: {np.sum(occupancy > 0)} navigable pixels")
            except Exception as e:
                logging.warning(f"Sampling-based occupancy also failed: {e}")

        # ==================== Step 4: 提取边界轮廓 ====================
        # 使用形态学梯度提取导航区域的边界
        kernel_boundary = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        boundary = cv2.morphologyEx(occupancy, cv2.MORPH_GRADIENT, kernel_boundary)
        
        # ==================== Step 5: 可选的距离变换着色 ====================
        if enable_distance_shading and np.sum(occupancy > 0) > 100:
            # 计算到边界的距离（用于柔和空间梯度）
            dist_transform = cv2.distanceTransform(occupancy, cv2.DIST_L2, 5)
            
            # 归一化距离图
            dist_max = dist_transform.max()
            if dist_max > 0:
                dist_normalized = dist_transform / dist_max
            else:
                dist_normalized = np.zeros_like(dist_transform)
        else:
            dist_normalized = None
        
        # ==================== Step 6: 根据风格渲染 RGB ====================
        rgb = self._render_structured_map_style(
            occupancy, boundary, dist_normalized, style, map_height, map_width
        )
        
        return rgb, map_bounds, meters_per_pixel

    def _render_structured_map_style(
        self,
        occupancy: np.ndarray,
        boundary: np.ndarray,
        dist_normalized: Optional[np.ndarray],
        style: str,
        map_height: int,
        map_width: int,
    ) -> np.ndarray:
        """
        根据风格渲染结构底图
        
        Args:
            occupancy: 占用栅格 (H, W) uint8, 255=可导航, 0=障碍
            boundary: 边界图 (H, W) uint8
            dist_normalized: 归一化距离图 (H, W) float32，可为 None
            style: 风格名称
            map_height, map_width: 地图尺寸
            
        Returns:
            RGB 图像 (H, W, 3) uint8
        """
        import cv2
        
        # 定义风格配色
        STYLES = {
            "academic": {
                "background": (250, 250, 250),      # 几乎白色背景
                "navigable_base": (225, 225, 225),  # 浅灰色导航区基色
                "navigable_center": (240, 240, 240),# 导航区中心（更浅）
                "boundary": (80, 80, 80),           # 深灰色边界
                "boundary_width": 2,
            },
            "blueprint": {
                "background": (30, 40, 60),         # 深蓝背景
                "navigable_base": (70, 100, 140),   # 中蓝导航区
                "navigable_center": (100, 140, 180),# 浅蓝中心
                "boundary": (200, 220, 255),        # 亮蓝边界
                "boundary_width": 2,
            },
            "grayscale": {
                "background": (200, 200, 200),      # 中灰背景
                "navigable_base": (245, 245, 245),  # 近白导航区
                "navigable_center": (255, 255, 255),# 纯白中心
                "boundary": (50, 50, 50),           # 深灰边界
                "boundary_width": 1,
            },
            "paper": {
                # 极简论文风格，模拟手绘平面图
                "background": (255, 253, 248),      # 米白色纸张
                "navigable_base": (248, 246, 241),  # 淡米色
                "navigable_center": (252, 250, 245),# 更浅
                "boundary": (60, 55, 50),           # 墨色边界
                "boundary_width": 2,
            },
        }
        
        if style not in STYLES:
            style = "academic"
        colors = STYLES[style]
        
        # 初始化背景
        rgb = np.full((map_height, map_width, 3), colors["background"], dtype=np.uint8)
        
        # 绘制导航区域
        nav_mask = occupancy > 0
        
        if dist_normalized is not None and np.any(nav_mask):
            # 使用距离变换创建柔和梯度
            # 边缘区域用 navigable_base，中心区域用 navigable_center
            base = np.array(colors["navigable_base"], dtype=np.float32)
            center = np.array(colors["navigable_center"], dtype=np.float32)
            
            # 应用 gamma 曲线使梯度更自然
            dist_gamma = np.power(dist_normalized, 0.7)
            
            # 混合颜色
            for c in range(3):
                rgb[:, :, c] = np.where(
                    nav_mask,
                    (base[c] + (center[c] - base[c]) * dist_gamma).astype(np.uint8),
                    rgb[:, :, c]
                )
        else:
            # 无距离着色，使用均匀颜色
            rgb[nav_mask] = colors["navigable_base"]
        
        # 绘制边界
        # 先膨胀边界线以增加可见性
        if colors["boundary_width"] > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (colors["boundary_width"], colors["boundary_width"])
            )
            boundary_thick = cv2.dilate(boundary, kernel, iterations=1)
        else:
            boundary_thick = boundary
        
        # 边界稍微模糊以避免锯齿
        boundary_soft = cv2.GaussianBlur(boundary_thick.astype(np.float32), (3, 3), 0)
        boundary_soft = (boundary_soft / boundary_soft.max() * 255).astype(np.uint8) if boundary_soft.max() > 0 else boundary_thick
        
        # 应用边界颜色（使用 alpha 混合以实现抗锯齿）
        boundary_alpha = boundary_soft.astype(np.float32) / 255.0
        boundary_color = np.array(colors["boundary"], dtype=np.float32)
        
        for c in range(3):
            rgb[:, :, c] = (
                rgb[:, :, c].astype(np.float32) * (1 - boundary_alpha) +
                boundary_color[c] * boundary_alpha
            ).astype(np.uint8)
        
        return rgb

    # ==================== 探索感知结构化底图 ====================
    
    def get_exploration_aware_topdown_map(
        self,
        trajectory: Optional[np.ndarray] = None,
        headings: Optional[np.ndarray] = None,
        sensor_fov_deg: float = 90.0,
        sensor_range_m: float = 5.0,
        meters_per_pixel: float = 0.02,  # 更高分辨率
        style: str = "clean",  # 默认使用清晰风格
        visited_radius_m: float = 0.5,
        enable_los_check: bool = False,
        show_frontier_glow: bool = False,  # 默认关闭蓝色光晕
        min_map_size: int = 800,  # 最小地图尺寸
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]], float, Dict[str, np.ndarray]]:
        """
        生成探索感知的结构化俯视底图（Exploration-Aware Structured Top-down Map）
        
        【已修复问题 V3】
        1. ✅ 底图轮廓清晰锐利（无模糊）
        2. ✅ 移除外圈蓝色光晕（可通过 show_frontier_glow 开启）
        3. ✅ 场景占据大部分画布（自动裁剪 + 最小尺寸保证）
        4. ✅ 已探索/未探索区域用明显不同颜色区分
        5. ✅ 返回 icon_avoidance_mask 用于图标防重叠布局
        
        【视觉规范】
        - 已探索区域：白色填充 + 深黑色清晰边界
        - 已观测区域：淡蓝灰色填充 + 中灰色边界
        - 未探索区域：深灰色统一背景，无任何结构信息
        - 边界线条：锐利无模糊，2-3px 宽度
        
        【风格选项】
        - "clean": 默认清晰风格（深灰背景）
        - "contrast": 高对比度版本
        - "print": 适合打印的黑白友好版本
        
        Args:
            trajectory: 机器人历史轨迹 (N, 3) [x, y, z] 或 (N, 2) [x, z]
            headings: 每个轨迹点的朝向角度（弧度）
            sensor_fov_deg: 传感器水平视场角（度）
            sensor_range_m: 传感器最大观测距离（米）
            meters_per_pixel: 分辨率（默认 0.02m = 2cm/pixel，更清晰）
            style: 风格选择 ("clean", "contrast", "print")
            visited_radius_m: 访问点周围视为"已访问"的半径（米）
            enable_los_check: 是否启用视线遮挡检测
            show_frontier_glow: 是否显示探索前沿光晕（默认 False）
            min_map_size: 最小地图尺寸（像素），确保场景足够大
            
        Returns:
            (rgb_map, map_bounds, meters_per_pixel, masks)
            - rgb_map: (H, W, 3) np.uint8 RGB 图像
            - map_bounds: (min_bound, max_bound) 各为 (3,) numpy 数组
            - meters_per_pixel: 实际使用的分辨率
            - masks: Dict 包含详细的探索状态掩码，包括 'icon_avoidance_mask'
        """
        import cv2
        
        # ==================== Step 1: 获取场景边界 ====================
        try:
            if not hasattr(self, "pathfinder") or self.pathfinder is None:
                logging.warning("No pathfinder available")
                return None, None, meters_per_pixel, {}
            
            bounds = self.pathfinder.get_bounds()
            min_bound = np.array(bounds[0])
            max_bound = np.array(bounds[1])
        except Exception as e:
            logging.warning(f"Failed to get pathfinder bounds: {e}")
            return None, None, meters_per_pixel, {}
        
        # ==================== Step 2: 计算优化后的地图尺寸 ====================
        x_range = max_bound[0] - min_bound[0]
        z_range = max_bound[2] - min_bound[2]
        
        # 添加边缘填充
        x_padding = x_range * 0.1
        z_padding = z_range * 0.1
        
        # 调整边界
        min_bound_padded = min_bound.copy()
        max_bound_padded = max_bound.copy()
        min_bound_padded[0] -= x_padding
        min_bound_padded[2] -= z_padding
        max_bound_padded[0] += x_padding
        max_bound_padded[2] += z_padding
        
        x_range_padded = max_bound_padded[0] - min_bound_padded[0]
        z_range_padded = max_bound_padded[2] - min_bound_padded[2]
        
        # 计算初始地图尺寸
        map_width = int(np.ceil(x_range_padded / meters_per_pixel))
        map_height = int(np.ceil(z_range_padded / meters_per_pixel))
        
        # 确保最小尺寸（场景占据足够大的画布）
        if max(map_width, map_height) < 800:
            scale_factor = 800 / max(map_width, map_height)
            map_width = int(map_width * scale_factor)
            map_height = int(map_height * scale_factor)
            meters_per_pixel = max(x_range_padded, z_range_padded) / max(map_width, map_height)
        
        # 限制最大尺寸
        max_dim = 3000
        if max(map_width, map_height) > max_dim:
            scale = max_dim / max(map_width, map_height)
            map_width = int(map_width * scale)
            map_height = int(map_height * scale)
            meters_per_pixel = max(x_range_padded, z_range_padded) / max(map_width, map_height)
        
        map_bounds = (min_bound_padded, max_bound_padded)
        
        # ==================== Step 3: 构建高质量占用栅格 ====================
        occupancy = np.zeros((map_height, map_width), dtype=np.uint8)
        
        try:
            navmesh_vertices = self.pathfinder.build_navmesh_vertices()
            navmesh_indices = self.pathfinder.build_navmesh_vertex_indices()
            
            if navmesh_vertices and navmesh_indices:
                vertices = np.array([np.array(v).flatten() for v in navmesh_vertices])
                
                for i in range(0, len(navmesh_indices), 3):
                    if i + 2 < len(navmesh_indices):
                        idx0, idx1, idx2 = navmesh_indices[i], navmesh_indices[i+1], navmesh_indices[i+2]
                        if idx0 < len(vertices) and idx1 < len(vertices) and idx2 < len(vertices):
                            v0, v1, v2 = vertices[idx0], vertices[idx1], vertices[idx2]
                            pts = np.array([
                                [(v0[0] - min_bound_padded[0]) / meters_per_pixel, 
                                 (v0[2] - min_bound_padded[2]) / meters_per_pixel],
                                [(v1[0] - min_bound_padded[0]) / meters_per_pixel, 
                                 (v1[2] - min_bound_padded[2]) / meters_per_pixel],
                                [(v2[0] - min_bound_padded[0]) / meters_per_pixel, 
                                 (v2[2] - min_bound_padded[2]) / meters_per_pixel],
                            ], dtype=np.int32)
                            pts[:, 0] = np.clip(pts[:, 0], 0, map_width - 1)
                            pts[:, 1] = np.clip(pts[:, 1], 0, map_height - 1)
                            cv2.fillPoly(occupancy, [pts], 255)
        except Exception as e:
            logging.warning(f"Failed to build occupancy from navmesh: {e}")
            return None, None, meters_per_pixel, {}
        
        # ==================== Step 4: 构建探索状态掩码 ====================
        explored_mask = np.zeros((map_height, map_width), dtype=np.uint8)
        observed_mask = np.zeros((map_height, map_width), dtype=np.uint8)
        
        visited_radius_px = max(1, int(visited_radius_m / meters_per_pixel))
        sensor_range_px = max(1, int(sensor_range_m / meters_per_pixel))
        fov_rad = np.radians(sensor_fov_deg)
        
        if trajectory is not None and len(trajectory) > 0:
            traj = np.array(trajectory)
            if traj.ndim == 1:
                traj = traj.reshape(1, -1)
            
            # 提取 x, z 坐标
            if traj.shape[1] == 3:
                traj_xz = traj[:, [0, 2]]
            elif traj.shape[1] == 2:
                traj_xz = traj
            else:
                traj_xz = traj[:, :2]
            
            # 转换为像素坐标
            traj_pixels = np.zeros_like(traj_xz)
            traj_pixels[:, 0] = (traj_xz[:, 0] - min_bound_padded[0]) / meters_per_pixel
            traj_pixels[:, 1] = (traj_xz[:, 1] - min_bound_padded[2]) / meters_per_pixel
            
            # 计算或使用提供的朝向
            if headings is None:
                headings = self._compute_headings_from_trajectory(traj_pixels)
            else:
                headings = np.array(headings)
            
            # 为每个轨迹点绘制探索区域
            for i, (px, py) in enumerate(traj_pixels):
                px_int, py_int = int(px), int(py)
                if not (0 <= px_int < map_width and 0 <= py_int < map_height):
                    continue
                
                # 已访问区域（圆形）
                cv2.circle(explored_mask, (px_int, py_int), visited_radius_px, 255, -1)
                
                # 已观测区域（扇形视野）
                heading = headings[i] if i < len(headings) else 0
                
                if enable_los_check:
                    self._draw_fov_with_los(
                        observed_mask, occupancy, px_int, py_int, heading,
                        fov_rad, sensor_range_px, map_width, map_height
                    )
                else:
                    self._draw_fov_sector(
                        observed_mask, px_int, py_int, heading,
                        fov_rad, sensor_range_px, map_width, map_height
                    )
        
        # ==================== Step 5: 计算区域分类 ====================
        explored_nav = (explored_mask > 0) & (occupancy > 0)
        observed_only = (observed_mask > 0) & ~(explored_mask > 0) & (occupancy > 0)
        visible_area = (explored_mask > 0) | (observed_mask > 0)
        unexplored = ~visible_area
        
        # ==================== Step 6: 提取清晰边界（无模糊）====================
        kernel_boundary = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # 已探索区域边界 - 使用形态学梯度获得清晰边界
        explored_binary = (explored_nav.astype(np.uint8) * 255)
        explored_boundary = cv2.morphologyEx(explored_binary, cv2.MORPH_GRADIENT, kernel_boundary)
        
        # 已观测区域边界
        observed_binary = (observed_only.astype(np.uint8) * 255)
        observed_boundary = cv2.morphologyEx(observed_binary, cv2.MORPH_GRADIENT, kernel_boundary)
        
        # 整体导航区域边界（用于外轮廓）
        nav_binary = (occupancy > 0).astype(np.uint8) * 255
        nav_boundary = cv2.morphologyEx(nav_binary, cv2.MORPH_GRADIENT, kernel_boundary)
        
        # ==================== Step 7: 渲染清晰 RGB 图像 ====================
        rgb = self._render_clean_exploration_map(
            occupancy=occupancy,
            explored_mask=explored_nav,
            observed_mask=observed_only,
            visible_area=visible_area,
            explored_boundary=explored_boundary,
            observed_boundary=observed_boundary,
            nav_boundary=nav_boundary,
            style=style,
            map_height=map_height,
            map_width=map_width,
            show_frontier_glow=show_frontier_glow,
        )

        # ── 叠加物体实例色块（黑底纯色风格）──────────────────────────────
        obj_layer = self._render_object_instance_map(
            map_height=map_height,
            map_width=map_width,
            min_bound=min_bound_padded,
            meters_per_pixel=meters_per_pixel,
        )
        # 将非黑区域（有物体色块）叠加到探索底图上
        obj_mask = obj_layer.any(axis=2)   # (H, W) bool：有物体的像素
        rgb[obj_mask] = obj_layer[obj_mask]
        
        # ==================== Step 8: 生成图标避让掩码 ====================
        # 标记边界线和密集区域，用于图标布局时避免重叠
        icon_avoidance_mask = np.zeros((map_height, map_width), dtype=np.uint8)
        
        # 边界区域需要避让
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        boundary_dilated = cv2.dilate(explored_boundary, kernel_dilate, iterations=1)
        icon_avoidance_mask = np.maximum(icon_avoidance_mask, boundary_dilated)
        
        # 未探索区域也需要避让
        icon_avoidance_mask = np.maximum(icon_avoidance_mask, (unexplored.astype(np.uint8) * 128))
        
        # 返回详细掩码
        masks = {
            'explored': explored_nav.astype(np.uint8) * 255,
            'observed': observed_only.astype(np.uint8) * 255,
            'unexplored': unexplored.astype(np.uint8) * 255,
            'visible_area': visible_area.astype(np.uint8) * 255,
            'occupancy': occupancy,
            'explored_boundary': explored_boundary,
            'observed_boundary': observed_boundary,
            'nav_boundary': nav_boundary,
            'icon_avoidance_mask': icon_avoidance_mask,
        }
        
        return rgb, map_bounds, meters_per_pixel, masks

    def _compute_headings_from_trajectory(self, traj_pixels: np.ndarray) -> np.ndarray:
        """
        从像素坐标轨迹计算每个点的瞬时航向角（弧度）。

        Habitat 坐标系：X 向右，Z 向前。在 2D 投影后 traj_pixels[:,0]=X, traj_pixels[:,1]=Z。
        arctan2(dz, dx) 给出从 +X 轴出发的逆时针角度；这里使用 atan2(Δcol, Δrow)
        的约定，与 _draw_fov_sector 保持一致。

        Args:
            traj_pixels: (N, 2) 像素坐标序列，每行 [px_x, px_z]

        Returns:
            (N,) 朝向角数组（弧度）
        """
        n = len(traj_pixels)
        headings = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i < n - 1:
                dx = traj_pixels[i + 1, 0] - traj_pixels[i, 0]
                dz = traj_pixels[i + 1, 1] - traj_pixels[i, 1]
            else:
                # 末尾点沿用前一点方向
                dx = traj_pixels[i, 0] - traj_pixels[i - 1, 0] if i > 0 else 0.0
                dz = traj_pixels[i, 1] - traj_pixels[i - 1, 1] if i > 0 else 0.0
            if abs(dx) < 1e-6 and abs(dz) < 1e-6:
                headings[i] = headings[i - 1] if i > 0 else 0.0
            else:
                headings[i] = np.arctan2(dz, dx)
        return headings

    def _draw_fov_sector(
        self,
        mask: np.ndarray,
        px: int, py: int,
        heading: float,
        fov_rad: float,
        range_px: int,
        map_width: int, map_height: int,
    ) -> None:
        """在掩码上绘制扇形视野区域（无视线遮挡）。"""
        import cv2
        half_fov = fov_rad / 2.0
        # 生成扇形顶点
        n_pts = 32
        pts = [[px, py]]
        for k in range(n_pts + 1):
            angle = heading - half_fov + fov_rad * k / n_pts
            ex = int(np.clip(px + range_px * np.cos(angle), 0, map_width - 1))
            ey = int(np.clip(py + range_px * np.sin(angle), 0, map_height - 1))
            pts.append([ex, ey])
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts_arr], 255)

    def _draw_fov_with_los(
        self,
        mask: np.ndarray,
        occupancy: np.ndarray,
        px: int, py: int,
        heading: float,
        fov_rad: float,
        range_px: int,
        map_width: int, map_height: int,
    ) -> None:
        """在掩码上绘制扇形视野区域（带简单视线遮挡检测）。"""
        import cv2
        half_fov = fov_rad / 2.0
        n_rays = 64
        pts = [[px, py]]
        for k in range(n_rays + 1):
            angle = heading - half_fov + fov_rad * k / n_rays
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            end_px, end_py = px, py
            for r in range(1, range_px + 1):
                cx = int(px + r * cos_a)
                cy = int(py + r * sin_a)
                if not (0 <= cx < map_width and 0 <= cy < map_height):
                    break
                if occupancy[cy, cx] == 0:  # 障碍物
                    break
                end_px, end_py = cx, cy
            pts.append([end_px, end_py])
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts_arr], 255)

    def _render_object_instance_map(
        self,
        map_height: int,
        map_width: int,
        min_bound: np.ndarray,
        meters_per_pixel: float,
    ) -> np.ndarray:
        """
        黑背景 + 每个物体实例分配固定纯色色块的俯视图。

        - 背景：纯黑 (0, 0, 0)
        - 每个 object_id 根据哈希分配一种高饱和度颜色
        - 根据 bbox 或 pcd 在 XZ 平面投影，填充对应色块
        """
        import cv2

        rgb = np.zeros((map_height, map_width, 3), dtype=np.uint8)

        def _obj_color(obj_id: int) -> tuple:
            """为 object_id 分配稳定的高饱和度颜色（HSV → RGB）。"""
            import colorsys
            hue = (obj_id * 137.508) % 360.0  # 黄金角散列，确保颜色间距大
            r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 0.95)
            return (int(r * 255), int(g * 255), int(b * 255))

        for obj_id, obj in self.objects.items():
            color = _obj_color(int(obj_id))
            bbox = obj.get("bbox", None)
            pcd  = obj.get("pcd", None)

            drawn = False

            # ── 优先用 bbox 的 8 个角点投影到 XZ 平面 ──────────────────────
            if bbox is not None and hasattr(bbox, "get_box_points"):
                try:
                    pts_3d = np.asarray(bbox.get_box_points())  # (8, 3)
                    xs = ((pts_3d[:, 0] - min_bound[0]) / meters_per_pixel).astype(int)
                    zs = ((pts_3d[:, 2] - min_bound[2]) / meters_per_pixel).astype(int)
                    xs = np.clip(xs, 0, map_width - 1)
                    zs = np.clip(zs, 0, map_height - 1)
                    hull_pts = np.stack([xs, zs], axis=1).reshape(-1, 1, 2).astype(np.int32)
                    hull = cv2.convexHull(hull_pts)
                    cv2.fillPoly(rgb, [hull], color)
                    drawn = True
                except Exception:
                    pass

            # ── 回退：用点云投影 ────────────────────────────────────────────
            if not drawn and pcd is not None and hasattr(pcd, "points"):
                try:
                    pts_3d = np.asarray(pcd.points)
                    if len(pts_3d) > 0:
                        xs = ((pts_3d[:, 0] - min_bound[0]) / meters_per_pixel).astype(int)
                        zs = ((pts_3d[:, 2] - min_bound[2]) / meters_per_pixel).astype(int)
                        xs = np.clip(xs, 0, map_width - 1)
                        zs = np.clip(zs, 0, map_height - 1)
                        hull_pts = np.stack([xs, zs], axis=1).reshape(-1, 1, 2).astype(np.int32)
                        hull = cv2.convexHull(hull_pts)
                        cv2.fillPoly(rgb, [hull], color)
                except Exception:
                    pass

        return rgb

    def _render_clean_exploration_map(
        self,
        occupancy: np.ndarray,
        explored_mask: np.ndarray,
        observed_mask: np.ndarray,
        visible_area: np.ndarray,
        explored_boundary: np.ndarray,
        observed_boundary: np.ndarray,
        nav_boundary: np.ndarray,
        style: str,
        map_height: int,
        map_width: int,
        show_frontier_glow: bool = False,
    ) -> np.ndarray:
        """
        渲染清晰的探索感知地图（无模糊、高对比）
        
        【颜色方案】
        - 已探索：白色填充 + 深黑色边界
        - 已观测：淡蓝灰色填充 + 中灰色边界  
        - 未探索：深灰色统一背景
        """
        import cv2
        
        # 风格配色方案
        STYLES = {
            "clean": {
                # 未探索背景 - 深灰色
                "unexplored_bg": (85, 85, 95),
                # 已观测区域 - 淡蓝灰色
                "observed_fill": (200, 210, 220),
                "observed_boundary": (130, 140, 150),
                # 已探索区域 - 白色
                "explored_fill": (255, 255, 255),
                "explored_boundary": (30, 30, 35),
                # 边界线宽度
                "boundary_width": 2,
            },
            "contrast": {
                # 更高对比度版本
                "unexplored_bg": (60, 60, 70),
                "observed_fill": (180, 195, 210),
                "observed_boundary": (100, 110, 120),
                "explored_fill": (255, 255, 255),
                "explored_boundary": (20, 20, 25),
                "boundary_width": 3,
            },
            "print": {
                # 适合打印的黑白友好版本
                "unexplored_bg": (200, 200, 200),
                "observed_fill": (235, 235, 235),
                "observed_boundary": (150, 150, 150),
                "explored_fill": (255, 255, 255),
                "explored_boundary": (0, 0, 0),
                "boundary_width": 2,
            },
        }
        
        if style not in STYLES:
            style = "clean"
        colors = STYLES[style]
        
        # ========== Layer 1: 未探索背景 ==========
        rgb = np.full((map_height, map_width, 3), colors["unexplored_bg"], dtype=np.uint8)
        
        # ========== Layer 2: 已观测区域（淡色填充）==========
        if observed_mask.any():
            for c in range(3):
                rgb[:, :, c] = np.where(observed_mask, colors["observed_fill"][c], rgb[:, :, c])
        
        # ========== Layer 3: 已探索区域（白色填充）==========
        if explored_mask.any():
            for c in range(3):
                rgb[:, :, c] = np.where(explored_mask, colors["explored_fill"][c], rgb[:, :, c])
        
        # ========== Layer 4: 清晰边界线（无模糊）==========
        boundary_width = colors["boundary_width"]
        
        # 已观测区域边界 - 中灰色
        if observed_boundary.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (boundary_width, boundary_width))
            observed_boundary_thick = cv2.dilate(observed_boundary, kernel, iterations=1)
            boundary_mask = observed_boundary_thick > 0
            for c in range(3):
                rgb[:, :, c] = np.where(boundary_mask, colors["observed_boundary"][c], rgb[:, :, c])
        
        # 已探索区域边界 - 深黑色（覆盖在已观测边界上）
        if explored_boundary.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (boundary_width, boundary_width))
            explored_boundary_thick = cv2.dilate(explored_boundary, kernel, iterations=1)
            boundary_mask = explored_boundary_thick > 0
            for c in range(3):
                rgb[:, :, c] = np.where(boundary_mask, colors["explored_boundary"][c], rgb[:, :, c])
        
        # ========== Layer 5: 可选的探索前沿光晕 ==========
        if show_frontier_glow:
            # 计算探索前沿
            kernel_glow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            explored_dilated = cv2.dilate(explored_mask.astype(np.uint8), kernel_glow, iterations=3)
            frontier = (explored_dilated > 0) & ~(explored_mask > 0) & ~visible_area
            
            if frontier.any():
                # 淡蓝色光晕
                frontier_color = np.array([100, 150, 200], dtype=np.float32)
                alpha = 0.3
                for c in range(3):
                    rgb[:, :, c] = np.where(
                        frontier,
                        (rgb[:, :, c].astype(np.float32) * (1 - alpha) + frontier_color[c] * alpha).astype(np.uint8),
                        rgb[:, :, c]
                    )
        
        return rgb
    
    def filter_nodes_by_exploration(
        self,
        node_positions: Dict[int, np.ndarray],
        explored_mask: np.ndarray,
        map_bounds: Tuple[np.ndarray, np.ndarray],
        meters_per_pixel: float,
        allow_observed: bool = False,
        observed_mask: Optional[np.ndarray] = None,
    ) -> Dict[int, np.ndarray]:
        """
        根据探索状态过滤节点，确保 Memory Graph 节点仅在已探索区域
        
        【设计原则】
        在线探索过程中，显式记忆图的节点应仅位于机器人"真正访问过"的区域，
        不允许在 observed-but-unvisited 或 unexplored 区域放置节点。
        
        Args:
            node_positions: {node_id: position (3,) [x, y, z]}
            explored_mask: 已探索区域掩码 (H, W) uint8
            map_bounds: (min_bound, max_bound)
            meters_per_pixel: 分辨率
            allow_observed: 如果 True，也允许 observed 区域的节点
            observed_mask: 已观测区域掩码（仅当 allow_observed=True 时使用）
            
        Returns:
            过滤后的节点位置字典
        """
        min_bound, max_bound = map_bounds
        map_height, map_width = explored_mask.shape
        
        filtered_nodes = {}
        
        for node_id, pos in node_positions.items():
            # 转换为像素坐标
            if len(pos) == 3:
                px = int((pos[0] - min_bound[0]) / meters_per_pixel)
                py = int((pos[2] - min_bound[2]) / meters_per_pixel)
            else:
                px = int((pos[0] - min_bound[0]) / meters_per_pixel)
                py = int((pos[1] - min_bound[2]) / meters_per_pixel)
            
            # 检查是否在地图范围内
            if not (0 <= px < map_width and 0 <= py < map_height):
                continue
            
            # 检查是否在已探索区域
            if explored_mask[py, px] > 0:
                filtered_nodes[node_id] = pos
            elif allow_observed and observed_mask is not None and observed_mask[py, px] > 0:
                filtered_nodes[node_id] = pos
        
        return filtered_nodes
