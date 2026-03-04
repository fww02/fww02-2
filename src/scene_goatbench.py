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
        if int(scene_id.split("-")[0]) < 800:
            split_path = os.path.join(cfg.scene_data_path, "train")
        else:
            split_path = os.path.join(cfg.scene_data_path, "val")
        scene_mesh_path = os.path.join(
            split_path, scene_id, scene_id.split("-")[1] + ".basis.glb"
        )
        navmesh_path = os.path.join(
            split_path, scene_id, scene_id.split("-")[1] + ".basis.navmesh"
        )
        semantic_texture_path = os.path.join(
            split_path, scene_id, scene_id.split("-")[1] + ".semantic.glb"
        )
        scene_semantic_annotation_path = os.path.join(
            split_path, scene_id, scene_id.split("-")[1] + ".semantic.txt"
        )
        assert os.path.exists(
            scene_mesh_path
        ), f"scene_mesh_path: {scene_mesh_path} does not exist"
        assert os.path.exists(
            navmesh_path
        ), f"navmesh_path: {navmesh_path} does not exist"
        assert os.path.exists(
            semantic_texture_path
        ), f"semantic_texture_path: {semantic_texture_path} does not exist"
        assert os.path.exists(
            scene_semantic_annotation_path
        ), f"scene_semantic_annotation_path: {scene_semantic_annotation_path} does not exist"

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
        sim_cfg = make_semantic_cfg(sim_settings)
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

        logging.info(f"Load scene {scene_id} successfully")

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

    def get_observation(self, pts, angle=None, rotation=None):
        assert (angle is None) ^ (
            rotation is None
        ), "Only one of angle and rotation should be specified"

        agent_state = habitat_sim.AgentState()
        agent_state.position = pts
        if angle is not None:
            agent_state.rotation = get_quaternion(angle, 0)
        else:
            agent_state.rotation = rotation
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
        semantic_obs=Optional[np.ndarray],
        gt_target_obj_ids=Optional[List[int]],
    ) -> Tuple[np.ndarray, List[int], Dict[int, int]]:
        # return annotated image; the detected object ids in current frame; the object id of the target object (if detected)
        assert not (
            (semantic_obs is None) ^ (gt_target_obj_ids is None)
        ), "semantic_obs and gt_target_obj_ids should be both None or both not None"

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
            return image_rgb, [], {}

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
            return image_rgb, [], {}

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
            return image_rgb, [], {}

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
            return image_rgb, [], {}

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
            return image_rgb, [], {}

        # compare the detections with the target object mask to see whether the target object is detected
        target_obj_id_mapping = {}
        if semantic_obs is not None:
            for target_gt_id in gt_target_obj_ids:
                target_obj_mask = semantic_obs == target_gt_id
                if (
                    np.sum(target_obj_mask)
                    / (target_obj_mask.shape[0] * target_obj_mask.shape[1])
                    > 0.0001
                ):
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
                        target_obj_id_mapping[target_gt_id] = max_iou_obj_id
                        logging.info(
                            f"Target object {target_gt_id} detected with IoU {max_iou} in {img_path}!!!"
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

        det_visual_prompt = sv.Detections(
            xyxy=gobs["xyxy"],
            class_id=gobs["class_id"],
        )

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(self.objects) == 0:
            logging.debug(
                f"No objects in the map yet, adding all detections of length {len(detection_list)}"
            )
            self.objects.update(detection_list)

            det_visual_prompt.data["obj_id"] = list(detection_list.keys())
            frame.visual_prompt = det_visual_prompt
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
            visualize_captions, target_obj_id_mapping, added_obj_ids, all_obj_ids = (
                self.merge_obj_matches(
                    detection_list=detection_list,
                    match_indices=match_indices,
                    obj_classes=obj_classes,
                    snapshot=frame,
                    target_obj_id_mapping=target_obj_id_mapping,
                )
            )

            det_visual_prompt.data["obj_id"] = all_obj_ids
            frame.visual_prompt = det_visual_prompt

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

        return annotated_image, added_obj_ids, target_obj_id_mapping

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
        target_obj_id_mapping: Dict[int, int],
    ) -> Tuple[List[str], Dict[int, int], List[int], List[int]]:
        visualize_captions = []
        all_obj_ids = []
        added_obj_ids = []
        for idx, (detected_obj_id, existing_obj_match_id) in enumerate(match_indices):
            if existing_obj_match_id is None:
                self.objects[detected_obj_id] = detection_list[detected_obj_id]
                visualize_captions.append(
                    f"{detected_obj_id} {self.objects[detected_obj_id]['class_name']} {self.objects[detected_obj_id]['conf']:.3f} N"
                )
                all_obj_ids.append(detected_obj_id)
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
                all_obj_ids.append(existing_obj_match_id)

                # update the mapping of target object id
                for gt_id, mapped_id in target_obj_id_mapping.items():
                    if mapped_id == detected_obj_id:
                        target_obj_id_mapping[gt_id] = existing_obj_match_id

        return visualize_captions, target_obj_id_mapping, added_obj_ids, all_obj_ids

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

    def periodic_cleanup_objects(self, frame_idx, pts, goal_obj_ids_mapping=None):
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
                goal_obj_ids_mapping=goal_obj_ids_mapping,
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

        # update the goal object ids mapping to remove the objects that have been removed
        if goal_obj_ids_mapping is not None:
            for goal_obj_id, mapped_obj_ids in goal_obj_ids_mapping.items():
                goal_obj_ids_mapping[goal_obj_id] = [
                    obj_id for obj_id in mapped_obj_ids if obj_id in self.objects.keys()
                ]

    def sanity_check(self, cfg):
        # sanity check
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

    def get_topdown_map(self, colorize: bool = True, meters_per_pixel: float = 0.05):
        """
        Return a top-down map for the current scene.
        
        Uses navmesh vertices to generate a proper topdown occupancy map.

        Args:
            colorize: If True, return RGB colorized map; if False, return binary occupancy
            meters_per_pixel: Resolution of the topdown map (default 0.05m = 5cm per pixel)

        Returns:
            (rgb_map, map_bounds)
            - rgb_map: HxWx3 uint8 RGB array (if available) or None
            - map_bounds: (min_bound, max_bound) as two (3,) numpy arrays, or None
        """
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

        # Create occupancy grid (0 = unknown/obstacle, 1 = navigable)
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
                    import cv2
                    
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

        return rgb

    # ==================== 新增：纯几何学术底图 ====================
    
    def get_structured_topdown_map(
        self, 
        meters_per_pixel: float = 0.025,
        style: str = "academic",
        enable_distance_shading: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]], float]:
        """
        生成纯几何结构的学术级俯视底图（Academic Structured Top-down Map）
        
        完全放弃 RGB 纹理拼接，仅从 navmesh/pathfinder 数据生成干净的结构底图。
        
        Args:
            meters_per_pixel: 分辨率（默认 0.025m = 2.5cm/pixel）
            style: 风格选择 ("academic", "blueprint", "grayscale", "paper")
            enable_distance_shading: 是否启用距离变换着色
            
        Returns:
            (rgb_map, map_bounds, meters_per_pixel)
        """
        import cv2
        
        # 获取场景边界
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
        
        # 计算地图尺寸
        x_range = max_bound[0] - min_bound[0]
        z_range = max_bound[2] - min_bound[2]
        
        map_width = int(np.ceil(x_range / meters_per_pixel))
        map_height = int(np.ceil(z_range / meters_per_pixel))
        
        # 限制最大尺寸
        max_dim = 3000
        if map_width > max_dim or map_height > max_dim:
            scale = max_dim / max(map_width, map_height)
            map_width = int(map_width * scale)
            map_height = int(map_height * scale)
            meters_per_pixel = max(x_range, z_range) / max_dim

        # 构建占用栅格
        occupancy = np.zeros((map_height, map_width), dtype=np.uint8)
        
        navmesh_available = False
        try:
            if hasattr(self, "pathfinder") and self.pathfinder is not None:
                navmesh_vertices = self.pathfinder.build_navmesh_vertices()
                navmesh_indices = self.pathfinder.build_navmesh_vertex_indices()
                
                if navmesh_vertices and navmesh_indices:
                    navmesh_available = True
                    vertices = np.array([np.array(v).flatten() for v in navmesh_vertices])
                    
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
        except Exception as e:
            logging.debug(f"Failed to build navmesh occupancy: {e}")
        
        # 采样回退方法
        if not navmesh_available or np.sum(occupancy > 0) < 100:
            try:
                if hasattr(self, "pathfinder") and self.pathfinder is not None:
                    n_samples = min(50000, map_width * map_height // 4)
                    for _ in range(n_samples):
                        pt = self.pathfinder.get_random_navigable_point()
                        if pt is not None:
                            px = int((pt[0] - min_bound[0]) / meters_per_pixel)
                            pz = int((pt[2] - min_bound[2]) / meters_per_pixel)
                            if 0 <= px < map_width and 0 <= pz < map_height:
                                radius = max(1, int(0.1 / meters_per_pixel))
                                cv2.circle(occupancy, (px, pz), radius, 255, -1)
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)
            except Exception as e:
                logging.warning(f"Sampling-based occupancy also failed: {e}")

        # 提取边界轮廓
        kernel_boundary = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        boundary = cv2.morphologyEx(occupancy, cv2.MORPH_GRADIENT, kernel_boundary)
        
        # 距离变换着色
        if enable_distance_shading and np.sum(occupancy > 0) > 100:
            dist_transform = cv2.distanceTransform(occupancy, cv2.DIST_L2, 5)
            dist_max = dist_transform.max()
            dist_normalized = dist_transform / dist_max if dist_max > 0 else np.zeros_like(dist_transform)
        else:
            dist_normalized = None
        
        # 渲染 RGB
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
        """根据风格渲染结构底图"""
        import cv2
        
        STYLES = {
            "academic": {
                "background": (250, 250, 250),
                "navigable_base": (225, 225, 225),
                "navigable_center": (240, 240, 240),
                "boundary": (80, 80, 80),
                "boundary_width": 2,
            },
            "blueprint": {
                "background": (30, 40, 60),
                "navigable_base": (70, 100, 140),
                "navigable_center": (100, 140, 180),
                "boundary": (200, 220, 255),
                "boundary_width": 2,
            },
            "grayscale": {
                "background": (200, 200, 200),
                "navigable_base": (245, 245, 245),
                "navigable_center": (255, 255, 255),
                "boundary": (50, 50, 50),
                "boundary_width": 1,
            },
            "paper": {
                "background": (255, 253, 248),
                "navigable_base": (248, 246, 241),
                "navigable_center": (252, 250, 245),
                "boundary": (60, 55, 50),
                "boundary_width": 2,
            },
        }
        
        if style not in STYLES:
            style = "academic"
        colors = STYLES[style]
        
        rgb = np.full((map_height, map_width, 3), colors["background"], dtype=np.uint8)
        nav_mask = occupancy > 0
        
        if dist_normalized is not None and np.any(nav_mask):
            base = np.array(colors["navigable_base"], dtype=np.float32)
            center = np.array(colors["navigable_center"], dtype=np.float32)
            dist_gamma = np.power(dist_normalized, 0.7)
            
            for c in range(3):
                rgb[:, :, c] = np.where(
                    nav_mask,
                    (base[c] + (center[c] - base[c]) * dist_gamma).astype(np.uint8),
                    rgb[:, :, c]
                )
        else:
            rgb[nav_mask] = colors["navigable_base"]
        
        # 绘制边界
        if colors["boundary_width"] > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (colors["boundary_width"], colors["boundary_width"])
            )
            boundary_thick = cv2.dilate(boundary, kernel, iterations=1)
        else:
            boundary_thick = boundary
        
        boundary_soft = cv2.GaussianBlur(boundary_thick.astype(np.float32), (3, 3), 0)
        boundary_soft = (boundary_soft / boundary_soft.max() * 255).astype(np.uint8) if boundary_soft.max() > 0 else boundary_thick
        
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
        sensor_fov_deg: float = 90.0,
        sensor_range_m: float = 5.0,
        meters_per_pixel: float = 0.025,
        style: str = "exploration",
        visited_radius_m: float = 0.5,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]], float, Dict[str, np.ndarray]]:
        """
        生成带有探索感知的结构化俯视底图
        
        详见 scene_aeqa.py 中的完整文档
        """
        import cv2
        
        # 获取基础占用栅格
        base_rgb, map_bounds, meters_per_pixel = self.get_structured_topdown_map(
            meters_per_pixel=meters_per_pixel,
            style="academic",
            enable_distance_shading=False,
        )
        
        if base_rgb is None or map_bounds is None:
            return None, None, meters_per_pixel, {}
        
        min_bound, max_bound = map_bounds
        map_height, map_width = base_rgb.shape[:2]
        
        # 重新构建 occupancy
        occupancy = np.zeros((map_height, map_width), dtype=np.uint8)
        try:
            if hasattr(self, "pathfinder") and self.pathfinder is not None:
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
        except Exception:
            gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
            occupancy = (gray > 200).astype(np.uint8) * 255
        
        # 构建探索掩码
        explored_mask = np.zeros((map_height, map_width), dtype=np.uint8)
        observed_mask = np.zeros((map_height, map_width), dtype=np.uint8)
        
        visited_radius_px = int(visited_radius_m / meters_per_pixel)
        sensor_range_px = int(sensor_range_m / meters_per_pixel)
        fov_rad = np.radians(sensor_fov_deg)
        
        if trajectory is not None and len(trajectory) > 0:
            traj = np.array(trajectory)
            if traj.ndim == 1:
                traj = traj.reshape(1, -1)
            
            if traj.shape[1] == 3:
                traj_xz = traj[:, [0, 2]]
            elif traj.shape[1] == 2:
                traj_xz = traj
            else:
                traj_xz = traj[:, :2]
            
            traj_pixels = np.zeros_like(traj_xz)
            traj_pixels[:, 0] = (traj_xz[:, 0] - min_bound[0]) / meters_per_pixel
            traj_pixels[:, 1] = (traj_xz[:, 1] - min_bound[2]) / meters_per_pixel
            
            headings = []
            for i in range(len(traj_pixels)):
                if i == 0:
                    if len(traj_pixels) > 1:
                        dx = traj_pixels[1, 0] - traj_pixels[0, 0]
                        dz = traj_pixels[1, 1] - traj_pixels[0, 1]
                    else:
                        dx, dz = 1, 0
                else:
                    dx = traj_pixels[i, 0] - traj_pixels[i-1, 0]
                    dz = traj_pixels[i, 1] - traj_pixels[i-1, 1]
                
                if abs(dx) < 1e-6 and abs(dz) < 1e-6:
                    headings.append(headings[-1] if headings else 0)
                else:
                    headings.append(np.arctan2(dz, dx))
            
            for i, (px, py) in enumerate(traj_pixels):
                px, py = int(px), int(py)
                if not (0 <= px < map_width and 0 <= py < map_height):
                    continue
                
                cv2.circle(explored_mask, (px, py), visited_radius_px, 255, -1)
                self._draw_fov_sector(
                    observed_mask, px, py, headings[i], 
                    fov_rad, sensor_range_px, map_width, map_height
                )
        else:
            observed_mask = occupancy.copy()
        
        # 计算三种状态区域
        explored_nav = (explored_mask > 0) & (occupancy > 0)
        observed_nav = (observed_mask > 0) & (occupancy > 0) & ~explored_nav
        unexplored = ~(explored_mask > 0) & ~(observed_mask > 0)
        
        # 提取边界
        kernel_boundary = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        explored_boundary = cv2.morphologyEx(
            (explored_nav.astype(np.uint8) * 255), cv2.MORPH_GRADIENT, kernel_boundary
        )
        observed_boundary = cv2.morphologyEx(
            (observed_nav.astype(np.uint8) * 255), cv2.MORPH_GRADIENT, kernel_boundary
        )
        frontier_mask = cv2.dilate(explored_mask, kernel_boundary, iterations=2) - explored_mask
        frontier_mask = frontier_mask.clip(0, 255).astype(np.uint8)
        
        # 渲染
        rgb = self._render_exploration_aware_style(
            occupancy, explored_nav, observed_nav,
            explored_boundary, observed_boundary, frontier_mask,
            style, map_height, map_width
        )
        
        masks = {
            'explored': explored_nav.astype(np.uint8) * 255,
            'observed': observed_nav.astype(np.uint8) * 255,
            'unexplored': unexplored.astype(np.uint8) * 255,
            'frontier': frontier_mask,
            'occupancy': occupancy,
        }
        
        return rgb, map_bounds, meters_per_pixel, masks

    def _draw_fov_sector(self, mask, cx, cy, heading, fov_rad, range_px, map_width, map_height):
        """绘制扇形视野区域"""
        import cv2
        
        start_angle = heading - fov_rad / 2
        end_angle = heading + fov_rad / 2
        n_points = 32
        angles = np.linspace(start_angle, end_angle, n_points)
        
        points = [(cx, cy)]
        for angle in angles:
            x = cx + int(range_px * np.cos(angle))
            y = cy + int(range_px * np.sin(angle))
            points.append((x, y))
        points.append((cx, cy))
        
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    def _render_exploration_aware_style(
        self, occupancy, explored_mask, observed_mask,
        explored_boundary, observed_boundary, frontier_mask,
        style, map_height, map_width
    ):
        """渲染带有探索感知的结构底图"""
        import cv2
        
        STYLES = {
            "exploration": {
                "background": (245, 245, 245),
                "explored_fill": (255, 255, 255),
                "explored_boundary": (40, 40, 40),
                "explored_boundary_width": 2,
                "observed_fill": (235, 235, 235),
                "observed_boundary": (160, 160, 160),
                "observed_alpha": 0.5,
                "frontier_color": (100, 149, 237),
                "frontier_alpha": 0.3,
            },
            "academic": {
                "background": (250, 250, 250),
                "explored_fill": (255, 255, 255),
                "explored_boundary": (60, 60, 60),
                "explored_boundary_width": 2,
                "observed_fill": (240, 240, 240),
                "observed_boundary": (150, 150, 150),
                "observed_alpha": 0.4,
                "frontier_color": (70, 130, 180),
                "frontier_alpha": 0.25,
            },
        }
        
        if style not in STYLES:
            style = "exploration"
        colors = STYLES[style]
        
        rgb = np.full((map_height, map_width, 3), colors["background"], dtype=np.uint8)
        
        # Layer 2: 已观测区域
        observed_color = np.array(colors["observed_fill"], dtype=np.float32)
        alpha = colors["observed_alpha"]
        for c in range(3):
            rgb[:, :, c] = np.where(
                observed_mask,
                (rgb[:, :, c].astype(np.float32) * (1 - alpha) + observed_color[c] * alpha).astype(np.uint8),
                rgb[:, :, c]
            )
        
        # Layer 3: 已探索区域
        explored_color = np.array(colors["explored_fill"], dtype=np.float32)
        for c in range(3):
            rgb[:, :, c] = np.where(explored_mask, explored_color[c].astype(np.uint8), rgb[:, :, c])
        
        # 边界
        boundary_width = colors["explored_boundary_width"]
        if boundary_width > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boundary_width, boundary_width))
            explored_boundary_thick = cv2.dilate(explored_boundary, kernel, iterations=1)
        else:
            explored_boundary_thick = explored_boundary
        
        explored_boundary_soft = cv2.GaussianBlur(explored_boundary_thick.astype(np.float32), (3, 3), 0)
        if explored_boundary_soft.max() > 0:
            explored_boundary_soft = explored_boundary_soft / explored_boundary_soft.max()
        
        explored_boundary_color = np.array(colors["explored_boundary"], dtype=np.float32)
        for c in range(3):
            rgb[:, :, c] = (
                rgb[:, :, c].astype(np.float32) * (1 - explored_boundary_soft) +
                explored_boundary_color[c] * explored_boundary_soft
            ).astype(np.uint8)
        
        return rgb
