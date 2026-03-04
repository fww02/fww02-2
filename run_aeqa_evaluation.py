import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import time
import json
import logging
import matplotlib.pyplot as plt

import open_clip
from ultralytics import SAM, YOLOWorld

from src.habitat import pose_habitat_to_tsdf
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot
from src.scene_aeqa import Scene
from src.utils import resize_image, get_pts_angle_aeqa
from src.query_vlm_aeqa import query_vlm_for_response
from src.logger_aeqa import Logger
from src.const import *
from src.explicit_memory_graph import ExplicitMemoryGraphBuilder


def main(cfg, start_ratio=0.0, end_ratio=1.0):
    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    questions_list = json.load(open(cfg.questions_list_path, "r"))
    total_questions = len(questions_list)
    # sort the data according to the question id
    questions_list = sorted(questions_list, key=lambda x: x["question_id"])
    logging.info(f"Total number of questions: {total_questions}")
    # only process a subset of the questions
    questions_list = questions_list[
        int(start_ratio * total_questions) : int(end_ratio * total_questions)
    ]
    logging.info(f"number of questions after splitting: {len(questions_list)}")
    logging.info(f"question path: {cfg.questions_list_path}")

    # load detection and segmentation models
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "laion2b_s34b_b79k"  # "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logging.info(f"Load CLIP model successful!")

    # Initialize the logger
    logger = Logger(
        cfg.output_dir,
        start_ratio,
        end_ratio,
        len(questions_list),
        voxel_size=cfg.tsdf_grid_size,
    )

    # Run all questions
    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data["question_id"]
        scene_id = question_data["episode_history"]
        if question_id in logger.success_list or question_id in logger.fail_list:
            logging.info(f"Question {question_id} already processed")
            continue
        if any([invalid_scene_id in scene_id for invalid_scene_id in INVALID_SCENE_ID]):
            logging.info(f"Skip invalid scene {scene_id}")
            continue
        logging.info(f"\n========\nIndex: {question_idx} Scene: {scene_id}")

        question = question_data["question"]
        answer = question_data["answer"]
        pts, angle = get_pts_angle_aeqa(
            question_data["position"], question_data["rotation"]
        )

        # load scene
        try:
            del scene
        except:
            pass
        scene = Scene(
            scene_id,
            cfg,
            cfg_cg,
            detection_model,
            sam_predictor,
            clip_model,
            clip_preprocess,
            clip_tokenizer,
        )

        # initialize the TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=get_scene_bnds(scene.pathfinder, floor_height=pts[1])[0],
            voxel_size=cfg.tsdf_grid_size,
            floor_height=pts[1],
            floor_height_offset=0,
            pts_init=pts,
            init_clearance=cfg.init_clearance * 2,
            save_visualization=cfg.save_visualization,
        )

        episode_dir, eps_chosen_snapshot_dir, eps_frontier_dir, eps_snapshot_dir = (
            logger.init_episode(
                question_id=question_id,
                init_pts_voxel=tsdf_planner.habitat2voxel(pts)[:2],
            )
        )

        # --- online lightweight explicit memory graph (GPU-friendly) ---
        emg = ExplicitMemoryGraphBuilder(
            save_root=episode_dir,
            voxel_size=cfg.tsdf_grid_size,
            floor_id="0",
        )
        emg.start_episode(episode_id=question_id, scene_id=scene_id)

        logging.info(f"\n\nQuestion id {question_id} initialization successful!")

        # run steps
        task_success = False
        cnt_step = -1

        gpt_answer = None
        n_filtered_snapshots = 0
        while cnt_step < cfg.num_step - 1:
            cnt_step += 1
            logging.info(f"\n== step: {cnt_step}")

            # (1) Observe the surroundings, update the scene graph and occupancy map
            # Determine the viewing angles for the current step
            if cnt_step == 0:
                angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
                total_views = 1 + cfg.extra_view_phase_2
            else:
                angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
                total_views = 1 + cfg.extra_view_phase_1
            all_angles = [
                angle + angle_increment * (i - total_views // 2)
                for i in range(total_views)
            ]
            # Let the main viewing angle be the last one to avoid potential overwriting problems
            main_angle = all_angles.pop(total_views // 2)
            all_angles.append(main_angle)

            rgb_egocentric_views = []
            all_added_obj_ids = (
                []
            )  # Record all the objects that are newly added in this step
            for view_idx, ang in enumerate(all_angles):
                # For each view
                obs, cam_pose = scene.get_observation(pts, ang)
                rgb = obs["color_sensor"]
                depth = obs["depth_sensor"]

                obs_file_name = f"{cnt_step}-view_{view_idx}.png"
                with torch.no_grad():
                    # Concept graph pipeline update
                    annotated_rgb, added_obj_ids, _ = scene.update_scene_graph(
                        image_rgb=rgb[..., :3],
                        depth=depth,
                        intrinsics=cam_intr,
                        cam_pos=cam_pose,
                        pts=pts,
                        pts_voxel=tsdf_planner.habitat2voxel(pts),
                        img_path=obs_file_name,
                        frame_idx=cnt_step * total_views + view_idx,
                        target_obj_mask=None,
                    )
                    resized_rgb = resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                    scene.all_observations[obs_file_name] = resized_rgb
                    rgb_egocentric_views.append(resized_rgb)
                    if cfg.save_visualization:
                        plt.imsave(
                            os.path.join(eps_snapshot_dir, obs_file_name), annotated_rgb
                        )
                    else:
                        plt.imsave(os.path.join(eps_snapshot_dir, obs_file_name), rgb)
                    all_added_obj_ids += added_obj_ids

                # Clean up or merge redundant objects periodically
                scene.periodic_cleanup_objects(
                    frame_idx=cnt_step * total_views + view_idx, pts=pts
                )

                # Update depth map, occupancy map
                tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=cam_intr,
                    cam_pose=pose_habitat_to_tsdf(cam_pose),
                    obs_weight=1.0,
                    margin_h=int(cfg.margin_h_ratio * img_height),
                    margin_w=int(cfg.margin_w_ratio * img_width),
                    explored_depth=cfg.explored_depth,
                )

            # --- update explicit memory graph after TSDF update ---
            try:
                emg.update(
                    step=cnt_step,
                    tsdf_planner=tsdf_planner,
                    scene_objects=scene.objects,
                    agent_pts_habitat=pts,
                )
            except Exception as e:
                logging.warning(f"ExplicitMemoryGraph update failed: {e}")

            # (2) Update Memory Snapshots with hierarchical clustering
            # Choose all the newly added objects as well as the objects nearby as the cluster targets
            all_added_obj_ids = [
                obj_id for obj_id in all_added_obj_ids if obj_id in scene.objects
            ]
            for obj_id, obj in scene.objects.items():
                if (
                    np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]])
                    < cfg.scene_graph.obj_include_dist + 0.5
                ):
                    all_added_obj_ids.append(obj_id)
            scene.update_snapshots(
                obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection
            )
            logging.info(
                f"Step {cnt_step}, update snapshots, {len(scene.objects)} objects, {len(scene.snapshots)} snapshots"
            )

            # (3) Update the Frontier Snapshots
            update_success = tsdf_planner.update_frontier_map(
                pts=pts,
                cfg=cfg.planner,
                scene=scene,
                cnt_step=cnt_step,
                save_frontier_image=cfg.save_visualization,
                eps_frontier_dir=eps_frontier_dir,
                prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
            )
            if not update_success:
                logging.info("Warning! Update frontier map failed!")
                if cnt_step == 0:  # if the first step fails, we should stop
                    logging.info(
                        f"Question id {question_id} invalid: update_frontier_map failed!"
                    )
                    break

            # (4) Choose the next navigation point by querying the VLM
            if cfg.choose_every_step:
                # if we choose to query vlm every step, we clear the target point every step
                if (
                    tsdf_planner.max_point is not None
                    and type(tsdf_planner.max_point) == Frontier
                ):
                    # reset target point to allow the model to choose again
                    tsdf_planner.max_point = None
                    tsdf_planner.target_point = None

            if tsdf_planner.max_point is None and tsdf_planner.target_point is None:
                # query the VLM for the next navigation point, and the reason for the choice
                vlm_response = query_vlm_for_response(
                    question=question,
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                    rgb_egocentric_views=rgb_egocentric_views,
                    cfg=cfg,
                    verbose=True,
                )
                if vlm_response is None:
                    logging.info(
                        f"Question id {question_id} invalid: query_vlm_for_response failed!"
                    )
                    break

                max_point_choice, gpt_answer, n_filtered_snapshots = vlm_response

                # set the vlm choice as the navigation target
                update_success = tsdf_planner.set_next_navigation_point(
                    choice=max_point_choice,
                    pts=pts,
                    objects=scene.objects,
                    cfg=cfg.planner,
                    pathfinder=scene.pathfinder,
                    random_position=False,
                )
                if not update_success:
                    logging.info(
                        f"Question id {question_id} invalid: set_next_navigation_point failed!"
                    )
                    break

            # (5) Agent navigate to the target point for one step
            return_values = tsdf_planner.agent_step(
                pts=pts,
                angle=angle,
                objects=scene.objects,
                snapshots=scene.snapshots,
                pathfinder=scene.pathfinder,
                cfg=cfg.planner,
                path_points=None,
                save_visualization=cfg.save_visualization,
            )
            if return_values[0] is None:
                logging.info(f"Question id {question_id} invalid: agent_step failed!")
                break

            # update agent's position and rotation
            pts, angle, pts_voxel, fig, _, target_arrived = return_values
            logger.log_step(pts_voxel=pts_voxel)
            logging.info(f"Current position: {pts}, {logger.explore_dist:.3f}")

            # sanity check about objects, scene graph, snapshots, ...
            scene.sanity_check(cfg=cfg)

            if cfg.save_visualization:
                # save the top-down visualization
                logger.save_topdown_visualization(
                    cnt_step=cnt_step,
                    fig=fig,
                )
                # save the visualization of vlm's choice at each step
                logger.save_frontier_visualization(
                    cnt_step=cnt_step,
                    tsdf_planner=tsdf_planner,
                    max_point_choice=max_point_choice,
                    global_caption=f"{question}\n{answer}",
                )

            # (6) Check if the agent has arrived at the target to finish the question
            if type(max_point_choice) == SnapShot and target_arrived:
                # when the target is a snapshot, and the agent arrives at the target
                # we consider the question is finished and save the chosen target snapshot
                snapshot_filename = max_point_choice.image.split(".")[0]
                os.system(
                    f"cp {os.path.join(eps_snapshot_dir, max_point_choice.image)} {os.path.join(eps_chosen_snapshot_dir, f'snapshot_{snapshot_filename}.png')}"
                )

                task_success = True
                logging.info(
                    f"Question id {question_id} finished after arriving at target!"
                )
                break

        # --- save explicit memory graph at end of episode ---
        try:
            # Extract top_down_map using Scene's helper method
            top_down_map = None
            map_bounds = None
            meters_per_pixel = 0.025  # Default resolution
            exploration_masks = None  # Exploration awareness masks
            
            # Get trajectory from recorded voxel positions (needed for exploration-aware map)
            trajectory = None
            if hasattr(emg, '_trajectory_voxels') and emg._trajectory_voxels:
                trajectory = np.array(emg._trajectory_voxels) * cfg.tsdf_grid_size
                # Convert 2D voxel coords to 3D world coords [x, 0, z]
                if trajectory.shape[1] == 2:
                    trajectory_3d = np.zeros((len(trajectory), 3))
                    trajectory_3d[:, 0] = trajectory[:, 0]  # x
                    trajectory_3d[:, 2] = trajectory[:, 1]  # z
                    trajectory = trajectory_3d
            
            try:
                # Method 1: Use exploration-aware structured map V3 (best quality)
                if hasattr(scene, 'get_exploration_aware_topdown_map') and trajectory is not None:
                    logging.info(f"[Visualization] Attempting exploration-aware map with trajectory shape={trajectory.shape if trajectory is not None else 'None'}")
                    top_down_map, map_bounds, meters_per_pixel, exploration_masks = scene.get_exploration_aware_topdown_map(
                        trajectory=trajectory,
                        sensor_fov_deg=cfg.hfov if hasattr(cfg, 'hfov') else 90.0,
                        sensor_range_m=5.0,
                        meters_per_pixel=0.02,  # 更高分辨率
                        style="clean",  # 使用清晰风格
                        visited_radius_m=0.5,
                        show_frontier_glow=False,  # 关闭蓝色光晕
                        min_map_size=800,  # 确保场景足够大
                    )
                    if top_down_map is not None:
                        logging.info(f"[Visualization] Got exploration-aware top_down_map V3: shape={top_down_map.shape}, mpp={meters_per_pixel:.4f}")
                    else:
                        logging.warning("[Visualization] get_exploration_aware_topdown_map returned None!")
                
                # Method 2: Fallback to basic structured map
                if top_down_map is None and hasattr(scene, 'get_structured_topdown_map'):
                    logging.info("[Visualization] Falling back to get_structured_topdown_map...")
                    top_down_map, map_bounds, meters_per_pixel = scene.get_structured_topdown_map(
                        meters_per_pixel=0.025,
                        style="academic",
                        enable_distance_shading=True,
                    )
                    if top_down_map is not None:
                        logging.info(f"[Visualization] Got structured top_down_map: shape={top_down_map.shape}, mpp={meters_per_pixel:.4f}")
                    else:
                        logging.warning("[Visualization] get_structured_topdown_map returned None!")
                
                # Method 3: Fallback to old textured method
                if top_down_map is None and hasattr(scene, 'get_topdown_map'):
                    logging.info("[Visualization] Falling back to get_topdown_map...")
                    top_down_map, map_bounds = scene.get_topdown_map(colorize=True)
                    if top_down_map is not None:
                        logging.info(f"[Visualization] Got textured top_down_map: shape={top_down_map.shape}")
                    else:
                        logging.warning("[Visualization] get_topdown_map returned None!")
                
                # Method 4: Final fallback - raw pathfinder
                if top_down_map is None:
                    logging.info("[Visualization] Final fallback: using raw pathfinder...")
                    if hasattr(scene, 'pathfinder') and scene.pathfinder is not None:
                        try:
                            td = scene.pathfinder.get_topdown_view()
                            if td is not None:
                                top_down_map = np.stack([td, td, td], axis=-1).astype(np.uint8)
                                logging.info(f"[Visualization] Got pathfinder top_down_map: shape={top_down_map.shape}")
                            bounds = scene.pathfinder.get_bounds()
                            if bounds is not None:
                                map_bounds = (np.array(bounds[0]), np.array(bounds[1]))
                        except Exception as e:
                            logging.error(f"[Visualization] Pathfinder fallback failed: {e}")
                
                # Final check
                if top_down_map is None:
                    logging.error("[Visualization] ALL methods failed to generate top_down_map! Textured visualization will be skipped.")
                    
            except Exception as e:
                logging.error(f"[Visualization] Could not extract top_down_map: {e}")
                import traceback
                traceback.print_exc()
            
            # Get decision history from global conflict resolver
            decision_history = None
            try:
                from src.conceptgraph.slam.mapping import _global_conflict_resolver
                if hasattr(_global_conflict_resolver, 'decision_history'):
                    decision_history = _global_conflict_resolver.decision_history
            except Exception as e:
                logging.debug(f"Could not get decision_history: {e}")
            
            # Save with all visualization parameters
            emg.save(
                scene.objects,
                scene_snapshots=scene.snapshots,
                visualization_mode=None,  # Generate all modes
                top_down_map=top_down_map,
                agent_position=pts,
                agent_heading=angle,
                trajectory=trajectory,
                map_bounds=map_bounds,
                decision_history=decision_history,
                meters_per_pixel=meters_per_pixel,
                exploration_masks=exploration_masks,
            )
        except Exception as e:
            logging.warning(f"ExplicitMemoryGraph save failed: {e}")

        logger.log_episode_result(
            success=task_success,
            question_id=question_id,
            explore_dist=logger.explore_dist,
            gpt_answer=gpt_answer,
            n_filtered_snapshots=n_filtered_snapshots,
            n_total_snapshots=len(scene.snapshots),
            n_total_frames=len(scene.frames),
        )

        logging.info(f"Scene graph of question {question_id}:")
        logging.info(f"Question: {question}")
        logging.info(f"Answer: {answer}")
        logging.info(f"Prediction: {gpt_answer}")
        scene.print_scene_graph()

        # update the saved results after each episode
        logger.save_results()

        if not cfg.save_visualization:
            # clear up the stored images to save memory
            os.system(f"rm -r {episode_dir}")

    logger.save_results()
    # aggregate the results from different splits into a single file
    logger.aggregate_results()

    logging.info(f"All scenes finish")


if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(
        str(cfg.output_dir), f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}.log"
    )

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # Set up the logging format
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg, args.start_ratio, args.end_ratio)