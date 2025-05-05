import copy
import pickle
import argparse
import datetime

import numpy as np
from scipy.spatial.transform import Rotation

import open3d as o3d
from PIL import Image

from graspnetAPI import GraspGroup
import robosuite.utils.camera_utils as CU

from AnyDexGrasp.utils.collision_detector import ModelFreeCollisionDetectorMultifinger, load_meshes_pointcloud
from AnyDexGrasp.utils.graspnet_utils import GraspNetRunner, get_grasp_features, flip_ggarray

from robosuite_env import make_robosuite_env, RobosuiteCameraInfo, execute_grasp
from InspireHandR_grasp import GRASP_TYPES, InspireHandRGraspGroup


DEBUG = False

MAX_GRASP_WIDTH = 0.1
MIN_GRASP_WIDTH = 0.02
INSPIREHANDR_VOXEL_GRID = 0.002
COLLISION_APPROACH_DIST = 0.04
NUM_OF_INSPIRE_DEPTH = 4

GRAB_SITE_OFFSET = np.array([-0.01, 0, 0])  # in the grip frame


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    # Using the downloaded checkpoint
    parser.add_argument("--checkpoint_path", default="logs/model/checkpoint.tar.18", help="Model checkpoint path")
    parser.add_argument(
        "--save_information_path",
        default="logs/data/decision_model/inspire/obj140/collect",
        help="inspire model result information path",
    )
    parser.add_argument(
        "--inspire_mesh_json_path",
        default="generate_mesh_and_pointcloud/inspire_urdf",
        help="InspireHandR meshes and json path",  # generated
    )
    parser.add_argument("--render", action="store_true", help="Render the scene")
    parser.add_argument("--camera_name", default="robot0_eye_in_hand", help="Robosuite camera name to use")
    parser.add_argument("--num_trial_per_type", default=10, help="Number of trials per grasp type")
    args = parser.parse_args()

    return args


def get_trial_info(
    two_fingers_grasp_used,
    InspireHandR_grasp_used,
    grasp_features_used,
):
    tfg = two_fingers_grasp_used
    two_fingers_array = (
        [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)]
        + np.array(tfg.rotation_matrix).reshape((-1)).tolist()
        + list(tfg.translation.reshape(-1).tolist())
        + [float(InspireHandR_grasp_used.object_id)]
    )
    grasp_features_dict = get_grasp_features(grasp_features_used)

    return {
        "grasp_preds_features": grasp_features_dict["grasp_preds_features"],
        "two_fingers_pose": list(two_fingers_array),
        "InspiredHandR_pose": list(InspireHandR_grasp_used.get_array_grasp()),
        "two_fingers_pose_angle_type": grasp_features_dict["grasp_angles"],
        "two_fingers_pose_depth_type": grasp_features_dict["grasp_depths"],
        "if_flip": grasp_features_dict["if_flip"],
        "InspiredHandR_pose_finger_type": int(InspireHandR_grasp_used.grasp_type + 0.1),
    }


def show_scene_cloud(points, multifinger_mesh=None, two_finger_mesh=None, trasform_matrix=None, width=1024, height=640):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
    scene_cloud = o3d.geometry.PointCloud()
    scene_cloud.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    scene_cloud = scene_cloud.voxel_down_sample(0.001)
    scene_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    geoms = [scene_cloud, frame, sphere]

    if multifinger_mesh is not None:
        geoms.append(multifinger_mesh)

    if two_finger_mesh is not None:
        geoms.append(two_finger_mesh)

    if trasform_matrix is not None:
        trans_geoms = []
        for g in geoms:
            g_copy = copy.deepcopy(g)
            g_copy.transform(trasform_matrix)
            trans_geoms.append(g_copy)
        geoms = trans_geoms

    o3d.visualization.draw_plotly(geoms, width=width, height=height)


def sample_grasp(
    graspnet_runner,
    depth_map,
    grasp_type,
    grasp_depth,
    inspire_mesh_json_path,
    meshes_pcls,
    min_grasp_width=MIN_GRASP_WIDTH,
    max_try_sample=5,
):
    cnt_try_sample = 0

    while True:
        cnt_try_sample += 1
        if cnt_try_sample > max_try_sample:
            return None, None, None, None

        ### Sample ggarray
        print(f"Sampling grasp, try {cnt_try_sample} ...")
        if cnt_try_sample <= 1:
            ggarray, points_down, grasp_features, _ = graspnet_runner.get_grasp(depth_map)
        else:
            ggarray, points_down, grasp_features, _ = graspnet_runner.get_ggarray_features(
                depth_map, num_augment=cnt_try_sample
            )

        if ggarray is None or len(ggarray) == 0:
            continue

        ### Prepare variables
        # Sort by score (ggaray col 0), and process the top 500 only
        ggarray = ggarray.cpu().numpy()
        sorted_idx = ggarray[:, 0].argsort()
        ggarray = ggarray[sorted_idx][::-1][:500]

        grasp_features = grasp_features.cpu().numpy()
        grasp_features = grasp_features[sorted_idx][::-1][:500]
        ggarray, if_flip = flip_ggarray(ggarray)
        grasp_features = np.c_[grasp_features, if_flip]

        InspireHandR_types = np.array([grasp_type] * len(ggarray))
        InspireHandR_depths = grasp_depth * 0.01
        # if grasp_type == 5:  # Medium_Wrap
        #     InspireHandR_depths = InspireHandR_depths + 0.02

        ### Graspnet returns two finger grasps, so init inspire grasps from these
        two_fingers_ggarray = GraspGroup(ggarray)
        InspireHandR_ggarray = InspireHandRGraspGroup()
        InspireHandR_ggarray.set_grasp_min_width(min_grasp_width)
        InspireHandR_ggarray.from_graspgroup(two_fingers_ggarray, InspireHandR_types, inspire_mesh_json_path)

        # NOTE: Initially, two finger grasp depths are set. Then add random InspireHandR_depths on top to experiment.
        InspireHandR_ggarray.depths += InspireHandR_depths

        ### Filter by z-axis
        z_filter_mask = InspireHandR_ggarray.filter_grasp_group_by_z_axis(0.4)
        two_fingers_ggarray = two_fingers_ggarray[z_filter_mask]
        grasp_features = grasp_features[z_filter_mask]

        if len(InspireHandR_ggarray) == 0:
            print("No grasp detected after z-axis filter")
            continue

        ### Check collision
        # TODO: refactor this? (low priority)
        mfcdetector = ModelFreeCollisionDetectorMultifinger(points_down.cpu().numpy())
        collision_free_mask = mfcdetector.detect(
            InspireHandR_ggarray,
            two_fingers_ggarray,
            inspire_mesh_json_path,
            meshes_pcls,
            min_grasp_width=MIN_GRASP_WIDTH,
            downsample_voxel_size=INSPIREHANDR_VOXEL_GRID,
            approach_dist=COLLISION_APPROACH_DIST,
            collision_thresh=1,
            adjust_gripper_centers=False,
        )

        InspireHandR_ggarray = InspireHandR_ggarray[collision_free_mask]
        two_fingers_ggarray = two_fingers_ggarray[collision_free_mask]
        grasp_features = grasp_features[collision_free_mask]

        if len(InspireHandR_ggarray) == 0:
            print("No grasp detected after collision detection")
            continue

        # OK, we got some grasp candidates to try
        break

    return InspireHandR_ggarray, two_fingers_ggarray, grasp_features, points_down


def env_reset_and_get_obs(robot_env, eef_offset=[0.10, -0.20, 0.10], camera_height=720, camera_width=1280):
    robot_env.reset()

    # The initial pose
    ref_id = robot_env.sim.model.site_name2id("gripper0_right_grip_site")
    eef_pos = robot_env.sim.data.site_xpos[ref_id]
    eef_ori_mat = robot_env.sim.data.site_xmat[ref_id].reshape((3, 3))
    eef_ori_aa = Rotation.from_matrix(eef_ori_mat).as_rotvec()

    search_pose = np.zeros(7)  # OSC_POSE
    search_pose[:3] = eef_pos + np.array(eef_offset)
    search_pose[3:6] = eef_ori_aa
    search_pose[6] = 1

    # Get the hand out of the camera view
    for _ in range(50):
        obs_dict, _, _, _ = robot_env.step(search_pose)

    camera = RobosuiteCameraInfo(
        robot_env.sim, cfgs.camera_name, camera_height=camera_height, camera_width=camera_width
    )

    return camera, obs_dict


if __name__ == "__main__":
    cfgs = parse_arguments()

    inspire_grasp_types = [int(x) for x in GRASP_TYPES.keys()]

    run_summary, exp_data = {}, {}
    for t in inspire_grasp_types:
        for d in range(NUM_OF_INSPIRE_DEPTH):
            run_summary[(t, d)] = {
                "name": GRASP_TYPES[str(t)]["name"],
                "inspire_depth": d,
                "count": 0,
                "no_valid_grasp_proposal": 0,
                "success": 0,
            }

    # Contact-centric grasp representation
    meshes_pcls = load_meshes_pointcloud(cfgs.inspire_mesh_json_path, voxel_grid=INSPIREHANDR_VOXEL_GRID)

    # Setup env
    robot_env = make_robosuite_env(task="Lift", camera_name=cfgs.camera_name, render=cfgs.render)

    camera = RobosuiteCameraInfo(robot_env.sim, cfgs.camera_name, camera_height=720, camera_width=1280)
    graspnet_runner = GraspNetRunner(
        camera, cfgs.checkpoint_path, max_grasp_width=MAX_GRASP_WIDTH, min_grasp_width=MIN_GRASP_WIDTH
    )

    for n in range(cfgs.num_trial_per_type):
        for grasp_type in inspire_grasp_types:
            for grasp_depth in range(NUM_OF_INSPIRE_DEPTH):
                camera, obs_dict = env_reset_and_get_obs(robot_env)

                if DEBUG:
                    Image.fromarray(obs_dict["{}_image".format(cfgs.camera_name)][::-1]).show()

                # grasp_type = np.random.choice(inspire_grasp_types)
                # grasp_depth = np.random.randint(NUM_OF_INSPIRE_DEPTH)
                run_summary[(grasp_type, grasp_depth)]["count"] += 1
                print(f"Round {n+1}, grasp_type: {GRASP_TYPES[str(grasp_type)]['name']}, grasp_depth: {grasp_depth}")

                if cfgs.render:
                    input("Press Enter to continue...")

                depth_map = CU.get_real_depth_map(
                    sim=robot_env.sim, depth_map=obs_dict["{}_depth".format(cfgs.camera_name)][::-1]
                ).squeeze()

                InspireHandR_ggarray, two_fingers_ggarray, grasp_features, points_down = sample_grasp(
                    graspnet_runner,
                    depth_map,
                    grasp_type=grasp_type,
                    grasp_depth=grasp_depth,
                    inspire_mesh_json_path=cfgs.inspire_mesh_json_path,
                    meshes_pcls=meshes_pcls,
                )

                # Could not get grasp from this depth map
                if InspireHandR_ggarray is None:
                    print("Could not get grasp in this trial ... Skip.\n")
                    run_summary[(grasp_type, grasp_depth)]["no_valid_grasp_proposal"] += 1
                    continue

                # Grasp proposals are already sorted, so pick the "0" index (one with the highest score)
                InspireHandR_grasp_used = InspireHandR_ggarray[0]
                two_fingers_grasp_used = two_fingers_ggarray[0]
                grasp_features_used = grasp_features[0]

                if DEBUG:
                    inspire_mesh = InspireHandR_grasp_used.load_mesh(
                        cfgs.inspire_mesh_json_path, two_fingers_grasp_used
                    )
                    inspire_mesh.paint_uniform_color([1, 0, 0])
                    two_finger_mesh = two_fingers_grasp_used.to_open3d_geometry()
                    show_scene_cloud(
                        points_down, inspire_mesh, two_finger_mesh, trasform_matrix=camera.camera_to_world_mat
                    )

                print("Executing a grasp ...")
                execute_grasp(
                    robot_env,
                    camera,
                    InspireHandR_grasp_used,
                    two_fingers_grasp_used,
                    grab_site_offset=GRAB_SITE_OFFSET,
                )
                result = int(robot_env.reward())

                print("Result:", "success" if result else "fail", "\n")
                run_summary[(grasp_type, grasp_depth)]["success"] += result

                trial_info = get_trial_info(two_fingers_grasp_used, InspireHandR_grasp_used, grasp_features_used)
                trial_info["InspiredHandR_pose_depth_type"] = grasp_depth
                trial_info["result"] = result  # Grasp & Lift success 1, otherwise 0

                trial_key = datetime.datetime.now().strftime("%m%d-%H%M%S")
                exp_data[trial_key] = trial_info

    result_file = f"grasp_data_{datetime.datetime.now().strftime('%m%d-%H%M%S')}.pkl"
    with open(result_file, "wb") as f:
        pickle.dump(
            {
                "run_summary": run_summary,
                "exp_data": exp_data,
            },
            f,
        )

    # Get trial info covers the most, but need to add multifinger_pose_depth_type and result

    # Below is the data required to train the grasp decision models
    """
    end_points = {}
    end_points["two_fingers_pose_angle_type"] = batch_data_label["two_fingers_pose_angle_type"]
    end_points["two_fingers_pose_depth_type"] = batch_data_label["two_fingers_pose_depth_type"]
    end_points["multifinger_pose_finger_type"] = batch_data_label["multifinger_pose_finger_type"]
    end_points["multifinger_pose_depth_type"] = batch_data_label["multifinger_pose_depth_type"]
    end_points["grasp_preds_features"] = batch_data_label["grasp_preds_features"]
    end_points["if_flip"] = batch_data_label["if_flip"]
    end_points["result"] = batch_data_label["result"]  # Grasp success 1, fail 0
    """

    # TODO: after executing grasp, save relevant information for training
    # Repeat this for 1000 trials per grasp type
    # The env should have multiple objects with a clean-up task
    # Sampling diverse geometry is the key

    print()

    # pass

    # t0 = time.time()
    # try:
    #     robot_grasp(cfgs)
    # finally:
    #     tn = time.time()
    #     print(f"total time:{tn - t0}")
