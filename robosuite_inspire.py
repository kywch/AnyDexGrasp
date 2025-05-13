import os
import sys
import copy
import json
import random
import argparse
import datetime

import torch
import numpy as np
from PIL import Image
from graspnetAPI import GraspGroup
import robosuite.utils.camera_utils as CU

from AnyDexGrasp.models import minkowski_graspnet
from AnyDexGrasp.utils.collision_detector import ModelFreeCollisionDetectorMultifinger, load_meshes_pointcloud
from AnyDexGrasp.utils.graspnet_utils import GraspNetRunner, flip_ggarray, get_trial_info

from robosuite_env import make_robosuite_env, env_reset_get_camera_obs, execute_grasp
from InspireHandR_grasp import InspireHandRGraspGroup

from diverse_lift import AGOD_OBJECT_PATH
from diverse_lift.kitchen_objects import OBJ_CATEGORIES

# NOTE: This is to fix the error when loading the author's pre-trained models using torch.load()
sys.modules["minkowski_graspnet"] = minkowski_graspnet


DEBUG = False
RANDOM_GRASP = False

MAX_GRASP_WIDTH = 0.11
MIN_GRASP_WIDTH = 0.04

NUM_OF_INSPIRE_DEPTH = 2
NUM_OF_INSPIRE_TYPE = 8
INSPIREHANDR_VOXEL_GRID = 0.003

GRASP_SCORE_THRESHOLD = 0.85
COLLISION_APPROACH_DIST = 0.06

EEF_SEARCH_POS = np.array([0.10, -0.25, 0.15])  # camera can view the whole table
GRAB_SITE_OFFSET = np.array([-0.01, 0, 0])  # in the grip frame


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    # Using the downloaded checkpoint
    parser.add_argument("--checkpoint_path", default="logs/model/checkpoint.tar.18", help="Model checkpoint path")
    parser.add_argument(
        "--inspire_model_path",
        default="best_models/Inspire",
        # default="logs/model/inspire_model/obj140/480",  # authors' pre-trained models
        help="Inspire grasp decision model checkpoint path",
    )
    parser.add_argument(
        "--inspire_mesh_json_path",
        default="generate_mesh_and_pointcloud/inspire_urdf",
        help="InspireHandR meshes and json path",
    )
    parser.add_argument(
        "--result_file_prefix",
        default="inspire_eval",
        help="Prefix for the result file name",
    )
    parser.add_argument("--task", default="DiverseLift", help="Robosuite task")
    # parser.add_argument("--render", action="store_true", help="Render the scene")
    parser.add_argument("--render", default=True, help="Render the scene")
    parser.add_argument("--camera_name", default="robot0_eye_in_hand", help="Robosuite camera name to use")
    parser.add_argument("--num_multifinger_depth", default=2, type=int, help="Number of multifinger depth levels")

    parser.add_argument(
        "--mode",
        default="random",
        choices=["random", "agod", "objaverse", "prob_suggestion", "prob_execution"],
    )
    parser.add_argument("--num_trial", default=20, type=int, help="Number of trials (only for random)")

    args = parser.parse_args()

    return args


def get_graspgroup_features(grasp_features_array, sinput):
    grasp_features = dict()
    grasp_features["grasp_angles"] = (grasp_features_array[:, -3] + 0.1).astype(int)
    grasp_features["grasp_depths"] = (grasp_features_array[:, -2] * 100 + 0.1).astype(int)
    grasp_features["stage3_grasp_scores"] = grasp_features_array[:, :240]
    grasp_features["grasp_preds_features"] = grasp_features_array[:, 240 : 240 + 480]
    grasp_features["stage3_grasp_features"] = grasp_features_array[:, 240 + 480 : 240 + 480 + 512]
    grasp_features["before_generator"] = grasp_features_array[:, 240 + 480 + 512 : 240 + 480 + 512 + 512]
    grasp_features["point_features"] = grasp_features_array[:, 240 + 480 + 512 + 512 : 240 + 480 + 512 + 512 + 512]
    grasp_features["point_id"] = (grasp_features_array[:, -4] + 0.1).astype(int)
    grasp_features["if_flip"] = grasp_features_array[:, -1]
    grasp_features["view_inds"] = (grasp_features_array[:, -6] + 0.1).astype(int)
    grasp_features["view_score"] = grasp_features_array[:, -5]
    grasp_features["sinput"] = sinput

    grasp_preds_features_rot = np.zeros(grasp_features["grasp_preds_features"].shape, dtype=np.float32)
    new_type = grasp_features["grasp_angles"]
    for idx, if_flip in enumerate(grasp_features["if_flip"]):
        if if_flip:
            first_half_scores = copy.deepcopy(grasp_features["grasp_preds_features"][idx, :120])
            last_half_scores = copy.deepcopy(grasp_features["grasp_preds_features"][idx, 120:240])
            first_half_widths = copy.deepcopy(grasp_features["grasp_preds_features"][idx, 240:360])
            last_half_widths = copy.deepcopy(grasp_features["grasp_preds_features"][idx, 360:480])
            grasp_features["grasp_preds_features"][idx, :120] = last_half_scores
            grasp_features["grasp_preds_features"][idx, 120:240] = first_half_scores
            grasp_features["grasp_preds_features"][idx, 240:360] = last_half_widths
            grasp_features["grasp_preds_features"][idx, 360:480] = first_half_widths
        grasp_preds_features_rot[idx, : 240 - new_type[idx] * 5] = grasp_features["grasp_preds_features"][
            idx, new_type[idx] * 5 : 240
        ]
        grasp_preds_features_rot[idx, 240 - new_type[idx] * 5 : 240] = grasp_features["grasp_preds_features"][
            idx, 0 : new_type[idx] * 5
        ]
        grasp_preds_features_rot[idx, 240 : 480 - new_type[idx] * 5] = grasp_features["grasp_preds_features"][
            idx, 240 + new_type[idx] * 5 : 480
        ]
        grasp_preds_features_rot[idx, 480 - new_type[idx] * 5 : 480] = grasp_features["grasp_preds_features"][
            idx, 240 : 240 + new_type[idx] * 5
        ]
    grasp_features["grasp_preds_features"] = grasp_preds_features_rot

    return grasp_features


def get_inspire_model(inspire_models_path, num_inspire_depth=2, model_input_dim=480):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Sort the keys (grasp types), from 1 to 8
    sub_models = os.listdir(inspire_models_path)
    sub_models.sort()

    inspire_models = dict()
    for model_class in sub_models:
        model_classs_type = os.path.join(inspire_models_path, model_class)
        model_files = os.listdir(model_classs_type)
        inspire_model_type_path = os.path.join(model_classs_type, model_files[0])  # Just use 1 file
        # NOTE: torch.load() legacy files need sys.modules["minkowski_graspnet"] = minkowski_graspnet
        inspire_net = torch.load(inspire_model_type_path)

        inspire_model = minkowski_graspnet.MultifingerGraspSuccessPredictor(
            input_num=model_input_dim,
            num_multifinger_depth=num_inspire_depth,  # 2 for the new training artifacts, 4 for the original
        )

        try:
            inspire_model.load_state_dict(inspire_net["model_state_dict"])  # new training artifact
        except AttributeError:
            inspire_model.load_state_dict(inspire_net.state_dict())  # authors' pre-trained artifact

        inspire_model.to(device)
        inspire_model.eval()
        inspire_models[model_class] = inspire_model

    return inspire_models


def get_inspire_grasp(inspire_models, grasp_features_dict, ggarray, grasp_features):
    num_proposal = len(ggarray)
    if RANDOM_GRASP:
        inspire_type = np.random.randint(1, NUM_OF_INSPIRE_TYPE + 1, (num_proposal,))
        inspire_depth = np.random.randint(0, NUM_OF_INSPIRE_DEPTH, (num_proposal,))
        inspire_depth = inspire_depth * 0.01
        scores = ggarray[:, 0]
        return inspire_depth, inspire_type, scores, ggarray, grasp_features

    model_keys = list(inspire_models.keys())
    model_keys.sort()
    model_param = next(inspire_models[model_keys[0]].parameters())
    device = model_param.device

    for k, v in grasp_features_dict.items():
        if k == "point_id" or k == "sinput":
            continue
        grasp_features_dict[k] = torch.tensor(copy.deepcopy(v), device=device)

    inspire_depth_type_scores = []
    for key in model_keys:
        sub_inspire_model = inspire_models[key]
        with torch.no_grad():
            grasp_pred = sub_inspire_model(
                grasp_features_dict["grasp_preds_features"]
            )  # (B, 1， NUM_OF_TWO_FINGER_DEPTH*NUM_OF_INSPIRE_DEPTH)
            grasp_pred = grasp_pred.view(grasp_pred.shape[0], 5 * NUM_OF_INSPIRE_DEPTH)

        two_fingers_depth = grasp_features_dict["grasp_depths"]  # (B, )
        base = torch.tensor(
            np.array([[i for i in range(NUM_OF_INSPIRE_DEPTH)] for _ in range(grasp_pred.size()[0])]), device=device
        )  # (B, NUM_OF_INSPIRE_DEPTH)
        select_index = (two_fingers_depth).view(-1, 1) * NUM_OF_INSPIRE_DEPTH + base
        inspire_depth_type_scores.append(grasp_pred.gather(1, select_index))

    inspire_depth_type_scores = torch.cat(inspire_depth_type_scores, axis=1).view(
        -1
    )  # (B, NUM_OF_INSPIRE_DEPTH*NUM_OF_INSPIRE_TYPE)

    scores, index = inspire_depth_type_scores.topk(num_proposal * 3)

    pose_index = (index / (NUM_OF_INSPIRE_DEPTH * NUM_OF_INSPIRE_TYPE)).long()
    ggarray = torch.tensor(copy.deepcopy(ggarray), device=device)[pose_index]
    grasp_features = torch.tensor(copy.deepcopy(grasp_features), device=device)[pose_index]

    # NOTE: Assume model_keys are sorted.
    inspire_type = ((index % (NUM_OF_INSPIRE_DEPTH * NUM_OF_INSPIRE_TYPE)) / NUM_OF_INSPIRE_DEPTH).int()  # 0-7
    inspire_depth = ((index % (NUM_OF_INSPIRE_DEPTH * NUM_OF_INSPIRE_TYPE)) % NUM_OF_INSPIRE_DEPTH).int()  # 0, 1

    return (
        inspire_type.cpu().numpy() + 1,  # Make it 1-8
        inspire_depth.cpu().numpy(),
        scores.detach().cpu().numpy(),
        ggarray.cpu().numpy(),
        grasp_features.cpu().numpy(),
    )


def choose_grasp(
    graspnet_runner,
    inspire_models,
    depth_map,
    camera,
    object_mask,
    inspire_mesh_json_path,
    meshes_pcls,
    min_grasp_width=MIN_GRASP_WIDTH,
    max_try_sample=5,
    visualize=False,
):
    cnt_try_sample = 0

    while True:
        cnt_try_sample += 1
        if cnt_try_sample > max_try_sample:
            return None, None, None, None, None

        ### Sample ggarray
        print(f"Sampling grasp, try {cnt_try_sample} ...")
        if cnt_try_sample <= 1:
            ggarray, points_down, grasp_features, sinput = graspnet_runner.get_grasp(
                depth_map, camera, object_mask, visualize=visualize
            )
        else:
            ggarray, points_down, grasp_features, sinput = graspnet_runner.get_ggarray_features(
                depth_map, camera, object_mask, num_augment=cnt_try_sample
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

        # Filter proposals with if_flip=True
        ggarray = ggarray[~np.array(if_flip)]
        grasp_features = grasp_features[~np.array(if_flip)]
        grasp_features_dict = get_graspgroup_features(grasp_features, sinput)

        inspire_type, inspire_depth, scores, ggarray, grasp_features = get_inspire_grasp(
            inspire_models, grasp_features_dict, ggarray, grasp_features
        )

        if not RANDOM_GRASP:
            # NOTE: scores come from grasp decision models
            mask = scores > GRASP_SCORE_THRESHOLD
            inspire_type = inspire_type[mask]
            inspire_depth = inspire_depth[mask]
            scores = scores[mask]
            ggarray = ggarray[mask]
            grasp_features = grasp_features[mask]

        if len(ggarray) == 0:
            print(f"There is no grasp that score greater than {GRASP_SCORE_THRESHOLD}")
            continue

        ### Graspnet returns two finger grasps, so init inspire grasps from these
        two_fingers_ggarray = GraspGroup(ggarray)
        InspireHandR_ggarray = InspireHandRGraspGroup()
        InspireHandR_ggarray.set_grasp_min_width(min_grasp_width)
        InspireHandR_ggarray.from_graspgroup(two_fingers_ggarray, inspire_type, inspire_mesh_json_path)
        InspireHandR_ggarray.depths += inspire_depth * 0.01
        InspireHandR_ggarray.scores = scores

        ### Filter by z-axis -- it's already done inside GraspNetRunner.get_grasp()
        # z_filter_mask = InspireHandR_ggarray.filter_grasp_group_by_z_axis(0.4)
        # two_fingers_ggarray = two_fingers_ggarray[z_filter_mask]
        # grasp_features = grasp_features[z_filter_mask]
        # inspire_depth = inspire_depth[z_filter_mask]

        # if len(InspireHandR_ggarray) == 0:
        #     print("No grasp detected after z-axis filter")
        #     continue

        ### Check collision
        mfcdetector = ModelFreeCollisionDetectorMultifinger(points_down.cpu().numpy())
        collision_free_mask = mfcdetector.detect(
            InspireHandR_ggarray,
            two_fingers_ggarray,
            inspire_mesh_json_path,
            meshes_pcls,
            min_grasp_width=min_grasp_width,
            downsample_voxel_size=INSPIREHANDR_VOXEL_GRID,
            approach_dist=COLLISION_APPROACH_DIST,
            collision_thresh=0,
            adjust_gripper_centers=False,
        )

        InspireHandR_ggarray = InspireHandR_ggarray[collision_free_mask]
        two_fingers_ggarray = two_fingers_ggarray[collision_free_mask]
        grasp_features = grasp_features[collision_free_mask]
        inspire_depth = inspire_depth[collision_free_mask]

        if len(InspireHandR_ggarray) == 0:
            print("No grasp detected after collision detection")
            continue

        # OK, we got some grasp candidates to try
        break

    return InspireHandR_ggarray, two_fingers_ggarray, grasp_features, inspire_depth, points_down


ALL_OBJAVERSE_PATH = [
    p
    for k in OBJ_CATEGORIES.keys()
    if "objaverse" in OBJ_CATEGORIES[k]
    for p in OBJ_CATEGORIES[k]["objaverse"].mjcf_paths
]


def key_in_path(path, key):
    for k in key:
        if k in path:
            return True
    return False


PROB_SUGGEST_OBJAVERSE = [
    "water_bottle",
    "cupcake",
    "bottled_",
    "spray_0",
    "fish_2",
    "eggplant_2",
    "cheese_1",
    "cheese_2",
    "corn_1",
    "wine_12",
    "baguette_1",
    "ketchup_0",
    "kettle_0",
]
PROB_SUGGEST_PATH = [p for p in ALL_OBJAVERSE_PATH if key_in_path(p, PROB_SUGGEST_OBJAVERSE)] + [
    os.path.join(p, "model.xml")
    for p in AGOD_OBJECT_PATH
    if key_in_path(p, ["24891", "23279", "20123", "22262", "22941"])
] * 4

PROB_EXECUTE_PATH = [
    p for p in ALL_OBJAVERSE_PATH if key_in_path(p, ["kettle", "pan", "teapot", "cereal", "donut"])
] + [
    os.path.join(p, "model.xml")
    for p in AGOD_OBJECT_PATH
    if key_in_path(p, ["23156", "24806", "21352", "24466", "70799", "22017"])
] * 4


def set_next_trial(cfgs, robot_env, n):
    # NOTE: use DiverseLift's config_next_sample
    is_done = cfgs.num_trial <= n

    if not hasattr(robot_env, "config_next_sample"):
        pass

    elif cfgs.mode == "random":
        if n % 2 == 0:
            robot_env.config_next_sample(source="objaverse")
        else:
            random_quat = int(bool(n % 3))  # 33% 0 or 67% 1
            robot_env.config_next_sample(source="agod", prob_random_quat=random_quat)

    elif cfgs.mode == "agod":
        num_agod = len(AGOD_OBJECT_PATH)
        # repeat 3 times
        is_done = 3 * num_agod <= n
        random_quat = int(bool(n // num_agod))
        path_idx = os.path.join(AGOD_OBJECT_PATH[n % num_agod], "model.xml")
        robot_env.config_next_sample(source="agod", group_or_index=path_idx, prob_random_quat=random_quat)

    elif cfgs.mode == "objaverse":
        num_objaverse = len(ALL_OBJAVERSE_PATH)
        is_done = num_objaverse <= n
        robot_env.config_next_sample(source="objaverse", group_or_index=ALL_OBJAVERSE_PATH[n % num_objaverse])

    elif cfgs.mode == "prob_suggestion":
        num_prob = len(PROB_SUGGEST_PATH)
        is_done = 3 * num_prob <= n
        idx = n % num_prob
        source = "objaverse" if "objaverse" in PROB_SUGGEST_PATH[idx] else "agod"
        robot_env.config_next_sample(source=source, group_or_index=PROB_SUGGEST_PATH[idx])

    if cfgs.mode == "prob_execution":
        num_prob = len(PROB_EXECUTE_PATH)
        is_done = 3 * num_prob <= n
        idx = n % num_prob
        source = "objaverse" if "objaverse" in PROB_EXECUTE_PATH[idx] else "agod"
        robot_env.config_next_sample(source=source, group_or_index=PROB_EXECUTE_PATH[idx])

    return is_done


def run_eval(cfgs):
    inspire_models = get_inspire_model(cfgs.inspire_model_path)
    meshes_pcls = load_meshes_pointcloud(cfgs.inspire_mesh_json_path, voxel_grid=INSPIREHANDR_VOXEL_GRID)

    # Setup env
    exp_data, by_object = [], {}
    robot_env = make_robosuite_env(cfgs.task, camera_name=cfgs.camera_name, render=cfgs.render)

    camera, _ = env_reset_get_camera_obs(robot_env, cfgs.camera_name, init_eef_pos=EEF_SEARCH_POS)
    graspnet_runner = GraspNetRunner(
        camera, cfgs.checkpoint_path, max_grasp_width=MAX_GRASP_WIDTH, min_grasp_width=MIN_GRASP_WIDTH
    )

    num_success = 0
    n, is_done = 0, False
    while not is_done:
        n += 1
        print("Trial", n)
        is_done = set_next_trial(cfgs, robot_env, n)

        camera, obs_dict = env_reset_get_camera_obs(robot_env, cfgs.camera_name, init_eef_pos=EEF_SEARCH_POS)

        target_obj = getattr(robot_env, "target_object", "cube")
        if target_obj not in by_object:
            by_object[target_obj] = {
                "count": 1,
                "no_valid_grasp_proposal": 0,
                "success": 0,
            }
        else:
            by_object[target_obj]["count"] += 1

        # if DEBUG:
        # Image.fromarray(obs_dict["{}_image".format(cfgs.camera_name)][::-1]).show()

        # if cfgs.render:
        #     input("Press Enter to continue...")

        # Object segmentation
        object_mask = robot_env.get_object_mask()

        depth_map = CU.get_real_depth_map(
            sim=robot_env.sim, depth_map=obs_dict["{}_depth".format(cfgs.camera_name)][::-1]
        ).squeeze()

        InspireHandR_ggarray, two_fingers_ggarray, grasp_features, inspire_depth, points_down = choose_grasp(
            graspnet_runner,
            inspire_models,
            depth_map,
            camera,
            object_mask,
            inspire_mesh_json_path=cfgs.inspire_mesh_json_path,
            meshes_pcls=meshes_pcls,
            visualize=cfgs.render,
        )

        result = {
            "object": target_obj,
            "used_grasp_type": None,
            "is_success": None,
        }

        # Could not get grasp from this depth map
        if InspireHandR_ggarray is None:
            print("Could not get grasp in this trial ... Skip.\n")
            by_object[target_obj]["no_valid_grasp_proposal"] += 1
            exp_data.append(result)
            continue

        # Randomly pick one among top 5
        grasp_idx = random.randint(0, min(len(InspireHandR_ggarray) - 1, 5))
        InspireHandR_grasp_used = InspireHandR_ggarray[grasp_idx]
        two_fingers_grasp_used = two_fingers_ggarray[grasp_idx]
        grasp_features_used = grasp_features[grasp_idx]
        inspire_depth_used = int(inspire_depth[grasp_idx])

        print(
            f"Executing a grasp ... (grasp type: {int(InspireHandR_grasp_used.grasp_type)}, depth: {inspire_depth_used}, score: {InspireHandR_grasp_used.score:.4f})"
        )
        is_success = execute_grasp(
            robot_env,
            camera,
            InspireHandR_grasp_used,
            two_fingers_grasp_used,
            grab_site_offset=GRAB_SITE_OFFSET,
        )
        num_success += is_success
        print(f"Result: {'success' if is_success else 'fail'}. So far {num_success} / {n}\n")
        by_object[target_obj]["success"] += is_success

        result["used_grasp_type"] = int(InspireHandR_grasp_used.grasp_type)
        result["used_multifinger_depth"] = inspire_depth_used
        result["used_grasp_score"] = InspireHandR_grasp_used.score
        result["top5_grasp"] = [(int(g.grasp_type), g.score) for g in InspireHandR_ggarray[:5]]
        result["is_success"] = is_success
        result["trial_info"] = get_trial_info(two_fingers_grasp_used, InspireHandR_grasp_used, grasp_features_used)

        exp_data.append(result)

    robot_env.close()

    return exp_data, by_object


if __name__ == "__main__":
    cfgs = parse_arguments()
    results, by_object = run_eval(cfgs)

    results_by_type = {}
    suggested = {}
    for t in results:
        if t["is_success"] is None:  # No grasp found, so the trial skipped
            continue

        if t["used_grasp_type"] not in results_by_type:
            results_by_type[t["used_grasp_type"]] = {"count": 1, "success": t["is_success"]}
        else:
            results_by_type[t["used_grasp_type"]]["count"] += 1
            results_by_type[t["used_grasp_type"]]["success"] += t["is_success"]

        if "top5_grasp" in t:
            for g in t["top5_grasp"]:
                if g[0] not in suggested:
                    suggested[g[0]] = 1
                else:
                    suggested[g[0]] += 1

    results_by_type = {k: v for k, v in sorted(results_by_type.items())}
    for k, v in results_by_type.items():
        results_by_type[k]["success_rate"] = v["success"] / v["count"]

    result_file = f"eval_{cfgs.mode}_{datetime.datetime.now().strftime('%m%d-%H%M%S')}.json"
    result_dict = cfgs.__dict__
    result_dict["overall_success_rate"] = np.mean([r["is_success"] for r in results if r["is_success"] is not None])
    result_dict["results_by_type"] = results_by_type
    result_dict["suggested"] = {k: v for k, v in sorted(suggested.items())}
    result_dict["results_by_object"] = by_object
    result_dict["results"] = results
    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=4)
