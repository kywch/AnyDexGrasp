import time
import json
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

import mujoco
import robosuite
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.fixed_base_robot import FixedBaseRobot
from robosuite.models.robots import Panda
from robosuite.models.robots.robot_model import register_robot
from robosuite.models.grippers import register_gripper
from robosuite.models.grippers.inspire_hands import InspireRightHand

import robosuite.utils.transform_utils as T
import robosuite.utils.camera_utils as CU
from robosuite.utils.mjcf_utils import find_elements
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

# Define custom robot
# Use keyboard, to control OSC delta controller  --- mink?
# Print out the 6 dof actions for the arm

from InspireHandR_grasp import GRASP_TYPES

INSPIRE_WIDTH_ANGLE_JSON = "generate_mesh_and_pointcloud/inspire_urdf/width_12Dangle_6Dangle.json"


class CustomInspireRightHand(InspireRightHand):
    def __init__(self, idn=0):
        super().__init__(idn)  # Use Rososuite's inspire right hand xml

        # Update visualization
        for site in ["ee_x", "ee_y", "ee_z", "grip_site_cylinder"]:
            site_id = self._sites.index(site)
            rgba = self._elements["sites"][site_id].attrib["rgba"][:-1]
            rgba += "1" if "ee" in site else "0"
            self._elements["sites"][site_id].attrib["rgba"] = rgba

        with open(INSPIRE_WIDTH_ANGLE_JSON, "r", encoding="utf-8") as f:
            self._width_angle_dict = json.load(f)

        # Get control range
        ctrl_range = np.array(
            [[float(x) for x in ac.attrib["ctrlrange"].split(" ")] for ac in self._elements["actuators"]]
        )
        self.control_range = (ctrl_range[:, 0] - ctrl_range[:, 1]) / 1000
        self.control_base = ctrl_range[:, 1]

        # Convert 6d to 12d: from robosuite/models/grippers/inspire_hands.py, InspireRightHand.format_action()
        self._map_6d_to_12d = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5])

        # Convert the json file finger pos (0, 1000) to the 12-dof control signal
        self.grip_info = {}
        for grip_type in self._width_angle_dict:
            self.grip_info[grip_type] = {"valid_widths": []}

            for width in self._width_angle_dict[grip_type]:
                # if 6d is -1, then it's not valid
                action_6d = self._width_angle_dict[grip_type][width]["6d"]
                if action_6d[0] > -1:
                    self.grip_info[grip_type]["valid_widths"].append(width)
                    self._width_angle_dict[grip_type][width]["12d"] = self._convert_6d_to_12d(action_6d)

            self.grip_info[grip_type]["idx_scale"] = (len(self.grip_info[grip_type]["valid_widths"]) - 1) / 2.0

        # Set the gripper type
        self._grip_type = "Ring"
        assert self._grip_type in self.grip_info, "Gripper type {} not found in gripper info!".format(self._grip_type)

        # See AnyDexGrasp mesh generation, open3d for this
        self.eef_to_wrist_hmat = np.eye(4)
        self.eef_to_wrist_hmat[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

        # Offset to apply to the gripper site. TODO: try to get rid of this
        self._grab_site_offset = np.zeros(3)

    def _convert_6d_to_12d(self, action_6d):
        # NOTE: It probably isn't linear like below. AnyDexGrasp used driver_routine_to_angle.xls to map
        return np.array(action_6d)[self._map_6d_to_12d] * self.control_range + self.control_base

    @property
    def grip_type(self):
        return self._grip_type

    def set_grip_type(self, grip_type):
        assert grip_type in self.grip_info, "Gripper type {} not found in gripper info!".format(grip_type)
        self._grip_type = grip_type

    def set_grab_site_offset(self, offset):
        assert len(offset) == 3, "Offset must be a 3-element array"
        self._grab_site_offset = np.array(offset)

    def get_grip_to_wrist_hmat(self, grip_type):
        action_to_idx = 10  # int(self.grip_info[self._grip_type]["idx_scale"])  # middle value
        width_key = self.grip_info[grip_type]["valid_widths"][action_to_idx]

        # Apply manual rot offset (z axis, +20 deg) to the given rotation mat
        rot_offset = T.quat2mat(np.array([0, 0, 0.174, 0.985]))

        rot_mat = rot_offset @ np.array(self._width_angle_dict[grip_type][width_key]["rotation"])

        # get the gripper offset in the wrist frame
        offset = rot_mat @ self._grab_site_offset

        trans = np.array(self._width_angle_dict[grip_type][width_key]["translation"]) + offset
        trans[0] -= 0.0078  # subtract the ring of metal, so set the origin to the center of wrist

        return T.make_pose(trans, rot_mat)  # 4x4 mat

    @lru_cache
    def get_grip_hmat(self, grip_type):
        # return self.get_grip_to_wrist_hmat(grip_type)
        return self.eef_to_wrist_hmat @ self.get_grip_to_wrist_hmat(grip_type)

    @lru_cache
    def get_inv_grip_hmat(self, grip_type):
        return np.linalg.inv(self.get_grip_hmat(grip_type))

    def get_grab_site_from_curr_eef(self, env):
        ref_id = env.sim.model.site_name2id("gripper0_right_grip_site")
        curr_ee_hmat = T.make_pose(env.sim.data.site_xpos[ref_id], env.sim.data.site_xmat[ref_id].reshape((3, 3)))

        grip_hmat = self.get_grip_hmat(self._grip_type)

        grab_site_hmat = curr_ee_hmat @ grip_hmat
        grab_pos, grab_ori_quat = T.mat2pose(grab_site_hmat)
        grab_ori_mat = T.quat2mat(grab_ori_quat)

        # Apply offset to the target gripper site
        return grab_pos, grab_ori_mat

    def get_eef_pose_for_grab(self, grab_pos, grab_ori_mat):
        """Given grab pos and ori_aa, return the eef pose to feed to the controller"""

        # Apply offset to the target gripper site
        grab_site_hmat = T.make_pose(grab_pos, grab_ori_mat)

        eef_hmat = grab_site_hmat @ self.get_inv_grip_hmat(self._grip_type)
        eef_pos, eef_quat = T.mat2pose(eef_hmat)
        eef_ori_aa = T.quat2axisangle(eef_quat)

        return eef_pos, eef_ori_aa

    def get_gripper_width_to_action(self, width):
        width_key = str(np.round(width * 100, 1))
        assert width_key in self.grip_info[self._grip_type]["valid_widths"], "Width {} not found in gripper info!"

        # assume valid_widths are sorted
        rank = self.grip_info[self._grip_type]["valid_widths"].index(width_key)

        # Normalize from -1 to 1
        return (rank / self.grip_info[self._grip_type]["idx_scale"]) - 1

    def format_action(self, action):
        assert len(action) == self.dof, "Action dimension {} does not match the gripper dof {}".format(
            len(action), self.dof
        )
        assert -1 <= action[0] <= 1, "Action value {} is not in [-1, 1]".format(action)

        action_to_idx = int((action[0] + 1) * self.grip_info[self._grip_type]["idx_scale"])
        width_key = self.grip_info[self._grip_type]["valid_widths"][action_to_idx]

        return self._width_angle_dict[self._grip_type][width_key]["12d"]

    @property
    def dof(self):
        return 1


class CustomPanda(Panda):
    def __init__(self, idn=0):
        super().__init__(idn)

        # Adjust eye_in_hand camera position under the palm
        hand_camera = find_elements(self.worldbody, "camera", attribs={"name": "robot0_eye_in_hand"})
        hand_camera.attrib["pos"] = "0 -0.1 0"
        # hand_camera.attrib["quat"] = "1 0 0 0"

    @property
    def default_gripper(self):
        return {"right": "CustomInspireRightHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    """
    NOTE: gripper_mount_quat_offset is overwritten. The format is [w, x, y, z]
    self.gripper[arm].worldbody.find("body").attrib["quat"] = array_to_string(
                    custom_gripper_mount_quat_offset
                )    
    """

    # A github issue shows that changing the robot/hand xml file can get rid of this offset
    # https://github.com/ARISE-Initiative/robosuite/pull/625
    @property
    def gripper_mount_quat_offset(self):
        return {"right": [-0.5, 0.5, 0.5, -0.5]}  # w, x, y, z


register_robot(CustomPanda)
ROBOT_CLASS_MAPPING["CustomPanda"] = FixedBaseRobot
register_gripper(CustomInspireRightHand)


def show_grab_site(env, grab_pos, grab_ori_mat, approach_len=0.3, offset=None):
    # mark the grab site
    viewer = env.viewer.viewer
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,  # mjGEOM_ARROW,
        size=[0.01, 0, 0],
        pos=grab_pos,
        mat=np.eye(3).flatten(),
        rgba=np.array([0, 1, 0, 1]),
    )

    approach_vec = grab_ori_mat[:, 0]
    gripper_vec = grab_ori_mat[:, 1]

    # Approach vector
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.01, 0, 0],
        pos=grab_pos,
        mat=np.eye(3).flatten(),
        rgba=np.array([1, 0, 0, 1]),
    )
    mujoco.mjv_connector(
        viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_LINE,
        width=0.01,
        from_=grab_pos,
        to=grab_pos + approach_len * approach_vec,
    )

    # Gripper closing
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[2],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.01, 0, 0],
        pos=grab_pos,
        mat=np.eye(3).flatten(),
        rgba=np.array([0, 1, 0, 1]),
    )
    mujoco.mjv_connector(
        viewer.user_scn.geoms[2],
        type=mujoco.mjtGeom.mjGEOM_LINE,
        width=0.01,
        from_=grab_pos,
        to=grab_pos + 0.05 * gripper_vec,
    )

    viewer.user_scn.ngeom = 3
    env.viewer.update()


def make_robosuite_env(
    task="Lift", robot="CustomPanda", camera_name="frontview", camera_height=720, camera_width=1280, render=True
):
    assert camera_name in ["frontview", "birdview", "agentview", "robot0_robotview", "robot0_eye_in_hand"], (
        "camera_name must be one of ['frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand']"
    )

    controller_config = robosuite.load_part_controller_config(default_controller="OSC_POSE")
    controller_config["input_type"] = "absolute"
    controller_config["input_ref_frame"] = "world"
    controller_config["damping_ratio"] = 3  # make robot slower
    controller_config = refactor_composite_controller_config(controller_config, robot, ["right"])
    # Match to robosuite/controllers/config/robots/default_panda_dex.json
    controller_config["body_parts"]["right"]["gripper"]["use_action_scaling"] = False

    # TODO: Create an env with cluttered objects, to "clean" the table
    # Also consider using mink or curobo along with the default OSC delta controller
    env = robosuite.make(
        task,
        robots=[robot],
        controller_configs=controller_config,
        has_renderer=render,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_names=[camera_name],
        camera_depths=[True],
        camera_heights=[int(camera_height)],
        camera_widths=[int(camera_width)],
        control_freq=20,
    )

    return env


def env_reset_get_camera_obs(robot_env, camera_name, init_eef_pos, camera_height=720, camera_width=1280):
    robot_env.reset()

    # The initial pose
    ref_id = robot_env.sim.model.site_name2id("gripper0_right_grip_site")
    eef_pos = robot_env.sim.data.site_xpos[ref_id]
    eef_ori_mat = robot_env.sim.data.site_xmat[ref_id].reshape((3, 3))
    eef_ori_aa = Rotation.from_matrix(eef_ori_mat).as_rotvec()

    search_pose = np.zeros(7)  # OSC_POSE
    search_pose[:3] = eef_pos + np.array(init_eef_pos)
    search_pose[3:6] = eef_ori_aa
    search_pose[6] = 1

    # Get the hand out of the camera view
    for _ in range(50):
        obs_dict, _, _, _ = robot_env.step(search_pose)

    camera = RobosuiteCameraInfo(
        robot_env.sim, camera_name, camera_height=camera_height, camera_width=camera_width
    )

    return camera, obs_dict


def move_eef_to(env, pose, show_grab=True, min_steps=None, max_steps=100):
    gripper = env.robots[0].gripper["right"]
    cnt_steps = 0

    while True:
        env.step(pose)

        # Visualize the grab site
        if env.has_renderer and show_grab:
            grab_pos, grab_ori_mat = gripper.get_grab_site_from_curr_eef(env)
            show_grab_site(env, grab_pos, grab_ori_mat)

        # Check if the eef is near the pose
        ref_id = env.sim.model.site_name2id("gripper0_right_grip_site")
        curr_pose = np.zeros(6)
        curr_pose[:3] = env.sim.data.site_xpos[ref_id]
        curr_pose[3:6] = Rotation.from_matrix(env.sim.data.site_xmat[ref_id].reshape((3, 3))).as_rotvec()

        if np.linalg.norm(curr_pose[:6] - pose[:6]) < 0.01:
            if min_steps is None or cnt_steps >= min_steps:
                break

        if env.has_renderer:
            time.sleep(0.03)

        cnt_steps += 1

        if cnt_steps > max_steps:
            break


def execute_grasp(env, camera, inspire_grasp, twofinger_grasp, grab_site_offset=None, approach_dist=0.15):
    gripper = env.robots[0].gripper["right"]
    grip_type = GRASP_TYPES[str(int(inspire_grasp.grasp_type))]["name"]
    gripper.set_grip_type(grip_type)

    if grab_site_offset is not None:
        gripper.set_grab_site_offset(grab_site_offset)

    twofinger_mat = T.make_pose(twofinger_grasp.translation, twofinger_grasp.rotation_matrix)
    tf_world = camera.camera_to_world_mat @ twofinger_mat
    apprach_vec = tf_world[0:3, 0]

    grab_pos = tf_world[0:3, 3].copy()
    grab_pos += inspire_grasp.depth * apprach_vec  # apply depth in the approach direction
    grab_ori_mat = tf_world[0:3, 0:3]
    target_eef_pos, target_eef_aa = gripper.get_eef_pose_for_grab(grab_pos, grab_ori_mat)

    gripper_open_action = gripper.get_gripper_width_to_action(inspire_grasp.width)
    ready_pose = np.array([0, 0, 0, 0, 0, 0, gripper_open_action], dtype=np.float32)
    approach_pose = np.array([0, 0, 0, 0, 0, 0, gripper_open_action], dtype=np.float32)
    grab_pose = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)  # close gripper
    lift_pose = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)  # close gripper

    ready_pose[:3] = target_eef_pos - approach_dist * apprach_vec
    ready_pose[3:6] = target_eef_aa

    approach_pose[:3] = target_eef_pos
    approach_pose[3:6] = target_eef_aa

    grab_pose[:6] = approach_pose[:6]
    lift_pose[:6] = ready_pose[:6]

    move_eef_to(env, ready_pose)
    move_eef_to(env, approach_pose)
    move_eef_to(env, grab_pose, min_steps=30)
    move_eef_to(env, lift_pose)

    # TODO: Add placing actions?


class RobosuiteCameraInfo:
    def __init__(self, sim, camera_name, camera_height, camera_width, scale=1.0):
        self.sim = sim
        self.camera_name = camera_name
        self.height = float(camera_height)
        self.width = float(camera_width)

        # Intrinsics don't change
        intrinsic = CU.get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width)
        self.fx = intrinsic[0][0]
        self.fy = intrinsic[1][1]
        self.cx = intrinsic[0][2]
        self.cy = intrinsic[1][2]
        self.scale = scale

        self.update_mappings()

    def update_mappings(self):
        # These mappings change according to the camera location
        # camera frame (for depth map) <-> world frame <-> camera pixel
        self.world_to_pixel_mat = CU.get_camera_transform_matrix(self.sim, self.camera_name, self.height, self.width)
        self.camera_to_world_mat = CU.get_camera_extrinsic_matrix(self.sim, self.camera_name)
        self.camera_to_pixel_mat = self.world_to_pixel_mat @ self.camera_to_world_mat

    def get_pixel_coords(self, world_coords):
        return CU.project_points_from_world_to_camera(
            points=world_coords,
            world_to_camera_transform=self.world_to_pixel_mat,
            camera_height=self.height,
            camera_width=self.width,
        ).astype(np.int64)


###################################################################################
if __name__ == "__main__":
    # NOTE: manually correcting offset. TODO: try to get rid of this?
    # There is also angle offset. May be due to NOT using the correct driver-angle mapping...?
    GRAB_SITE_OFFSET = np.array([-0.01, 0.02, 0])  # in the grip frame

    TEST_EEF_MOVE = True
    TARGET_POS = np.array([0, 0, 0.95])
    TARGET_ORI_MAT = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # approach: -z, grab: +y

    # NOTE: This script will open and close the gripper. -1 is closed, 1 is open
    gripper_seq = [-1.0] * 10 + list(np.arange(-1, 1, 0.042)) + [1.0] * 2 + list(np.arange(1, -1, -0.042))

    ### Setup the robot and env
    env = make_robosuite_env()
    env.reset()
    env.step(np.zeros(7))

    gripper = env.robots[0].gripper["right"]
    gripper.set_grab_site_offset(GRAB_SITE_OFFSET)

    ### Default eef pose, which is the initial pose
    # See env.robots[0].part_controllers for arm and hand control
    ref_id = env.sim.model.site_name2id("gripper0_right_grip_site")
    eef_pos = env.sim.data.site_xpos[ref_id]

    # T.mat2quat() -> T.quat2axisangle() is the same as Rotation.from_matrix().as_rotvec()
    eef_quat = T.mat2quat(env.sim.data.site_xmat[ref_id].reshape((3, 3)))
    eef_ori_aa = T.quat2axisangle(eef_quat)

    # Keep the gripper pos and ori constant
    eef_pose = np.zeros(7)  # OSC_POSE
    eef_pose[:3] = eef_pos
    eef_pose[3:6] = eef_ori_aa

    while True:
        for grip_type in list(gripper.grip_info.keys()):
            gripper.set_grip_type(grip_type)
            print("Playing grip:", grip_type)

            if TEST_EEF_MOVE:
                eef_pos, eef_ori_aa = gripper.get_eef_pose_for_grab(TARGET_POS, TARGET_ORI_MAT)
                eef_pose[:3] = eef_pos
                eef_pose[3:6] = eef_ori_aa

            for _ in range(3):
                for i in range(len(gripper_seq)):
                    eef_pose[-1] = gripper_seq[i]
                    env.step(eef_pose)

                    # Visualize the grab site
                    grab_pos, grab_ori_mat = gripper.get_grab_site_from_curr_eef(env)
                    show_grab_site(env, grab_pos, grab_ori_mat, offset=[-0.01, 0.02])

                    time.sleep(0.02)

    print("Done.")
