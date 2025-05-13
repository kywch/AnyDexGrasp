import os
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy import ndimage

import robosuite
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat, random_quat
from robosuite.utils.mjcf_utils import find_elements
import robosuite.utils.camera_utils as CU

from robosuite.environments.manipulation.lift import Lift

# NOTE: this is for debugging
# from diverse_lift.rs_debug.lift import Lift

from diverse_lift import AGOD_OBJECT_PATH
from diverse_lift.objects import MJCFObject
from diverse_lift.kitchen_objects import OBJ_GROUPS
from diverse_lift.kitchen_object_utils import sample_kitchen_object


def sample_agod_object(name, idx=None, max_size=0.14, **obj_args):
    # TODO: leave some test set out

    if idx is not None:
        obj_path = AGOD_OBJECT_PATH[idx]
    else:
        obj_path = np.random.choice(AGOD_OBJECT_PATH)

    target = "agod_" + os.path.basename(obj_path)

    # Get the object size, and adjust the scale to not exceed max size
    scale = obj_args["scale"] or None
    if scale is not None:
        with open(os.path.join(obj_path, "object_size.json"), "r") as f:
            obj_size = json.load(f)

        max_scale = [max_size / v for v in obj_size.values()]
        obj_args["scale"] = [min(s, scale) for s in max_scale]

    model_xml = os.path.join(obj_path, "model.xml")
    print(f"Loading the object from {model_xml}")

    ### NOTE: if collision-related adjustment is needed, set values here
    # obj_args["solref"] = (0.001, 1.5)
    # obj_args["solimp"] = (0.90, 0.995, 0.01)
    obj_args["margin"] = 0.001

    return MJCFObject(name, model_xml, **obj_args), target


def sample_objaverse_object(name, group=None, split="A"):
    if group is not None:
        assert group in OBJ_GROUPS or (group.endswith(".xml") and os.path.exists(group)), "Invalid group specified"

    mjcf_kwargs, _ = sample_kitchen_object(
        groups=group or "all",
        graspable=True,
        obj_registries=["objaverse"],
        split=split,  # A: first half, B: second half, None: all
    )

    target = "objaverse_" + os.path.basename(os.path.dirname(mjcf_kwargs["mjcf_path"]))
    print(f"Loading {target}")

    return MJCFObject(name, **mjcf_kwargs), target


class RandomOrientationSampler(UniformRandomSampler):
    prob_random_quat = 0.0

    # NOTE: this ignores rotation, rotation_axis args and just apply random quat
    def _sample_quat(self):
        if np.random.rand() > self.prob_random_quat:
            # This is z-rotation only
            return super()._sample_quat()

        quat_xyzw = random_quat()
        return convert_quat(quat_xyzw, to="wxyz")


class DiverseLift(Lift):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 0.1, 0.1),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.target_object = None
        self._next_source = None
        self._next_group_or_index = None
        self._next_prob_random_quat = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def _set_solver_xml(self, xml_str):
        tree = ET.fromstring(xml_str)

        # Find the <option> tag. It's usually a direct child of the root <mujoco> tag.
        option_tag = find_elements(root=tree, tags="option", return_first=True)

        if option_tag is None:
            # If <option> tag doesn't exist, create it
            option_tag = ET.Element("option")
            # Insert it after <compiler> or <custom> if they exist, or at the beginning of <mujoco>
            compiler_tag = find_elements(root=tree, tags="compiler", return_first=True)
            custom_tag = find_elements(root=tree, tags="custom", return_first=True)
            if custom_tag is not None:
                custom_tag_idx = list(tree).index(custom_tag)
                tree.insert(custom_tag_idx + 1, option_tag)
            elif compiler_tag is not None:
                compiler_tag_idx = list(tree).index(compiler_tag)
                tree.insert(compiler_tag_idx + 1, option_tag)
            else:
                tree.insert(0, option_tag)  # Or choose a more appropriate position

        ### Set solver attributes
        # See https://mujoco.readthedocs.io/en/stable/modeling.html#calgorithms
        """When contact slip is a problem, the best way to suppress it is to use
        elliptic cones, large impratio, and the Newton algorithm with very small tolerance.
        If that is not sufficient, enable the Noslip solver.
        """
        # option_tag.set("timestep", str(0.0005))
        # option_tag.set("solver", "Newton")  # or CG, PGS. Newton is the default
        # option_tag.set("tolerance", str(1e-10))  # default: 1e-8
        # option_tag.set("cone", "elliptic")
        # option_tag.set("impratio", str(50))  # robosuite default: 20
        # option_tag.set("noslip_iterations", str(3))
        # option_tag.set("noslip_tolerance", str(1e-8))
        # option_tag.set("integrator", "implicit")

        return ET.tostring(tree, encoding="utf8").decode("utf8")

    def config_next_sample(self, source=None, group_or_index=None, prob_random_quat=None):
        assert source is None or source in ["objaverse", "agod"], "Invalid source specified"
        self._next_source = source

        assert (
            group_or_index is None
            or group_or_index in OBJ_GROUPS
            or group_or_index in range(len(AGOD_OBJECT_PATH))
            or (group_or_index.endswith(".xml") and os.path.exists(group_or_index))
        ), "Invalid group specified"
        self._next_group_or_index = group_or_index

        assert prob_random_quat is None or 0 <= prob_random_quat <= 1, "Invalid rand_quat_prob specified"
        self._next_prob_random_quat = prob_random_quat

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        # NOTE: If simulation is unstable, change the solver setting with below
        # self._xml_processors.insert(0, self._set_solver_xml)

        # Load robots
        self._load_robots()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # NOTE: table collision can be changed like below
        # mujoco_arena.table_collision.attrib["margin"] = "0.001"

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        ### NOTE: using self.cube as the object placeholder, so that Lift code works. This is a hack.
        name_hack = "cube"  # This is a hack

        if self._next_source is None or self._next_source == "agod":
            scale = 0.5 + np.random.rand()
            self.cube, self.target_object = sample_agod_object(
                name_hack, idx=self._next_group_or_index, scale=scale, rgba=[1.0, 0.5, 0, 1]
            )
        elif self._next_source == "objaverse":
            self.cube, self.target_object = sample_objaverse_object(name_hack, group=self._next_group_or_index)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = RandomOrientationSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube or [],
                x_range=[-0.1, 0.1],
                y_range=[-0.05, 0.1],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.001,
            )

        # Set which quat sampler to use
        self.placement_initializer.prob_random_quat = self._next_prob_random_quat or 0.0

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _check_success(self):
        """
        Check if object has been lifted.

        Returns:
            bool: True if object has been lifted
        """
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        # cube is higher than the table top above a margin
        return cube_height > table_height + 0.11

    def get_object_mask(self):
        # NOTE: use only one camera for now.
        cam_name, height, width = self.camera_names[0], self.camera_heights[0], self.camera_widths[0]
        seg_map = CU.get_camera_segmentation(self.sim, cam_name, height, width)[:, :, [1]]  # ones containing geom ids

        # target object geom
        geom_id = self.sim.model.geom_name2id(self.cube.visual_geoms[0])

        object_mask = ndimage.median_filter(seg_map == geom_id, 5)

        return object_mask


if __name__ == "__main__":
    from robosuite.environments.base import register_env
    from diverse_lift import OBJAVERSE_PATH

    register_env(DiverseLift)

    env = robosuite.make(
        "DiverseLift",
        robots=["Panda"],
        has_renderer=True,
        ignore_done=True,
    )

    # env.config_next_sample(source="agod", prob_random_quat=1, group_or_index=1)
    env.config_next_sample(source="objaverse", group_or_index="beer")
    # env.config_next_sample(source="objaverse", group_or_index=os.path.join(OBJAVERSE_PATH, "cake/cake_2/model.xml"))
    env.reset()
    action = np.zeros(7)
    while True:
        env.step(action)
