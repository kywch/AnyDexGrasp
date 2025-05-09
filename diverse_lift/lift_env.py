import os
import xml.etree.ElementTree as ET

import numpy as np

import robosuite
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat, random_quat
from robosuite.utils.mjcf_utils import find_elements
from robosuite.environments.manipulation.lift import Lift

from diverse_lift import AGOD_OBJECT_PATH
from diverse_lift.objects import MJCFObject
from diverse_lift.kitchen_objects import OBJ_GROUPS
from diverse_lift.kitchen_object_utils import sample_kitchen_object


def sample_agod_object(name, idx=None, **obj_args):
    # TODO: max size check
    # TODO: leave some test set out

    if idx is not None:
        obj_path = AGOD_OBJECT_PATH[idx]
    else:
        obj_path = np.random.choice(AGOD_OBJECT_PATH)

    model_xml = os.path.join(obj_path, "model.xml")
    return MJCFObject(name, model_xml, **obj_args)


def sample_objaverse_object(name, group=None, split="A"):
    if group is not None:
        assert group in OBJ_GROUPS, "Invalid group specified"

    mjcf_kwargs, _ = sample_kitchen_object(
        groups=group or "all",
        graspable=True,
        obj_registries=["objaverse"],
        split=split,  # A: first half, B: second half, None: all
    )
    return MJCFObject(name, **mjcf_kwargs)


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
        # NOTE: agod objects have narrow contact. Try bit higher torsion/rolling friction.
        table_friction=(2.0, 0.5, 0.1),
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

        print()

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
                tree.insert(0, option_tag) # Or choose a more appropriate position

        ### Set solver attributes
        # See https://mujoco.readthedocs.io/en/stable/modeling.html#calgorithms
        """When contact slip is a problem, the best way to suppress it is to use
        elliptic cones, large impratio, and the Newton algorithm with very small tolerance.
        If that is not sufficient, enable the Noslip solver.
        """
        option_tag.set("solver", "Newton")  # or CG, PGS. Newton is the default
        option_tag.set("tolerance", str(1e-10))  # default: 1e-8
        option_tag.set("cone", "elliptic")
        # option_tag.set("impratio", str(50))  # robosuite default: 20
        option_tag.set("noslip_iterations", str(3))
        # option_tag.set("noslip_tolerance", str(1e-8))

        return ET.tostring(tree, encoding="utf8").decode("utf8")

    def config_next_sample(self, source=None, group_or_index=None, prob_random_quat=None):
        assert source is None or source in ["objaverse", "agod"], "Invalid source specified"
        self._next_source = source

        assert (
            group_or_index is None or group_or_index in OBJ_GROUPS or group_or_index in range(len(AGOD_OBJECT_PATH))
        ), "Invalid group specified"
        self._next_group_or_index = group_or_index

        assert prob_random_quat is None or 0 <= prob_random_quat <= 1, "Invalid rand_quat_prob specified"
        self._next_prob_random_quat = prob_random_quat

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        # Add solver config to prevent AGOD objects drifting
        self._xml_processors.insert(0, self._set_solver_xml)

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

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        ### NOTE: using self.cube as the object placeholder, so that Lift code works. This is a hack.
        name_hack = "cube"  # This is a hack

        if self._next_source is None or self._next_source == "agod":
            scale = 0.7 + 0.3 * np.random.rand()
            self.cube = sample_agod_object(name_hack, idx=self._next_group_or_index, scale=scale, rgba=[0.5, 0, 0, 1])
        elif self._next_source == "objaverse":
            self.cube = sample_objaverse_object(name_hack, group=self._next_group_or_index)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = RandomOrientationSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube or [],
                x_range=[-0.1, 0.1],
                y_range=[-0.1, 0.1],
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


if __name__ == "__main__":
    from robosuite.environments.base import register_env

    register_env(DiverseLift)

    env = robosuite.make(
        "DiverseLift",
        robots=["Panda"],
        has_renderer=True,
    )

    # env.config_next_sample(source="agod", group_or_index=2, prob_random_quat=1)
    env.config_next_sample(source="objaverse", group_or_index="beer")
    env.reset()
    action = np.zeros(7)
    for _ in range(10):
        env.step(action)

    print()
