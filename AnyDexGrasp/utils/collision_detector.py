__version__ = "1.0"

import os
import copy
import math

import numpy as np
import open3d as o3d


# class CollisionType:
#     NONE = 0b00000000
#     SELF = 0b00000001
#     OTHERS = 0b00000010
#     TABLE = 0b00000100
#     BOX = 0b00001000
#     ANY = 0b11111111


def load_meshes_pointcloud(
    path,
    file_prefix="meshes/source_pointclouds/voxel_size_",
    voxel_grid=0.002,
):
    meshes_pcls = dict()
    meshes_pcl_path = os.path.join(path, file_prefix + str(int(voxel_grid * 1000)))
    for type in os.listdir(meshes_pcl_path):
        type_path = os.path.join(meshes_pcl_path, type)
        for name in os.listdir(type_path):
            width = name[:-4]
            name_path = os.path.join(type_path, name)
            meshes_pcl = o3d.io.read_point_cloud(name_path)
            meshes_pcls[type + "_" + width] = meshes_pcl
    return meshes_pcls


class ModelFreeCollisionDetectorMultifinger:
    def __init__(self, scene_points):
        """Init function. Current finger width and length are fixed.
        Input:
            scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
            voxel_size: [float]
                    used for downsample
        """
        self.finger_length = 0.06

        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        self.scene_cloud = scene_cloud
        self.scene_points = np.array(scene_cloud.points, dtype=np.float32)

    # NOTE: What does this do?
    def _adjust_gripper_centers(self, grasp_group, targets, heights, depths, widths):
        targets = np.array(targets, dtype=np.float32)
        ## get point masks
        # height mask
        mask1 = (targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2)
        # left finger mask
        mask2 = (targets[:, :, 0] > depths - self.finger_length) & (targets[:, :, 0] < depths)
        mask4 = targets[:, :, 1] < -widths / 2
        # right finger mask
        mask6 = targets[:, :, 1] > widths / 2
        # get inner mask of each point
        inner_mask = mask1 & mask2 & (~mask4) & (~mask6)

        ## adjust targets and gripper centers
        # get point bounds
        targets_y = targets[:, :, 1].copy()
        targets_y[~inner_mask] = 0
        ymin = targets_y.min(axis=1)
        ymax = targets_y.max(axis=1)
        # get offsets
        offsets = np.zeros([targets.shape[0], 3], dtype=targets.dtype)
        offsets[:, 1] = (ymin + ymax) / 2
        # adjust targets
        targets[:, :, 1] -= offsets[:, np.newaxis, 1]
        # adjust gripper centers
        R = grasp_group.rotation_matrices
        grasp_group.widths = np.maximum(0.025 * np.ones(ymax.shape), 1.7 * (ymax - ymin))  # Why 1.7?
        grasp_group.translations += np.matmul(R, offsets[:, :, np.newaxis]).squeeze(2)
        return grasp_group, targets

    def normalize(self, x):
        return np.array([x[0], x[1], x[2]]) / math.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))

    def load_meshes_pcls(self, meshes_pcls, two_fingers_ggarray, multifinger_ggarray):
        multifinger_pcls = []
        for id, multifinger_grasp in enumerate(multifinger_ggarray):
            grasp_type = multifinger_grasp.get_grasp_type_with_finger_name()
            width = multifinger_grasp.width
            width = str(round(width * 100, 1))
            key = grasp_type + "_" + width

            translation = multifinger_grasp.translation.reshape(3, 1)
            rotation = multifinger_grasp.rotation_matrix.reshape(3, 3)
            depth = multifinger_grasp.depth

            t = two_fingers_ggarray.translations[id].reshape(3, 1)
            r = two_fingers_ggarray.rotation_matrices[id].reshape(3, 3)

            transform_mat_two_fingers = np.vstack((np.hstack((r, t)), np.array([0, 0, 0, 1])))

            grasp_direction = np.dot(transform_mat_two_fingers, np.array([[0.01], [0], [0], [1]]))[:3]
            grasp_direction = np.array(grasp_direction).reshape(3)
            grasp_direction = grasp_direction - two_fingers_ggarray.translations[id].reshape((3))
            grasp_direction = self.normalize(grasp_direction)
            grasp_depth = np.array(
                [
                    [1, 0, 0, grasp_direction[0] * depth],
                    [0, 1, 0, grasp_direction[1] * depth],
                    [0, 0, 1, grasp_direction[2] * depth],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )

            transform_mat = np.vstack((np.hstack((rotation, translation)), np.array([0, 0, 0, 1])))
            source_mesh_pointclouds = copy.deepcopy(meshes_pcls[key])
            source_mesh_pointclouds.transform(transform_mat)
            source_mesh_pointclouds.transform(grasp_depth)
            multifinger_pcls.append(source_mesh_pointclouds)
        return multifinger_pcls

    def detect(
        self,
        multifinger_ggarray,
        two_fingers_ggarray,
        path_mesh_json,
        meshes_pcls,
        min_grasp_width=0.05,
        downsample_voxel_size=0.03,
        approach_dist=0.04,
        collision_thresh=10,
        adjust_gripper_centers=False,
        DEBUG=False,
    ):
        """Detect collision of grasps.
        Input:
            multifinger_ggarray(class multifingerGraspGroup()): [multifinger_ggarray, M grasps]
                    the grasps to check
            two_fingers_ggarray(class graspnetAPI.GraspGroup): [GraspGroup, M grasps]
            approach_dist: [float]
                    the distance for a gripper to move along approaching direction before grasping
                    this shifting space requires no point either
            collision_thresh: [float]
                    if global collision iou is greater than this threshold,
                    a collision is detected
            adjust_gripper_centers: [bool]
                    if True, add an offset to grasp which makes grasp point closer to object center
        Output:
            collision_free_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies collision-free (and valid) grasp
        """
        # Store original arrays to return later
        mf_gg_original = multifinger_ggarray
        tf_gg_original = two_fingers_ggarray

        ## adjust gripper centers
        if adjust_gripper_centers:
            T = two_fingers_ggarray.translations
            R = two_fingers_ggarray.rotation_matrices
            heights = two_fingers_ggarray.heights[:, np.newaxis]
            depths = two_fingers_ggarray.depths[:, np.newaxis]
            widths = two_fingers_ggarray.widths[:, np.newaxis]
            targets = self.scene_points[np.newaxis, :, :] - T[:, np.newaxis, :]
            targets = np.matmul(targets, R)

            two_fingers_ggarray, targets = self._adjust_gripper_centers(
                two_fingers_ggarray, targets, heights, depths, widths
            )
        
        else:
            # self.__adjust_gripper_centers() multiplies 1.7
            two_fingers_ggarray.widths = two_fingers_ggarray.widths * 1.7  # Why 1.7?

        # 1. Calculate width mask based on original array
        min_width_mask = np.array(tf_gg_original.widths > min_grasp_width)
        valid_width_indices = np.argwhere(min_width_mask).flatten()

        # 2. Filter arrays based on width
        mf_gg_filtered = mf_gg_original[min_width_mask]
        tf_gg_filtered = tf_gg_original[min_width_mask]

        if len(mf_gg_filtered) == 0:
            print("min_grasp_width filter resulted in 0 grasps")
            valid_grasp_mask = np.zeros(len(mf_gg_original), dtype=bool) # All False
            return mf_gg_original, tf_gg_original, valid_grasp_mask

        # 3. Perform expensive operations only on filtered arrays
        mf_gg_filtered.graspgroupTR_2_TR(tf_gg_filtered, path_mesh_json)
        meshes_pointclouds_filtered = self.load_meshes_pcls(meshes_pcls, tf_gg_filtered, mf_gg_filtered)

        # multifinger_ggarray.graspgroupTR_2_TR(two_fingers_ggarray, path_mesh_json)
        # meshes_pointclouds_multifinger = self.load_meshes_pcls(meshes_pcls, two_fingers_ggarray, multifinger_ggarray)
        
        voxel_collision_flags = []  # NOTE: this is internal only, and not being returned
        is_collision_free = []

        ps = self.scene_cloud.points
        ps = o3d.utility.Vector3dVector(ps)
        # st = time.time()
        for idx_filtered, _ in enumerate(mf_gg_filtered):
            meshes_pointclouds = meshes_pointclouds_filtered[idx_filtered]
            grasp_direction = tf_gg_filtered.rotation_matrices[idx_filtered].reshape(3, 3)[:3, 0]
            for i in range(2, int(approach_dist * 100) + 1, 3):
                meshes_pointclouds_cache = copy.deepcopy(meshes_pointclouds)
                back_distance = [
                    [1, 0, 0, -grasp_direction[0] * i * 0.01],
                    [0, 1, 0, -grasp_direction[1] * i * 0.01],
                    [0, 0, 1, -grasp_direction[2] * i * 0.01],
                    [0, 0, 0, 1],
                ]
                meshes_pointclouds_cache.transform(np.array(back_distance, dtype=np.float32)).paint_uniform_color(
                    [0, 1, 0]
                )
                meshes_pointclouds = meshes_pointclouds + meshes_pointclouds_cache
            # st3 = time.time()
            meshes_pointclouds = meshes_pointclouds.voxel_down_sample(downsample_voxel_size)

            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=meshes_pointclouds, voxel_size=downsample_voxel_size)
            output = voxel_grid.check_if_included(ps)
            voxel_collision_flags.append(output)

            if DEBUG and idx_filtered < 5:
                print("transformed mesh")
                FOR_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                o3d.visualization.draw_plotly(
                    [
                        self.scene_cloud,
                        meshes_pointclouds,
                        FOR_base,
                        tf_gg_filtered[idx_filtered].to_open3d_geometry(),
                    ],
                    width=1024,
                    height=640,
                )

            if np.array(output).astype(int).sum() > collision_thresh:
                is_collision_free.append(False)

                if DEBUG and idx_filtered < 5:
                    print("collision")
                    collision_point_cloud = o3d.geometry.PointCloud()
                    collision_point_cloud.points = o3d.utility.Vector3dVector(
                        np.array(self.scene_cloud.points)[np.array(output)]
                    )
                    collision_point_cloud.paint_uniform_color([1, 0, 0])
                    normal_point_cloud = o3d.geometry.PointCloud()
                    normal_point_cloud.points = o3d.utility.Vector3dVector(
                        np.array(self.scene_cloud.points)[~(np.array(output))]
                    )
                    normal_point_cloud.paint_uniform_color([0, 0, 1])
                    o3d.visualization.draw_plotly(
                        [
                            normal_point_cloud,
                            collision_point_cloud,
                            FOR_base,
                            tf_gg_filtered[idx_filtered].to_open3d_geometry(),
                        ],
                        width=1024,
                        height=640,
                    )
            else:
                is_collision_free.append(True)

        # valid grasp mask combines both collision free and min width filter
        valid_grasp_mask = np.zeros(len(mf_gg_original), dtype=bool) # Initialize with False
        collision_results = np.array(is_collision_free)
        valid_grasp_mask[valid_width_indices] = collision_results 

        return valid_grasp_mask
