import copy
import os
import json
import xlrd2
import numpy as np
import open3d as o3
from transforms3d.euler import euler2mat, quat2mat
import pybullet as p
import pybullet_data
import math
from tqdm import tqdm
import sys
from angles2stl import InspireAngles2STLs


# Below copied from: ur_toolbox.robot.InspireHandR_grasp import grasp_types
# What are facenet_thumb and facenet_index?
grasp_types = {
    "1": {
        "name": "Ring",
        "facenet_thumb": [[207598, 207599]],
        "facenet_index": [[146358, 146357], [53344, 53345]],
        "width": [0, 0.11],
    },
    "2": {
        "name": "Prismatic_2_Finger",
        "facenet_thumb": [[207598, 207599]],
        "facenet_index": [[146358, 146357], [53344, 53345]],
        "width": [0, 0.11],
    },
    "3": {
        "name": "Prismatic_3_Finger",
        "facenet_thumb": [[207598, 207599]],
        "facenet_index": [[146358, 146357], [53344, 53345]],
        "width": [0, 0.11],
    },
    "4": {
        "name": "Large_Diameter",
        "facenet_thumb": [[207598, 207599]],
        "facenet_index": [[146358, 146357], [53344, 53345]],
        "width": [0, 0.11],
    },
    "5": {
        "name": "Medium_Wrap",
        "facenet_thumb": [[207416]],
        "facenet_index": [[146358, 146357], [53344, 53345]],
        "width": [0, 0.11],
    },
    "6": {
        "name": "Tripod",
        "facenet_thumb": [[207416]],
        "facenet_index": [[146358, 146357], [52220]],
        "width": [0.025, 0.10],
    },
    "7": {
        "name": "Sphere_3_Finger",
        "facenet_thumb": [[207416]],
        "facenet_index": [[146358, 146357], [81881]],
        "width": [0.025, 0.10],
    },
    "8": {
        "name": "Distal_Type",
        "facenet_thumb": [[207606]],
        "facenet_index": [[53344, 53345], [82744]],
        "width": [0.025, 0.10],
    },
}


joint_mesh_mapping = {
    "base": "Link111.STL",
    "0": "Link1.STL",
    "1": "Link11.STL",
    "2": "Link2.STL",
    "3": "Link22.STL",
    "4": "Link3.STL",
    "5": "Link33.STL",
    "6": "Link4.STL",
    "7": "Link44.STL",
    "8": "Link5.STL",
    "9": "Link51.STL",
    "10": "Link52.STL",
    "11": "Link53.STL",
}


def four_meta_to_matrix(x):
    xyz = [x[0], x[1], x[2]]
    rpy = [x[6], x[3], x[4], x[5]]

    rot_mat = quat2mat(rpy)
    return np.array(
        [
            [rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], xyz[0]],
            [rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], xyz[1]],
            [rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], xyz[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def rate(rate1, rate2, num1):
    return (num1 - rate1[0]) / (rate1[1] - rate1[0]) * (rate2[1] - rate2[0]) + rate2[0]


def anti_rate(rate1, rate2, num1):
    return rate2[1] - (num1 - rate1[0]) / (rate1[1] - rate1[0]) * (rate2[1] - rate2[0])


def get_angle(ids, pose_6d, pose_12d):
    if pose_6d[ids] == 1000:
        return 0, 0, 1000
    return pose_12d[0], pose_12d[1], pose_6d[ids]


def read_excel_6Dangle_to_12Dangle(path_6d, path_12d):
    wb_12d = xlrd2.open_workbook(filename=path_12d)
    sheet_12d = wb_12d.sheet_by_index(0)

    pos_12d = sheet_12d.col_values(0)[2:]

    # NOTE: from driver_routine_to_angle.xls (sheet_12d), cols 1-4 are for thumb
    # There are 2000 rows, which are then mapped to 0-1000 below, see vars *_scope
    thumb_bending_1_12d = sheet_12d.col_values(1)[2:]
    thumb_bending_2_12d = sheet_12d.col_values(2)[2:]
    thumb_bending_3_12d = sheet_12d.col_values(3)[2:]
    thumb_rotation_12d = sheet_12d.col_values(4)[2:]

    # NOTE: from driver_routine_to_angle.xls, cols 5-6 are for other 4 fingers
    little_1_12d = sheet_12d.col_values(5)[2:]
    little_2_12d = sheet_12d.col_values(6)[2:]
    ring_1_12d = sheet_12d.col_values(5)[2:]
    ring_2_12d = sheet_12d.col_values(6)[2:]
    mid_1_12d = sheet_12d.col_values(5)[2:]
    mid_2_12d = sheet_12d.col_values(6)[2:]
    index_1_12d = sheet_12d.col_values(5)[2:]
    index_2_12d = sheet_12d.col_values(6)[2:]

    angles = [[] for _ in range(len(grasp_types))]
    pos_scope = [(90, 1631), (89, 1650), (68, 1645), (67, 1635), (0, 1335), (400, 1826)]
    pos_scope_true = [(49, 1856), (50, 1873), (28, 1871), (20, 1857), (53, 1606), (399, 1827)]
    angle_scope = [
        (-1.6, 0),
        (-1.7, 0),
        (-1.6, 0),
        (-1.7, 0),
        (-1.6, 0),
        (-1.7, 0),
        (-1.6, 0),
        (-1.7, 0),
        (-1.0, 0.3),
        (0.0, 0.4),
        (-0.4, 0.0),
        (-1.0, 0.0),
    ]
    angle_scope_ture = [
        (-1.6, 0),
        (-1.7, 0),
        (-1.6, 0),
        (-1.7, 0),
        (-1.6, 0),
        (-1.7, 0),
        (-1.4856, 0.0),
        (-1.57845, 0.0),
        (-1.447, 0.3),
        (-0.3, 0.133),
        (-0.267, 0.179),
        (-0.803, -0.3),
    ]
    angle_scope1 = [
        [(178.6632021, 180.7766672), (93.96967048, 88.69547598)],
        [(139.2725166, 188.0809, 169.9457), (110.5302465, 154.2716, 147.5372)],
        [(106.6702328), (175.0204983)],
    ]
    angle_scope2 = [
        (173.678487604719, 104.222063933174),
        (177.2716606, 105.8915597),
        (173.742853862258, 103.3899537),
        (175.117175212283, 103.6093653),
        (175.050712328838, 104.0472525),
        (81.6751504, 150.2673673),
        (133.3610164, 105.5103515),
        (
            180.6024,
            149.1983,
        ),
        (166.9032, 141.6969),
    ]

    # NOTE: inspire_hand_routine_to_angle.xls (path_6d) file has 0-1000 finger angles for eight grip types
    wb_6d = xlrd2.open_workbook(path_6d)

    for pose_id in range(len(wb_6d.sheets())):
        sheet_6d = wb_6d.sheet_by_index(pose_id)

        # NOTE: width is in mm, but changing it to cm
        width = sheet_6d.col_values(0)[1:]
        width = np.array(width) * 0.1

        # 0-1000 normalized angles
        little_6d = sheet_6d.col_values(1)[1:]
        ring_6d = sheet_6d.col_values(2)[1:]
        mid_6d = sheet_6d.col_values(3)[1:]
        index_6d = sheet_6d.col_values(4)[1:]
        thumb_bending_6d = sheet_6d.col_values(5)[1:]
        thumb_rotation_6d = sheet_6d.col_values(6)[1:]
        for ids, _ in enumerate(index_6d):
            if index_6d[ids] == -1:
                continue

            # 1000: finger angles are close to 180 -- close to fully open
            # 0: finger angles are close to 90, these angle values come from sheet_12d (driver to angle)
            # (0, -1.511) radian values? these seem to related to the numbers in the URDF (or mujoco) file,
            # which limits the joint movement range.
            # So get_angle() function seems to map the 0-1000 range to joint angle. These are source of noise.
            # But, probably very important for sim-to-real transfer.
            index_scope = int(float(index_6d[ids]) * (1857 - 20) / 1000)
            index = [
                anti_rate((178.66, 93.97), (0, -1.511), index_1_12d[index_scope]),
                anti_rate((180.78, 88.7), (0, -1.5416), index_2_12d[index_scope]) + 0.25,
            ]
            mid_scope = int(float(mid_6d[ids]) * (1857 - 20) / 1000)
            mid = [
                anti_rate((178.66, 93.97), (0, -1.511), mid_1_12d[mid_scope]),
                anti_rate((180.78, 88.7), (0, -1.5416), mid_2_12d[mid_scope]) + 0.25,
            ]
            ring_scope = int(float(ring_6d[ids]) * (1857 - 20) / 1000)
            ring = [
                anti_rate((178.66, 93.97), (0, -1.511), ring_1_12d[ring_scope]),
                anti_rate((180.78, 88.7), (0, -1.5416), ring_2_12d[ring_scope]) + 0.25,
            ]
            little_scope = int(float(little_6d[ids]) * (1857 - 20) / 1000)
            little = [
                anti_rate((178.66, 93.97), (0, -1.511), little_1_12d[little_scope]),
                anti_rate((180.78, 88.7), (0, -1.5416), little_2_12d[little_scope]) + 0.25,
            ]

            thumb_bending_scope = int((thumb_bending_6d[ids]) * (1606 - 53) / 1000)
            thumb_bending = [
                (thumb_bending_1_12d[thumb_bending_scope] - 139.27) / (139.27 - 110.53) * 0.433 + 0.13,
                (thumb_bending_2_12d[thumb_bending_scope] - 188.08) / (154.27 - 188.08) * (0.179 + 0.267) - 0.237,
                (thumb_bending_3_12d[thumb_bending_scope] - 169.94) / (147.53 - 169.94) * 0.503 - 0.7,
            ]

            thumb_rotation_scope = int(float(thumb_rotation_6d[ids]) * (2000 - 400) / 1000) + 400
            thumb_rotation = anti_rate(
                (185.5631219, 106.72277927615), (-1.3, 0.3), thumb_rotation_12d[thumb_rotation_scope]
            )
            if pose_id == 0:
                thumb_rotation = anti_rate((81.6751504, 174.02), (-1.2, 0.3), 169.2747667)
            index0, index1, index_angle = get_angle(ids, index_6d, index)
            mid0, mid1, mid_angle = get_angle(ids, mid_6d, mid)
            ring0, ring1, ring_angle = get_angle(ids, ring_6d, ring)
            little0, litte1, little_angle = get_angle(ids, little_6d, little)

            # NOTE: 6d output is still in 0-1000 range, 12d output is assumed to be in radians, which are used in the URDF
            # The URDF model here joint limit seems to be different from the robosuite inspire hand model
            angles[pose_id].append(
                [
                    index0,
                    index1,
                    mid0,
                    mid1,
                    ring0,
                    ring1,
                    little0,
                    litte1,
                    thumb_rotation,
                    thumb_bending[0],
                    thumb_bending[1],
                    thumb_bending[2],
                    width[ids],
                    little_angle,
                    ring_angle,
                    mid_angle,
                    index_angle,
                    thumb_bending_6d[ids],
                    thumb_rotation_6d[ids],
                ]
            )

    return angles


def open_pybullet(urdf_path):
    clid = p.connect(p.SHARED_MEMORY)
    if clid < 0:
        p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(solverResidualThreshold=0, maxNumCmdPer1ms=1000)
    fps = 240
    timeStep = 1.0 / fps
    p.setTimeStep(timeStep)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0.35, -0.13, 0.5]
    )

    flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

    base_orn = [0, 0, 0]
    base_orn = p.getQuaternionFromEuler(base_orn)

    hand = p.loadURDF(urdf_path, [0.0, 0.0, 0.0], base_orn, useFixedBase=True)
    return p, hand


def get_mesh(po, id, PATH):
    t_angle = four_meta_to_matrix(po)
    path_file = PATH + joint_mesh_mapping[str(id)]
    link = o3.io.read_triangle_mesh(path_file)
    link = link.transform(t_angle)
    return link


def save_stl_and_pointcloud(name, angle, mesh, output_path):
    if not os.path.exists(output_path + "source/" + name):
        os.makedirs(output_path + "source/" + name)
    save_stl_path = output_path + "source/" + name + "/" + str(np.round(angle[12], 1)) + ".STL"
    o3.io.write_triangle_mesh(save_stl_path, mesh)

    pcd = o3.geometry.TriangleMesh.sample_points_uniformly(mesh, 300000)
    pcd.estimate_normals(search_param=o3.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    points = np.array(pcd.points)
    for i in range(1, 3):
        points = np.vstack((points, np.array(pcd.points) - np.array(pcd.normals) * i * 1.5 * 0.001))
    for i in range(3, 8):
        points = np.vstack(
            (points, np.array(pcd.points)[: 170000 * 1] - np.array(pcd.normals)[: 170000 * 1] * i * 1.5 * 0.001)
        )
    pcd_extend = o3.geometry.PointCloud()
    pcd_extend.points = o3.utility.Vector3dVector(points)
    for i in range(1, 6):
        pcd_saved = pcd_extend.voxel_down_sample(0.001 * i)
        if not os.path.exists(output_path + "source_pointclouds/voxel_size_" + str(i) + "/" + name):
            os.makedirs(output_path + "source_pointclouds/voxel_size_" + str(i) + "/" + name)
        save_pcd_path = (
            output_path
            + "source_pointclouds/voxel_size_"
            + str(i)
            + "/"
            + name
            + "/"
            + str(np.round(angle[12], 1))
            + ".ply"
        )
        o3.io.write_point_cloud(save_pcd_path, pcd_saved)


def get_meshes(angles, stl_path, output_path, width_12Dangle_6Dangel_json, urdf_path, if_source, vis):
    p, hand = open_pybullet(urdf_path)
    angles_to_stls = InspireAngles2STLs(grasp_types)
    width_12Dangle_6Dangel = dict()
    for grasp_type, angle8 in enumerate(angles):
        for id in tqdm([i for i in range(len(angle8))]):
            angle = angle8[id]
            width = np.round(angle[12], 1)

            path_file = stl_path + joint_mesh_mapping["base"]
            base = o3.io.read_triangle_mesh(path_file)
            pose = []
            for joint_id in range(12):
                p.resetJointState(hand, joint_id, angle[joint_id])
            for joint_id in range(12):
                po = p.getLinkState(hand, joint_id)
                pose.append([po[4][0], po[4][1], po[4][2], po[5][0], po[5][1], po[5][2], po[5][3]])
            for joint_id in range(2, 8):
                mesh = get_mesh(pose[joint_id], joint_id, stl_path)
                base = base + mesh
            for joint_id in range(2):
                mesh = get_mesh(pose[joint_id], joint_id, stl_path)
                base = base + mesh
            for joint_id in range(8, 12):
                mesh = get_mesh(pose[joint_id], joint_id, stl_path)
                base = base + mesh
            # transform to the center of wrist
            base.transform([[1, 0, 0, 0.04123], [0, 1, 0, 0.00804], [0, 0, 1, -0.01796], [0, 0, 0, 1]])
            # add the ring of metal which is used to fix screw
            base.transform([[1, 0, 0, 0.0078], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            base.compute_triangle_normals()
            name = grasp_types[str(grasp_type + 1)]["name"]
            save_stl_and_pointcloud(name, angle, base, output_path)
            if if_source:
                if grasp_type == 4:
                    translation = width_12Dangle_6Dangel["Ring"][str(width)]["translation"]
                    rotation = width_12Dangle_6Dangel["Ring"][str(width)]["rotation"]
                else:
                    translation, rotation = angles_to_stls.get_pose_information(base, grasp_type, vis=vis)
                    translation, rotation = translation.tolist(), rotation.tolist()
                if str(name) not in width_12Dangle_6Dangel.keys():
                    width_12Dangle_6Dangel[name] = dict()
                if width in width_12Dangle_6Dangel[grasp_types[str(grasp_type + 1)]["name"]].keys():
                    print(
                        "*******************************\n\nerror!!!!!!!!",
                        grasp_type,
                        grasp_type[str(grasp_type + 1)],
                        width,
                    )
                width_12Dangle_6Dangel[name][str(width)] = {
                    "12d": angles[grasp_type][id][:12],
                    "6d": angles[grasp_type][id][13:],
                    "translation": translation,
                    "rotation": rotation,
                }

    if if_source:
        json_str = json.dumps(width_12Dangle_6Dangel, indent=4)
        with open(width_12Dangle_6Dangel_json, "w") as json_file:
            json_file.write(json_str)


if __name__ == "__main__":
    path_6d = "./generate_mesh_and_pointcloud/inspire_urdf/inspire_hand_routine_to_angle-use.xlsx"
    path_12d = "./generate_mesh_and_pointcloud/inspire_urdf/driver_routine_to_angle.xls"

    source_stl_path = "./generate_mesh_and_pointcloud/inspire_urdf/urdf-five3/meshes/"
    output_path = "./generate_mesh_and_pointcloud/inspire_urdf/meshes/"
    json_path = "./generate_mesh_and_pointcloud/inspire_urdf/width_12Dangle_6Dangle.json"

    urdf_path = "./generate_mesh_and_pointcloud/inspire_urdf/urdf-five3/robots/urdf-five3.urdf"

    angles = read_excel_6Dangle_to_12Dangle(path_6d, path_12d)
    get_meshes(angles, source_stl_path, output_path, json_path, urdf_path, if_source=True, vis=False)
