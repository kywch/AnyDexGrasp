import os
import re
import json
import argparse

import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import open3d as o3d


SITE_RGBA = "0 0 0 0"  # Transparent
SITE_SIZE = "0.005"


def sanitize_name_for_xml_and_filename(name):
    """Sanitizes a string to be used as an XML name or filename component."""
    name = re.sub(r"[^\w_.-]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    if not name:
        return "unnamed_part"
    return name


def get_translated_obj_lines(obj_filepath, translation_vector):
    """Reads an OBJ file, translates its 'v' vertices, and returns all lines (translated 'v' and original others)."""
    processed_lines = []
    try:
        with open(obj_filepath, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    x = float(parts[1]) - translation_vector[0]
                    y = float(parts[2]) - translation_vector[1]
                    z = float(parts[3]) - translation_vector[2]
                    processed_lines.append(f"v {x:.5f} {y:.5f} {z:.5f}\n")
                else:
                    processed_lines.append(line)
    except Exception as e:
        print(f"    Error processing {obj_filepath} for translation: {e}")
        return []
    return processed_lines


def split_obj_lines_into_parts(all_obj_lines, output_meshes_dir, original_basename_for_default_part_name):
    """
    Splits OBJ data (provided as a list of lines) into multiple OBJ files based on 'o' tags.
    Each new OBJ file will contain all vertex data (v, vn, vt) and mtllib lines
    from the input lines, and the face/material/smoothing group lines (f, usemtl, s)
    associated with its 'o' object group.
    If no 'o' tags are found but faces exist, the entire file is treated as one part.
    Returns a list of basenames of the created .obj files.
    """
    vertex_data_lines = []
    mtllib_lines = []

    for line in all_obj_lines:
        if line.startswith(("v ", "vn ", "vt ")):
            vertex_data_lines.append(line)
        elif line.startswith("mtllib "):
            mtllib_lines.append(line)

    parts_data = []  # List of {'name': str, 'content_lines': [str]}
    current_part_content_lines = []

    current_part_name = sanitize_name_for_xml_and_filename(original_basename_for_default_part_name)
    object_idx_counter = 0

    has_o_tags = any(line.strip().startswith("o ") for line in all_obj_lines)

    if has_o_tags:
        first_o_tag_processed = False
        for line in all_obj_lines:
            stripped_line = line.strip()
            if stripped_line.startswith("o "):
                if first_o_tag_processed and current_part_content_lines:
                    parts_data.append({"name": current_part_name, "content_lines": list(current_part_content_lines)})
                    current_part_content_lines.clear()

                name_candidate = (
                    stripped_line.split(maxsplit=1)[1]
                    if len(stripped_line.split(maxsplit=1)) > 1
                    else f"part_{object_idx_counter}"
                )
                current_part_name = sanitize_name_for_xml_and_filename(name_candidate)
                object_idx_counter += 1
                first_o_tag_processed = True
            elif not stripped_line.startswith(("v ", "vn ", "vt ", "mtllib ")):
                if first_o_tag_processed:
                    current_part_content_lines.append(line)

        if current_part_content_lines:
            parts_data.append({"name": current_part_name, "content_lines": list(current_part_content_lines)})
    else:
        single_part_content = [line for line in all_obj_lines if not line.startswith(("v ", "vn ", "vt ", "mtllib "))]
        if any(line.strip().startswith("f ") for line in single_part_content):  # Only create a part if there are faces
            parts_data.append({"name": current_part_name, "content_lines": single_part_content})

    output_filenames = []
    if not parts_data and vertex_data_lines:  # If no parts but vertices exist, save all as one part
        print(
            f"    Note: No 'o' tags or distinct face groups found. Writing all geometry as '{current_part_name}.obj'."
        )
        output_filepath = os.path.join(output_meshes_dir, f"{current_part_name}.obj")
        with open(output_filepath, "w") as outfile:
            outfile.writelines(mtllib_lines)
            outfile.writelines(vertex_data_lines)  # These are the *centered* vertex data
            if not has_o_tags:  # Add all non-vertex/mtllib lines if no 'o' tags were found
                outfile.writelines(
                    [line for line in all_obj_lines if not line.startswith(("v ", "vn ", "vt ", "mtllib "))]
                )
        return [f"{current_part_name}.obj"]
    elif not parts_data:
        return []  # No parts and no vertices, nothing to write

    final_part_filenames = []
    name_counts = {}
    for i, part in enumerate(parts_data):
        base_name = part["name"]
        if not base_name:
            base_name = f"part_{i}"

        unique_filename_base = base_name
        count = name_counts.get(base_name, 0)
        if count > 0:
            unique_filename_base = f"{base_name}_{count}"
        name_counts[base_name] = count + 1
        final_part_filenames.append(f"{unique_filename_base}.obj")

    for i, part_data in enumerate(parts_data):
        part_filename = final_part_filenames[i]
        output_filepath = os.path.join(output_meshes_dir, part_filename)

        with open(output_filepath, "w") as outfile:
            outfile.write(f"# Original object name hint: {part_data['name']}\n")
            outfile.writelines(mtllib_lines)
            outfile.writelines(vertex_data_lines)
            outfile.writelines(part_data["content_lines"])
        output_filenames.append(part_filename)

    return output_filenames


def load_vertices_from_obj(obj_filepath):
    """Loads vertex data from an OBJ file."""
    vertices = []
    try:
        with open(obj_filepath, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    try:
                        # Convert to float, handling potential scientific notation
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError as e:
                        print(f"    Warning: Could not parse vertex line in {obj_filepath}: {line.strip()} - {e}")
                        continue
    except FileNotFoundError:
        print(f"    Warning: OBJ part file not found: {obj_filepath}")
    except Exception as e:
        print(f"    Error reading OBJ part file {obj_filepath}: {e}")
    return vertices


def load_vertices_from_obj_lines(obj_lines_list):
    """Loads vertex data from a list of OBJ file lines."""
    vertices = []
    for line in obj_lines_list:
        if line.startswith("v "):
            parts = line.split()
            try:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError as e:
                print(f"    Warning: Could not parse vertex from line: {line.strip()} - {e}")
                continue
    return vertices


def prettify_xml(elem_or_str):
    rough_string = ET.tostring(elem_or_str, "utf-8") if isinstance(elem_or_str, ET.Element) else elem_or_str
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def main():
    parser = argparse.ArgumentParser(
        description="Convert AGOD OBJ files to MuJoCo XML models with visual and collision meshes."
    )
    parser.add_argument(
        "--source_obj_dir",
        default="agod_objects/objectMeshes",
        type=str,
        help="Directory containing the source OBJ files (e.g., objectMeshes).",
    )
    parser.add_argument(
        "--output_base_dir",
        default="agod_objects/AGOD",
        type=str,
        help="Base directory where output model folders (e.g., AGOD/XXXXX/) will be created.",
    )
    parser.add_argument(
        "--target_triangles",
        type=int,
        default=24,
        help="Target number of triangles for collision mesh simplification (per part). Default: 24.",
    )
    parser.add_argument(
        "--exclude_obj", nargs="+", default=["20050", "21446"], type=str, help="Objects to exclude from processing"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.source_obj_dir):
        print(f"Error: Source directory not found: {args.source_obj_dir}")
        return

    print(f"Scanning for OBJ files in: {args.source_obj_dir}")
    file_list = os.listdir(args.source_obj_dir)
    file_list.sort()

    processed_count = 0
    for obj_filename in file_list:
        if not (obj_filename.startswith("mesh_") and obj_filename.endswith(".obj")):
            continue

        print(f"\nProcessing '{obj_filename}'...")

        match = re.search(r"mesh_(\d+)\.obj", obj_filename)
        if not match:
            print(f"  Warning: Could not extract ID from '{obj_filename}'. Skipping.")
            continue

        full_numeric_part = match.group(1)
        if len(full_numeric_part) < 5:
            print(
                f"  Warning: Numeric part '{full_numeric_part}' in '{obj_filename}' is too short for 5-digit ID. Skipping."
            )
            continue
        five_digit_id = full_numeric_part[-5:]

        if five_digit_id in args.exclude_obj:
            print(f"  ID {five_digit_id} is in the excluding list. Skipping.")
            continue

        current_obj_output_dir = os.path.join(args.output_base_dir, five_digit_id)
        visual_mesh_dir = os.path.join(current_obj_output_dir, "visual")
        collision_mesh_dir = os.path.join(current_obj_output_dir, "collision")
        os.makedirs(visual_mesh_dir, exist_ok=True)
        os.makedirs(collision_mesh_dir, exist_ok=True)
        print(f"  Output directory: {current_obj_output_dir}")

        source_obj_filepath = os.path.join(args.source_obj_dir, obj_filename)

        # (1) Read the file and determine original vertices
        original_vertices = load_vertices_from_obj(source_obj_filepath)
        if not original_vertices:
            print(f"  Warning: No vertices found in '{obj_filename}'. Skipping.")
            continue

        # (1.cont) Calculate offset to center and get centered OBJ lines
        original_vertices_np = np.array(original_vertices)
        x_min_orig, y_min_orig, z_min_orig = np.min(original_vertices_np, axis=0)
        x_max_orig, y_max_orig, z_max_orig = np.max(original_vertices_np, axis=0)
        offset_to_center = np.array(
            [(x_min_orig + x_max_orig) / 2.0, (y_min_orig + y_max_orig) / 2.0, (z_min_orig + z_max_orig) / 2.0]
        )
        print(f"  Calculated offset to center: {offset_to_center}")
        centered_obj_lines_list = get_translated_obj_lines(
            source_obj_filepath, offset_to_center
        )  # translate by -offset

        # (2) Get bounding box for sites from centered vertices
        all_object_vertices = load_vertices_from_obj_lines(
            centered_obj_lines_list
        )  # This will now collect centered vertices
        if not all_object_vertices:  # Should not happen if original_vertices was populated and translation worked
            print(f"  Warning: No vertices found after centering for '{obj_filename}'. Skipping.")
            continue

        # (3) Break down the centered obj file into each convex part, save them under meshes
        print("  Splitting centered OBJ data into parts (based on 'o' tags)...")
        original_basename = os.path.splitext(obj_filename)[0]
        created_visual_part_basenames = split_obj_lines_into_parts(
            centered_obj_lines_list, visual_mesh_dir, original_basename
        )

        if not created_visual_part_basenames:
            print(f"  Warning: No mesh parts were generated for '{obj_filename}'. Skipping XML generation.")
            continue

        model_xml_path = os.path.join(current_obj_output_dir, "model.xml")
        model_name = f"agod_{five_digit_id}"

        mujoco_node = ET.Element("mujoco", model=model_name)
        asset_node = ET.SubElement(mujoco_node, "asset")
        ET.SubElement(
            asset_node, "material", name="default_material", specular="0.5", shininess="0.25", rgba="0.7 0.7 0.7 1"
        )

        worldbody_node = ET.SubElement(mujoco_node, "worldbody")
        unnamed_body_node = ET.SubElement(worldbody_node, "body")  # The unnamed parent body
        object_body_node = ET.SubElement(unnamed_body_node, "body", name="object")

        object_body_node.append(ET.Comment(" Visual Geometry "))
        object_body_node.append(
            ET.Comment(" Collision Geometry ")
        )  # Will interleave visual and collision for each part

        all_o3d_visual_meshes_for_bounds = []

        for visual_part_basename in created_visual_part_basenames:
            part_base_name_for_xml = sanitize_name_for_xml_and_filename(os.path.splitext(visual_part_basename)[0])

            # Visual Mesh Asset and Geom
            visual_asset_name = f"{part_base_name_for_xml}_vis"
            visual_mesh_file_relpath = os.path.join("visual", visual_part_basename).replace("\\", "/")
            ET.SubElement(
                asset_node, "mesh", name=visual_asset_name, file=visual_mesh_file_relpath, scale="1.0 1.0 1.0"
            )
            ET.SubElement(
                object_body_node,
                "geom",
                type="mesh",
                mesh=visual_asset_name,
                material="default_material",
                conaffinity="0",
                contype="0",
                group="1",
            )
            print(f"    Added visual mesh: {visual_part_basename}")

            # Load visual mesh for bounding box calculation and collision generation
            visual_mesh_full_path = os.path.join(visual_mesh_dir, visual_part_basename)
            try:
                mesh_o3d = o3d.io.read_triangle_mesh(visual_mesh_full_path)
                if mesh_o3d.has_vertices():
                    all_o3d_visual_meshes_for_bounds.append(mesh_o3d)
                else:
                    print(f"    Warning: Visual part {visual_part_basename} is empty or unreadable for bounds.")
            except Exception as e:
                print(f"    Warning: Could not load visual part {visual_part_basename} for bounds: {e}")
                continue  # Skip collision generation if visual part is bad

            # Collision Mesh Asset and Geom
            collision_asset_name = f"{part_base_name_for_xml}_coll"
            collision_basename = f"{part_base_name_for_xml}_coll.obj"
            collision_mesh_full_path = os.path.join(collision_mesh_dir, collision_basename)
            collision_mesh_file_relpath = os.path.join("collision", collision_basename).replace("\\", "/")

            simplified_mesh_o3d = None
            try:
                if not mesh_o3d.has_vertices():  # Should have been caught above, but double check
                    raise ValueError(f"Mesh {visual_part_basename} is empty, cannot simplify.")

                mesh_o3d.remove_duplicated_vertices()
                mesh_o3d.remove_duplicated_triangles()
                mesh_o3d.remove_degenerate_triangles()
                mesh_o3d.remove_unreferenced_vertices()

                current_triangles = len(mesh_o3d.triangles)
                if current_triangles > 0 and args.target_triangles > 0 and args.target_triangles < current_triangles:
                    print(
                        f"      Simplifying {visual_part_basename} from {current_triangles} to {args.target_triangles} triangles for collision."
                    )
                    simplified_mesh_o3d = mesh_o3d.simplify_quadric_decimation(
                        target_number_of_triangles=int(args.target_triangles)
                    )
                else:
                    if current_triangles == 0:
                        print(
                            f"      Warning: Mesh {visual_part_basename} has no triangles after cleaning. Using its current form for collision."
                        )
                    elif not (args.target_triangles > 0 and args.target_triangles < current_triangles):
                        print(
                            f"      Info: Simplification not performed for {visual_part_basename} (current: {current_triangles}, target: {args.target_triangles}). Using its current form for collision."
                        )
                    simplified_mesh_o3d = mesh_o3d  # Use cleaned mesh if not simplifying

                o3d.io.write_triangle_mesh(
                    collision_mesh_full_path, simplified_mesh_o3d, write_vertex_normals=False, write_vertex_colors=False
                )
                ET.SubElement(
                    asset_node, "mesh", name=collision_asset_name, file=collision_mesh_file_relpath, scale="1.0 1.0 1.0"
                )
                ET.SubElement(
                    object_body_node,
                    "geom",
                    type="mesh",
                    mesh=collision_asset_name,
                    solimp="0.998 0.998 0.001",
                    solref="0.001 1",
                    density="100",
                    friction="0.95 0.3 0.1",
                    group="0",
                    rgba="0.8 0.8 0.8 0.0",
                )
                print(f"    Added collision mesh: {collision_basename}")

            except Exception as e:
                print(
                    f"    Error processing/simplifying {visual_part_basename} for collision: {e}. Skipping collision mesh for this part."
                )

        # Calculate overall bounding box for sites from all loaded visual meshes
        if all_o3d_visual_meshes_for_bounds:
            min_bound_overall = np.array([np.inf, np.inf, np.inf])
            max_bound_overall = np.array([-np.inf, -np.inf, -np.inf])
            has_valid_bounds = False
            for mesh in all_o3d_visual_meshes_for_bounds:
                if mesh.has_vertices():  # Ensure mesh is not empty
                    min_bound_overall = np.minimum(min_bound_overall, mesh.get_min_bound())
                    max_bound_overall = np.maximum(max_bound_overall, mesh.get_max_bound())
                    has_valid_bounds = True

            if has_valid_bounds and not np.isinf(min_bound_overall).any() and not np.isinf(max_bound_overall).any():
                center_overall = (min_bound_overall + max_bound_overall) / 2.0
                # Sites are relative to the unnamed_body_node, and object_body_node is at its origin.
                # Since meshes are already centered, their bounds are relative to this origin.
                pos_bottom_site = f"{center_overall[0]:.5f} {center_overall[1]:.5f} {min_bound_overall[2]:.5f}"
                pos_top_site = f"{center_overall[0]:.5f} {center_overall[1]:.5f} {max_bound_overall[2]:.5f}"
                pos_horizontal_site = f"{max_bound_overall[0]:.5f} {max_bound_overall[1]:.5f} {center_overall[2]:.5f}"

                common_site_attrs = {"rgba": SITE_RGBA, "size": SITE_SIZE}
                ET.SubElement(
                    unnamed_body_node, "site", {**common_site_attrs, "name": "bottom_site", "pos": pos_bottom_site}
                )
                ET.SubElement(unnamed_body_node, "site", {**common_site_attrs, "name": "top_site", "pos": pos_top_site})
                ET.SubElement(
                    unnamed_body_node,
                    "site",
                    {**common_site_attrs, "name": "horizontal_radius_site", "pos": pos_horizontal_site},
                )
                print(f"    Added bottom_site at: {pos_bottom_site}")
                print(f"    Added top_site at: {pos_top_site}")
                print(f"    Added horizontal_radius_site at: {pos_horizontal_site}")
            else:
                print(
                    "    Warning: Could not determine valid overall bounding box for sites. Using default site positions."
                )
                ET.SubElement(
                    unnamed_body_node, "site", name="bottom_site", pos="0 0 -0.05", size=SITE_SIZE, rgba=SITE_RGBA
                )
                ET.SubElement(
                    unnamed_body_node, "site", name="top_site", pos="0 0 0.05", size=SITE_SIZE, rgba=SITE_RGBA
                )
                ET.SubElement(
                    unnamed_body_node,
                    "site",
                    name="horizontal_radius_site",
                    pos="0.05 0.05 0",
                    size=SITE_SIZE,
                    rgba=SITE_RGBA,
                )
        else:
            print(
                "    Warning: No valid visual meshes found for bounding box calculation. Skipping site generation or using defaults."
            )
            ET.SubElement(
                unnamed_body_node, "site", name="bottom_site", pos="0 0 -0.05", size=SITE_SIZE, rgba=SITE_RGBA
            )
            ET.SubElement(unnamed_body_node, "site", name="top_site", pos="0 0 0.05", size=SITE_SIZE, rgba=SITE_RGBA)
            ET.SubElement(
                unnamed_body_node,
                "site",
                name="horizontal_radius_site",
                pos="0.05 0.05 0",
                size=SITE_SIZE,
                rgba=SITE_RGBA,
            )

        xml_string_pretty = prettify_xml(mujoco_node)
        with open(model_xml_path, "w") as f:
            f.write(xml_string_pretty)

        # Save the dimension, so object size can be easily read
        object_size = {
            "dim_x": x_max_orig - x_min_orig,
            "dim_y": y_max_orig - y_min_orig,
            "dim_z": z_max_orig - z_min_orig,
        }
        with open(os.path.join(current_obj_output_dir, "object_size.json"), "w") as f:
            json.dump(object_size, f)

        print(f"  Generated MuJoCo XML: {model_xml_path}")
        processed_count += 1

    print(f"\nFinished processing. {processed_count} OBJ files processed.")


if __name__ == "__main__":
    main()
