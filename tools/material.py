# GPL License

import os
import bpy
from bpy.types import (
    Mesh,
    Material,
    Context,
    Object,
    NodeSocket,
    Node,
    bpy_prop_array,
)
import numpy as np
from typing import Union, Iterable

from . import common as Common
from .register import register_wrap
from .translations import t

IGNORED_MAT_HASH_NODE_LABELS = {"Material Output", "mmd_tex_uv", "Cats Export Shader"}
IGNORED_MAT_HASH_NODE_NAMES = IGNORED_MAT_HASH_NODE_LABELS  # The same as labels for now.
IGNORED_MAT_HASH_NODE_ID_NAMES = {"ShaderNodeOutputMaterial"}


@register_wrap
class CombineMaterialsButton(bpy.types.Operator):
    bl_idname = "cats_material.combine_mats"
    bl_label = t("CombineMaterialsButton.label")
    bl_description = t("CombineMaterialsButton.desc")
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    @staticmethod
    def get_meshes():
        return Common.get_meshes_objects(check=False)

    @classmethod
    def poll(cls, context: Context):
        if context.mode.startswith("EDIT"):
            # The meshes to operate on can't be in edit mode, and the current mode can't be an edit mode either,
            # otherwise undo/redo won't work properly because we will be modifying meshes that are not in the current
            # edit mode.
            return False

        if Common.get_armature() is None:
            return False

        meshes = cls.get_meshes()
        if not meshes:
            return False

        for me in meshes:
            if me.mode == 'EDIT':
                # The material indices of polygons cannot be get/set quickly while a mesh is in edit mode (and
                # completely different code would be needed).
                return False
        return True

    @staticmethod
    def combine_exact_duplicate_mats(ob: Object, unique_sorted_mat_indices: np.ndarray):
        mat_names = ob.material_slots.keys()

        # Find duplicate materials and get the first index of the material slot with that same material
        mat_first_occurrence: dict[str, int] = {}
        # Note that empty material slots use '' as the material name
        for i, mat_name in enumerate(mat_names):
            if mat_name not in mat_first_occurrence:
                # This is the first time we've seen this material, add it to the first occurrence dict with the current
                # index
                mat_first_occurrence[mat_name] = i
            else:
                # We've seen this material already, find its occurrences (if any) in the unique mat indices array and
                # set it to the index of the first occurrence of this material
                unique_sorted_mat_indices[unique_sorted_mat_indices == i] = mat_first_occurrence[mat_name]

        return unique_sorted_mat_indices

    @staticmethod
    def remove_unused_mat_slots(ob: Object, used_mat_indices: Iterable[int]):
        mesh: Mesh = ob.data
        used_mat_indices = set(used_mat_indices)

        # We iterate in reverse order, so that removing a material doesn't change the indices of any material slots we
        # are yet to iterate.
        for i in reversed(range(len(mesh.materials))):
            if i not in used_mat_indices:
                mesh.materials.pop(index=i)

    @staticmethod
    def generate_mat_hash(mat: Material) -> str:
        if not mat:
            return ""

        node_tree = mat.node_tree
        if not mat.use_nodes or not node_tree:
            # Materials almost always use nodes, but on the off chance that a material doesn't, create the hash
            # based on the non-node properties
            return str(mat.diffuse_color[:]) + str(mat.metallic) + str(mat.roughness) + str(mat.specular_intensity)

        hash_parts: list[str] = []
        nodes = mat.node_tree.nodes

        # NodeSocket.links iterates through all links in the node tree every time it is called, we don't want to do this
        # for every input socket that is linked, so we'll store a mapping from input sockets to links.
        input_socket_to_from_node_lookup: dict[NodeSocket, Union[Node, set[Node]]] = {}
        for link in node_tree.links:
            # Each linked input socket will usually only be linked to one node, since it's typically only geometry nodes
            # that support sockets with multiple links.
            to_socket = link.to_socket
            if to_socket.link_limit == 1:
                input_socket_to_from_node_lookup[to_socket] = link.from_node
            else:
                input_socket_to_from_node_lookup.setdefault(link.to_socket, set()).add(link.from_node)

        # TODO: The iteration through nodes in this function needs a re-work because the order of the nodes in a node
        #  tree is not well-defined and can change even when only changing which node is active. Iterating the entire
        #  node tree also means disconnected nodes are included in the hash which is not ideal.
        for node in nodes:
            node_name = node.name

            # Skip certain known nodes
            if node_name in IGNORED_MAT_HASH_NODE_NAMES:
                continue
            if node.label in IGNORED_MAT_HASH_NODE_LABELS:
                continue
            node_idname = node.bl_idname
            if node_idname in IGNORED_MAT_HASH_NODE_ID_NAMES:
                continue
            # Skip nodes with no input
            node_inputs = node.inputs
            if not node_inputs:
                continue

            # Add images to hash and skip toon and sphere textures used by mmd
            if node_idname == "ShaderNodeTexImage":
                if "toon" in node_name or "sphere" in node_name:
                    continue
                image = node.image
                if not image:
                    continue
                hash_parts.append(node_name)
                hash_parts.append(image.name)
            # On MMD models only add diffuse and transparency to the hash
            elif node_idname == "ShaderNodeGroup" and node_name == "mmd_shader":
                hash_parts.append(node_name)
                hash_parts.append(str(node_inputs["Diffuse Color"].default_value[:]))
                hash_parts.append(str(node_inputs["Alpha"].default_value))
            else:
                # Add the node name and its inputs to the hash
                hash_parts.append(node_name)
                for node_input in node_inputs:
                    if node_input.is_linked:
                        from_node = input_socket_to_from_node_lookup[node_input]
                        if isinstance(from_node, Node):
                            hash_parts.append(from_node.name)
                        else:
                            from_nodes = from_node
                            for from_node in from_nodes:
                                hash_parts.append(from_node.name)
                    elif (node_input_default_value := getattr(node_input, "default_value", None)) is not None:
                        if isinstance(node_input_default_value, bpy_prop_array):
                            # Should be NodeSockets with the 'VECTOR' and 'RGBA' types.
                            hash_parts.append(str(node_input_default_value[:]))
                        else:
                            hash_parts.append(str(node_input_default_value))
                    else:
                        hash_parts.append(node_input.name)

        return "".join(hash_parts)

    def execute(self, context: Context) -> set[str]:
        print("COMBINE MATERIALS!")
        num_combined = 0

        # Hashes of all found materials
        mat_hashes: dict[str, str] = {}
        # The first material found for each hash
        first_mats_by_hash: dict[str, Material] = {}
        for mesh_obj in self.get_meshes():
            # Generate material hashes and re-assign material slots to the first found material that produces the same
            # hash.
            for mat_name, mat_slot in mesh_obj.material_slots.items():
                mat = mat_slot.material

                # Get the material hash, generating it if needed
                if mat_name not in mat_hashes:
                    mat_hash = self.generate_mat_hash(mat)
                    mat_hashes[mat_name] = mat_hash
                else:
                    mat_hash = mat_hashes[mat_name]

                # If a material with the same hash has already been found, re-assign the material slot to the previously
                # found material, otherwise, add the material to the dictionary of first found materials
                if mat_hash in first_mats_by_hash:
                    replacement_material = first_mats_by_hash[mat_hash]
                    # The replacement_material material could be the current material if the current material was also
                    # used on another mesh that was iterated before this mesh.
                    if mat != replacement_material:
                        mat_slot.material = replacement_material
                        num_combined += 1
                else:
                    first_mats_by_hash[mat_hash] = mat
            # Combine exact duplicate materials within the same mesh
            mesh: Mesh = mesh_obj.data
            if bpy.app.version < (3, 4):
                material_indices_collection = mesh.polygons
                data_attribute = "material_index"
                data_dtype = np.ushort
            else:
                attribute = mesh.attributes.get("material_index")
                if attribute.domain != 'FACE' or attribute.data_type != 'INT':
                    material_indices_collection = None
                else:
                    material_indices_collection = attribute.data
                data_attribute = "value"
                data_dtype = np.intc

            if material_indices_collection:
                # Get polygon material indices
                material_indices = np.empty(len(material_indices_collection), dtype=data_dtype)
                material_indices_collection.foreach_get(data_attribute, material_indices)

                # Find unique sorted material indices and get the inverse array to reconstruct the material indices
                # array.
                unique_sorted_material_indices, unique_inverse = np.unique(material_indices, return_inverse=True)
                # Working with only the unique material indices means we don't need to operate on the entire array.
                combined_material_indices = self.combine_exact_duplicate_mats(mesh_obj, unique_sorted_material_indices)

                # Update the material indices
                material_indices_collection.foreach_set(data_attribute, combined_material_indices[unique_inverse])
            else:
                combined_material_indices = [0]

            # Remove any unused material slots
            self.remove_unused_mat_slots(mesh_obj, combined_material_indices)

            # Clean material names
            Common.clean_material_names(mesh_obj)

        if num_combined == 0:
            self.report({'INFO'}, t("CombineMaterialsButton.error.noChanges"))
        else:
            self.report({'INFO'}, t("CombineMaterialsButton.success", number=str(num_combined)))

        return {'FINISHED'}


@register_wrap
class ConvertAllToPngButton(bpy.types.Operator):
    bl_idname = 'cats_material.convert_all_to_png'
    bl_label = t('ConvertAllToPngButton.label')
    bl_description = t('ConvertAllToPngButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    # Inspired by:
    # https://cdn.discordapp.com/attachments/387450722410561547/526638724570677309/BlenderImageconvert.png

    @classmethod
    def poll(cls, context):
        return bpy.data.images

    def execute(self, context):
        images_to_convert = self.get_convert_list()

        if images_to_convert:
            current_step = 0
            wm = bpy.context.window_manager
            wm.progress_begin(current_step, len(images_to_convert))

            for image in images_to_convert:
                self.convert(image)
                current_step += 1
                wm.progress_update(current_step)

            wm.progress_end()

        self.report({'INFO'}, t('ConvertAllToPngButton.success', number=str(len(images_to_convert))))
        return {'FINISHED'}

    def get_convert_list(self):
        images_to_convert = []
        for image in bpy.data.images:
            # Get texture path and check if the file should be converted
            tex_path = bpy.path.abspath(image.filepath)
            if tex_path.endswith(('.png', '.spa', '.sph')) or not os.path.isfile(tex_path):
                print('IGNORED:', image.name, tex_path)
                continue
            images_to_convert.append(image)
        return images_to_convert

    def convert(self, image):
        # Set the new image file name
        image_name = image.name
        print(image_name)
        image_name_new = ''
        for s in image_name.split('.')[0:-1]:
            image_name_new += s + '.'
        image_name_new += 'png'
        print(image_name_new)

        # Set the new image file path
        tex_path = bpy.path.abspath(image.filepath)
        print(tex_path)
        tex_path_new = ''
        for s in tex_path.split('.')[0:-1]:
            tex_path_new += s + '.'
        tex_path_new += 'png'
        print(tex_path_new)

        # Save the Color Management View Transform and change it to Standard, as any other would screw with the colors
        view_transform = bpy.context.scene.view_settings.view_transform
        bpy.context.scene.view_settings.view_transform = 'Standard'

        # Save the image as a new png file
        scene = bpy.context.scene
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '16'
        scene.render.image_settings.compression = 100
        image.save_render(tex_path_new, scene=scene)  # TODO: FInd out how to use image.save here, to prevent anything from changing the colors

        # Change the view transform back
        bpy.context.scene.view_settings.view_transform = view_transform

        # Exchange the old image in blender for the new one
        bpy.data.images[image_name].filepath = tex_path_new
        bpy.data.images[image_name].name = image_name_new

        return True
