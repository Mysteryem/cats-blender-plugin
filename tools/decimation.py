# MIT License

# Copyright (c) 2018 Hotox

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Code author: Hotox
# Repo: https://github.com/michaeldegroot/cats-blender-plugin
# Edits by:

import bpy
import math
import mathutils
import struct
import bmesh
import numpy as np
from time import perf_counter

from . import common as Common
from . import armature_bones as Bones
from .register import register_wrap
from .translations import t


ignore_shapes = []
ignore_meshes = []


@register_wrap
class ScanButton(bpy.types.Operator):
    bl_idname = 'cats_decimation.auto_scan'
    bl_label = t('ScanButton.label')
    bl_description = t('ScanButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        if context.scene.add_shape_key == "":
            return False

        return True

    def execute(self, context):
        shape = context.scene.add_shape_key
        shapes = Common.get_shapekeys_decimation_list(self, context)
        count = len(shapes)

        if count > 1 and shapes.index(shape) == count - 1:
            context.scene.add_shape_key = shapes[count - 2]

        ignore_shapes.append(shape)

        return {'FINISHED'}


@register_wrap
class AddShapeButton(bpy.types.Operator):
    bl_idname = 'cats_decimation.add_shape'
    bl_label = t('AddShapeButton.label')
    bl_description = t('AddShapeButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        if context.scene.add_shape_key == "":
            return False
        return True

    def execute(self, context):
        shape = context.scene.add_shape_key
        shapes = [x[0] for x in Common.get_shapekeys_decimation_list(self, context)]
        count = len(shapes)

        if count > 1 and shapes.index(shape) == count - 1:
            context.scene.add_shape_key = shapes[count - 2]

        ignore_shapes.append(shape)

        return {'FINISHED'}


@register_wrap
class AddMeshButton(bpy.types.Operator):
    bl_idname = 'cats_decimation.add_mesh'
    bl_label = t('AddMeshButton.label')
    bl_description = t('AddMeshButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        if context.scene.add_mesh == "":
            return False
        return True

    def execute(self, context):
        ignore_meshes.append(context.scene.add_mesh)

        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                if obj.parent and obj.parent.type == 'ARMATURE' and obj.parent.name == bpy.context.scene.armature:
                    if obj.name in ignore_meshes:
                        continue
                    context.scene.add_mesh = obj.name
                    break

        return {'FINISHED'}


@register_wrap
class RemoveShapeButton(bpy.types.Operator):
    bl_idname = 'cats_decimation.remove_shape'
    bl_label = t('RemoveShapeButton.label')
    bl_description = t('RemoveShapeButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    shape_name = bpy.props.StringProperty()

    def execute(self, context):
        ignore_shapes.remove(self.shape_name)

        return {'FINISHED'}


@register_wrap
class RemoveMeshButton(bpy.types.Operator):
    bl_idname = 'cats_decimation.remove_mesh'
    bl_label = t('RemoveMeshButton.label')
    bl_description = t('RemoveMeshButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    mesh_name = bpy.props.StringProperty()

    def execute(self, context):
        ignore_meshes.remove(self.mesh_name)

        return {'FINISHED'}


@register_wrap
class AutoDecimateButton(bpy.types.Operator):
    bl_idname = 'cats_decimation.auto_decimate'
    bl_label = t('AutoDecimateButton.label')
    bl_description = t('AutoDecimateButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    armature_name = bpy.props.StringProperty(
        name='armature_name',
    )

    preserve_seams = bpy.props.BoolProperty(
        name='preserve_seams',
    )

    seperate_materials = bpy.props.BoolProperty(
        name='seperate_materials'
    )

    cats_animation_vg_name = "CATS Animation"

    def execute(self, context):
        start = perf_counter()
        meshes = Common.get_meshes_objects()
        if not meshes or len(meshes) == 0:
            self.report({'ERROR'}, t('AutoDecimateButton.error.noMesh'))
            return {'FINISHED'}

        saved_data = Common.SavedData()

        if context.scene.decimation_mode != 'CUSTOM':
            mesh = Common.join_meshes(repair_shape_keys=False, armature_name=self.armature_name)
            if self.seperate_materials:
                Common.separate_by_materials(context, mesh)

        self.decimate(context)

        Common.join_meshes(armature_name=self.armature_name)

        saved_data.load()

        end = perf_counter()
        print(f'Decimation finished in {end - start} seconds')

        return {'FINISHED'}

    @staticmethod
    def _get_animation_weighting(mesh, armature=None):
        bone_weights = dict()

        # It would be nice to get some explanation on what the maths in this. What is the relevance of pairing each
        # vertex group together? - Mysteryem
        def calc_weight_pairs(v_idx, groups):
            # This has a time complexity of O((m^2 - m)/2), which is still the same class as O(m^2) for iterating
            # groups in its entirety in two loops, but will result in just under half the number of iterations.
            #
            # We want to skip the case when w1 == w2, and, because multiplication is commutative, we know that (w1, w2)
            # will result in the same values as (w2, w1), so we want to skip those as well.
            # If we were to iterate the list [0,1,2], 0 only needs to be paired with [1,2], 1 only needs to be
            # paired with [2] and 2 doesn't need to be paired with anything.
            # As a general formula, my_list[n] only needs to be paired with my_list[n+1:].
            # Since the last element doesn't need to be paired with anything, we can skip it in the initial iteration.
            for g_idx, w1 in enumerate(groups[:-1]):
                w1_group = w1.group
                w1_weight = w1.weight
                # Now iterate over the remaining groups that w1 needs to be paired with
                for w2 in groups[g_idx + 1:]:
                    w2_group = w2.group
                    # w1_group and w2_group can be in any order so sort them when making the key
                    key = (w1_group, w2_group) if w1_group < w2_group else (w2_group, w1_group)
                    # Weight [vgroup * vgroup] for index = <mult>
                    weight = w1_weight * w2.weight
                    if key not in bone_weights:
                        # Add a new dictionary
                        bone_weights[key] = {v_idx: weight}
                    else:
                        # Add the value 'weight' with key 'v_idx'
                        bone_weights[key][v_idx] = weight

        # Weight by multiplied bone weights for every pair of bones.
        # This is O(n*m^2) for n verts and m bones in the worst case of every vertex being assigned to every bone's
        # vertex group.
        # Generally runs relatively quickly since each vertex is likely to only be assigned to around 4 or fewer
        # vertex groups, making it much more like O(n) in most cases
        if armature is not None:
            deform_bone_names = Common.get_deform_bone_names(mesh, armature)
            deform_bone_vg_idx = {mesh.vertex_groups[name].index for name in deform_bone_names}
            for v_idx, vertex in enumerate(mesh.data.vertices):
                deform_groups = [g for g in vertex.groups if g.group in deform_bone_vg_idx]
                calc_weight_pairs(v_idx, deform_groups)
        else:
            for v_idx, vertex in enumerate(mesh.data.vertices):
                calc_weight_pairs(v_idx, vertex.groups)

        # Normalize per vertex group pair, in-place
        for pair, weighting in bone_weights.items():
            weight_vals = weighting.values()
            m_min = min(weight_vals)
            m_max = max(weight_vals)
            # m_max and m_min will always be in the range [0,1] since vertex group weights are always in the range [0,1]
            # and [0,1] * [0,1] -> [0,1]
            diff = m_max - m_min
            # Normalize and update each weight
            # If diff equals zero then max and min are the same, meaning all these weight values must be the same. There
            # is no min and max to normalize them between, so we'll leave them unchanged
            # TODO: When diff == 0, is there a specific value that should be set instead of leaving the weights
            #       unchanged?
            if diff != 0:
                for v_index, weight in weighting.items():
                    weighting[v_index] = (weight - m_min) / diff

        newweights = dict()
        # Collect all the normalized weights for each vertex
        for weighting in bone_weights.values():
            for v_index, weight in weighting.items():
                weights = newweights.setdefault(v_index, [])
                weights.append(weight)

        s_weights = dict()
        # FIXME: This assumes the first shape key is always the relative key
        # TODO: This can be done ~10 times quicker with numpy and foreach_get, but the rest of this function and its
        #       uses would have to be changed to use ndarrays instead of nested dictionaries
        # Weight by relative shape key movement. This is kind of slow, but not too bad. It's O(n*m) for n verts and m shape keys,
        # but shape keys contain every vert (not just the ones they impact)
        if mesh.data.shape_keys is not None and len(mesh.data.shape_keys.key_blocks) > 1:
            key_blocks = mesh.data.shape_keys.key_blocks

            # Pre-allocate a dictionary per shape key other than the basis so there's no need to spend time checking if
            # a shape key's dictionary already exists, and creating it if not, while iterating
            s_idx_weights = {key_idx: {} for key_idx in range(1, len(key_blocks))}
            # We want to make sure we only iterate through the data (vertex) of each shape key at most once
            # So, iterate through each datum (vertex) in each key_block simultaneously
            for vert_idx, kb_data_tuple in enumerate(zip(*(kb.data for kb in key_blocks))):
                # The first element in the tuple will be the data for the basis shape key
                basis_co = kb_data_tuple[0].co
                # For the rest of the key_blocks in the tuple:
                for key_idx, key_datum in enumerate(kb_data_tuple[1:], start=1):
                    # Find the distance between the basis and the current key_block and add that distance to
                    # the dictionary of the current key_block with the vertex index as the key
                    s_idx_weights[key_idx][vert_idx] = (basis_co - key_datum.co).length

            # The current s_idx_weights' keys are the index of each key_block, but we want the keys to be the
            # key_blocks' names so remap from key_idx:vert_distances to key_name:vert_distances
            s_weights = {key_blocks[key_idx].name: vert_distances for key_idx, vert_distances in s_idx_weights.items()}

        # normalize min/max vert movement, in-place
        for keyname, weighting in s_weights.items():
            weight_vals = weighting.values()
            m_min = min(weight_vals)
            m_max = max(weight_vals)

            diff = m_max - m_min

            # Normalize and update each weight
            # If diff equals zero then max and min are the same meaning all these weight values must be the same. There
            # is no min and max to normalize them between, so we'll leave them unchanged
            # TODO: When diff == 0, wouldn't a value in [0,1] be better than leaving the weights unchanged?
            if diff != 0:
                for v_index, weight in weighting.items():
                    weighting[v_index] = (weight - m_min) / diff

        # collect all the normalized movement over all shape keys into newweights
        for pair, weighting in s_weights.items():
            for v_index, weight in weighting.items():
                weights = newweights.setdefault(v_index, [])
                weights.append(weight)

        # Update each {key:value_list} pair to {key:max(value_list)}
        for v_index, weights in newweights.items():
            newweights[v_index] = max(weights)

        return newweights

    @staticmethod
    def _create_cats_animation_vertex_groups(meshes_obj, armature_obj):
        for mesh in meshes_obj:
            newweights = AutoDecimateButton._get_animation_weighting(mesh, armature_obj)

            # Vertices weight painted to a single bone are likely to end up with the same newweight
            # Mirrored meshes are likely to have the same newweight for each pair of mirrored vertices
            # All vertices with the same weight can be updated at the same time, so we'll flip the keys and values
            # to get all the vertices which need to be set to each unique weight
            weight_to_indices = {}
            for idx, weight in newweights.items():
                vertex_indices = weight_to_indices.setdefault(weight, [])
                vertex_indices.append(idx)

            cats_animation_vg = mesh.vertex_groups.new(name=AutoDecimateButton.cats_animation_vg_name)

            for weight, vertex_indices in weight_to_indices.items():
                cats_animation_vg.add(vertex_indices, weight, "REPLACE")

    @staticmethod
    def _separate_fingers(meshes_obj):
        """For each mesh, separate all the vertices belonging to a finger vertex group into a new mesh."""
        def select_fingers(mesh_obj):
            vertex_groups = mesh_obj.vertex_groups
            finger_bones = Bones.bone_finger_list
            # Irrelevant for 2.79 since only the active object can be opened in edit mode, but for 2.80+, which can
            # open multiple objects in edit mode at the same time, it makes it so there's no need to keep changing
            # the active object in order to use the vertex_group_select operator.
            context_override = {'object': mesh_obj}

            for side_suffix in ['L', 'R']:
                for finger_bone in finger_bones:
                    vertex_group = vertex_groups.get(finger_bone + side_suffix)
                    if vertex_group:
                        mesh_obj.vertex_groups.active_index = vertex_group.index
                        bpy.ops.object.vertex_group_select(context_override)

        def separate_selected():
            try:
                bpy.ops.mesh.separate(type='SELECTED')
            except RuntimeError:
                # Raises RuntimeError when there's nothing selected
                pass

        # Clear current selection so we don't edit anything extra
        Common.unselect_all()

        if Common.version_2_79_or_older():
            # 2.79 can only open one mesh in edit mode at a time, so we have to edit them one by one
            for mesh in meshes_obj:
                if len(mesh.vertex_groups) > 0:
                    Common.select(mesh)
                    Common.switch('EDIT')
                    bpy.ops.mesh.select_mode(type='VERT')
                    select_fingers(mesh)
                    # Separate the selected vertices of the mesh into a separate mesh
                    separate_selected()
        else:
            # 2.80+ can open multiple meshes in edit mode at the same time
            # Select all the mesh objects with vertex groups and add them to a list for further iteration later on
            meshes_with_vertex_groups = []
            for mesh in meshes_obj:
                if len(mesh.vertex_groups) > 0:
                    Common.select(mesh)
                    meshes_with_vertex_groups.append(mesh)

            # Open the selected meshes in edit mode
            Common.switch('EDIT')
            bpy.ops.mesh.select_mode(type='VERT')

            # For each mesh, select all the vertices in any finger vertex group
            for mesh in meshes_with_vertex_groups:
                select_fingers(mesh)

            # Separate the selected vertices of each mesh into a separate mesh (one new mesh per mesh with at least
            # one vertex selected)
            separate_selected()

        # Go back to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

    @staticmethod
    def _get_loop_decimate_weights(context, mesh_obj):
        mod = mesh_obj.modifiers.new("Decimate", 'DECIMATE')
        mod.use_symmetry = True
        mod.symmetry_axis = 'X'

        # Dark magic... encode the vert index into (usually) the x components of a UV map

        # The itemsize of the single precision float and uint types may vary depending on compiler
        # Typically, both will be 4 bytes.
        # We can support uint smaller than single, but we can only support uint up to twice the size of single.
        # Any more than twice the size and there wouldn't be enough space in the uvs of the uv map to store the
        # vertex indices, though this is a very unlikely case.
        vertex_index_type = np.dtype(np.uintc)
        uv_type = np.dtype(np.single)
        more_than_one_index_can_fit_in_single = vertex_index_type.itemsize < uv_type.itemsize
        if more_than_one_index_can_fit_in_single:
            # More than one uintc can fit in a single precision float
            indices_per_uv_component = uv_type.itemsize // vertex_index_type.itemsize
            view_slice = slice(None, None, 2 * indices_per_uv_component)
        else:
            # Either uintc and single are the same size or more than one single is needed to represent a uintc
            # (we can only support up to 2 single precision floats)
            uv_components_per_index = vertex_index_type.itemsize // uv_type.itemsize
            if uv_components_per_index > 2:
                raise RuntimeError(
                    """There is not enough space to store uint vertex indices in a single precision float uv map in this build of blender.
                    If you want to use loop decimation, please compile Blender such that sizeof(uint) is no more than twice sizeof(float).
                    If you are seeing this message and didn't compile Blender yourself, please report a bug.""")
            view_slice = slice(None, None, 2 // uv_components_per_index)

        loops = mesh_obj.data.loops
        loop_vertex_indices = np.empty(len(loops), dtype=vertex_index_type)
        # Ideally, we would foreach_get directly into a uvs array viewed as uintc, but blender sees it as a
        # different type, seemingly 'B' (unsigned char), meaning Blender would have to iterate and cast every
        # element, which is slower than a direct copy when the type matches 'I' (uintc).
        loops.foreach_get('vertex_index', loop_vertex_indices)

        # Create the uv map
        vert_uv_layer = mesh_obj.data.uv_layers.new(name='CATS Vert', do_init=False)
        if not vert_uv_layer:
            # vert_uv_layer will be None if it could not be created, this usually occurs when mesh_obj already has 8
            # (the maximum) uv maps.
            # We could copy all of an existing UV Layer's data (uv, pin_uv and select) into ndarrays and restore them
            # later, though this wouldn't account for any custom data added by addons.
            raise RuntimeError("Cannot loop decimate {} as a new uv map could not be created, it may already have the maximum (8) number of uv maps".format(mesh_obj))

        # len(vert_uv_layer.data) == len(loops)
        vert_uvs = np.empty(len(loops) * 2, dtype=uv_type)
        # View the uvs as np.uintc and set the vertex indices into the data according to the slice
        vert_uvs.view(vertex_index_type)[view_slice] = loop_vertex_indices
        # Set the uvs of the uv map
        vert_uv_layer.data.foreach_set('uv', vert_uvs)

        # decimate N times, n/N% each time, and observe the result to get a list of leftover verts (the ones decimated)
        iterations = 100
        # If a vertex is never decimated, it will have a weight of zero
        # If a vertex is immediately decimated, it will have a weight of close to one
        weights_np = np.zeros(len(mesh_obj.data.vertices))

        # FIXME: We need to disable other modifiers that alter geometry because those will affect the evaluated_get!
        depsgraph = context.evaluated_depsgraph_get()
        mesh_decimated = mesh_obj.evaluated_get(depsgraph)
        last_remaining_vertex_indices = loop_vertex_indices
        # Start at almost no decimation (ratio of almost 1) and gradually decrease the ratio so that more
        # decimation occurs with each iteration
        for i in range(iterations - 1, 0, -1):
            ith_ratio = (i / iterations)
            mod.ratio = ith_ratio
            # Update for the new ratio
            depsgraph.update()
            decimated_uv_layer = mesh_decimated.data.uv_layers['CATS Vert']
            decimated_uvs = np.empty(len(decimated_uv_layer.data) * 2, dtype=uv_type)
            decimated_uv_layer.data.foreach_get('uv', decimated_uvs)
            remaining_vertex_indices = decimated_uvs.view(vertex_index_type)[view_slice]

            # TODO: Might be faster to set(remaining_vertex_indices.tolist()) and use Python Set methods with
            #  iteration like the original implementation

            # Since we only care about when we first find that a vertex has been decimated, we can compare
            # against the remaining vertex indices from the last iteration instead of the full list each time.
            # This speeds things up as the decimation ratio increases and the number of remaining vertices decreases.
            indices_decimated = np.setdiff1d(last_remaining_vertex_indices, remaining_vertex_indices)
            # print(f'indices decimated at {ith_ratio}: {(indices_decimated.tolist()}')
            # There's no guarantee that a vertex will continue to always be removed once a ratio has been reached that
            # removes that vertex, a vertex may re-appear again at a smaller ratio (more decimation).
            # We therefore get the current weights and ensure we don't set a smaller value than the existing value.
            current_weights = weights_np[indices_decimated]
            max_of_weights = np.maximum(ith_ratio, current_weights)
            # Update the weights
            weights_np[indices_decimated] = max_of_weights
            # Update the array of remaining vertex indices for the next iteration
            last_remaining_vertex_indices = remaining_vertex_indices

        # TODO: Probably faster to convert to Python array before iterating, though ideally, we would only work with
        #  ndarrays until we actually need a Python Set
        weights = {idx: weight for idx, weight in enumerate(weights_np)}

        # Remove the modifier
        mesh_obj.modifiers.remove(mod)
        # Remove the uv map
        mesh_obj.data.uv_layers.remove(vert_uv_layer)

        return weights

    # TODO: If a mesh is entirely made up of loose polygons (every polygon has its own vertices instead of sharing
    #  the vertices of its neighbours), this will be very slow. Face selection mode and bpy.ops.mesh.select_loose() may
    #  be useful since I don't see an is_loose attribute of individual polygons.
    # TODO: Skip loose edges
    # Note: Assumes the passed in mesh_obj is the currently active and only selected object
    @staticmethod
    def _get_loop_decimate_edge_loops(mesh_obj):
        timings = {'start': perf_counter()}
        me = mesh_obj.data

        edges = me.edges
        edges_left_to_select = set(range(len(edges)))
        edge_loops = []

        # When an edge is manifold geometry and not in an n-gon, if neither of its vertices are in 4 edges then the edge
        # loop it is a part of contains only itself. This allows us to quickly find the edge loops for triangulated
        # parts of a mesh.

        # First find all ngons
        polygons = me.polygons
        # Get loop totals to find which polygons are ngons
        poly_loop_totals = np.empty(len(polygons), dtype=np.uintc)
        polygons.foreach_get('loop_total', poly_loop_totals)
        poly_is_ngon = poly_loop_totals > 4
        # Repeat whether the poly is an ngon to match the loops
        loop_is_in_ngon = np.repeat(poly_is_ngon, poly_loop_totals)

        # Get edge_indices of all loops
        loops = me.loops
        loop_edges = np.empty(len(loops), dtype=np.uintc)
        loops.foreach_get('edge_index', loop_edges)

        # Get indices of all edges in ngons
        edges_in_ngons = loop_edges[loop_is_in_ngon]

        # Create a bool array for each edge which is True when the edge is in an ngon
        edge_is_not_in_ngon = np.ones(len(edges), dtype=bool)
        edge_is_not_in_ngon[edges_in_ngons] = False

        # Next find all edges which are manifold
        # Count the number of times each edge index appears in a loop
        num_loops_edge_is_in = np.bincount(loop_edges, minlength=len(me.edges))
        # To be manifold geometry, an edge must be in 2 loops
        edge_is_manifold = num_loops_edge_is_in == 2

        # Get vertex indices of all loops, these get flattened, e.g. [(0, 1), (2, 0), (2, 1)] becomes [0, 1, 2, 0, 2, 1]
        edge_verts = np.empty(len(edges) * 2, dtype=np.uintc)
        edges.foreach_get('vertices', edge_verts)
        # Count the number of times each vertex appears in an edge
        num_edges_vert_is_in = np.bincount(edge_verts, minlength=len(me.vertices))

        # A vertex in manifold geometry only connects two of its edges into an edge loop if it is in exactly 4 edges
        vert_is_non_connecting_if_manifold = num_edges_vert_is_in != 4

        # Use the vertex indices of each edge to index vert_is_non_connecting_if_manifold
        edge_verts_non_connecting_if_manifold = vert_is_non_connecting_if_manifold[edge_verts]
        # Unflatten the edge_verts and get whether both vertices in each pair are non-connecting
        edge_is_alone_in_edge_loop_if_manifold = edge_verts_non_connecting_if_manifold.reshape(-1, 2).all(axis=1)

        # Combine whether an edge is manifold, whether it's the only edge in its edge loop and whether it's not in an
        # ngon
        # np.logical_and only acts on two arrays at a time, but we can use np.logical_and.reduce to repeat the execution
        to_reduce = (edge_is_alone_in_edge_loop_if_manifold, edge_is_manifold, edge_is_not_in_ngon)
        # out is optional, but has to be either the first or second array since those are the first used
        edge_is_alone_in_edge_loop = np.logical_and.reduce(to_reduce, out=to_reduce[0])

        # Get the indices of each edge that we now know must be alone in its edge loop
        edge_is_alone_in_edge_loop_idx = edge_is_alone_in_edge_loop.nonzero()[0]
        # Converting to a list and iterating that instead of the ndarray is usually a bit faster
        edge_is_alone_in_edge_loop_idx_list = edge_is_alone_in_edge_loop_idx.tolist()

        # Remove the edges from the set of edges left to select
        edges_left_to_select.difference_update(edge_is_alone_in_edge_loop_idx_list)

        # Add the edge loops for these edges
        edge_loops.extend((edge_idx,) for edge_idx in edge_is_alone_in_edge_loop_idx_list)

        # We can get different edge loops if we start an edge loop from an edge in an ngon vs an edge outside an ngon.
        # For consistency, we will skip all ngon edges in the initial edge loop selection and do them last:
        #  ┌────┬────┐  ┌────┬────┐
        #  │    │    │  │    x    │  Selecting an edge loop starting from 'x' will select 'x' and 'a'
        #  ├────┼────┤  ├────┼────┤  Selecting an edge loop starting from 'a' will select 'a' and 'b'
        #  │    │    │  │    a    │  Selecting an edge loop starting from 'b' will select 'a' and 'b'
        #  │    ├────┤  │    ├────┤  Selecting an edge loop starting from 'y' will select 'b' and 'y'
        #  │    │    │  │    b    │
        #  ├────┼────┤  ├────┼────┤
        #  │    │    │  │    y    │
        #  └────┴────┘  └────┴────┘
        edges_in_ngons_set = set(edges_in_ngons.tolist())
        edges.foreach_set('hide', edge_is_alone_in_edge_loop)
        timings['pre edit mode done'] = perf_counter()
        # Enter EDIT mode and set up EDGE selection mode and clear the current selection
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_all(action="DESELECT")
        # Create bmesh from edit mesh
        bm = bmesh.from_edit_mesh(me)
        bm.select_mode = {'EDGE'}
        bm_edges = bm.edges
        # TODO: Can we hide all of the edges in edge_is_alone_in_edge_loop_idx_list so that we have a chance of creating
        #  more, smaller areas? (hide them in advance with numpy! (and unhide everything else))
        # Find areas that are not attached to each other, as we know there will be no edge loops that go between parts
        # of the mesh that aren't attached
        remaining_edge_idx_set = set(range(len(edges)))
        remaining_edge_idx_set.difference_update(edge_is_alone_in_edge_loop_idx_list)
        remaining_edges_set = {edge for edge in bm_edges if not edge.hide}
        linked_edges_dicts = []
        non_ngon_linked_edges_dicts = []
        # Ensure access by index is available
        bm_edges.ensure_lookup_table()
        timings['find regions start'] = perf_counter()
        while remaining_edge_idx_set:
            # Select an edge
            edge_idx = next(iter(remaining_edge_idx_set))
            bm_edges[edge_idx].select = True
            # Select all linked edges
            bpy.ops.mesh.select_linked()
            linked_edges_dict = {}
            non_ngon_linked_edges_dict = {}
            # Iterate the remaining edges and build dictionaries for the linked edges
            for edge in remaining_edges_set:
                if edge.select:
                    # TODO: It might be faster to bpy.ops.mesh.select_all(action='DESELECT')
                    edge.select = False
                    idx = edge.index
                    remaining_edge_idx_set.remove(idx)
                    # Skip all edges that we know will be alone
                    if idx in edges_left_to_select:
                        linked_edges_dict[idx] = edge
                        # We keep a separate dict for edges that aren't in ngons as we only want to start edge loops
                        # from ngon edges once we've started edge loops from all the other edges, otherwise the edge
                        # loops found may be inconsistent.
                        if idx not in edges_in_ngons_set:
                            non_ngon_linked_edges_dict[idx] = edge
            remaining_edges_set.difference_update(linked_edges_dict.values())
            if linked_edges_dict:
                linked_edges_dicts.append(linked_edges_dict)
                non_ngon_linked_edges_dicts.append(non_ngon_linked_edges_dict)
        print('found {} separate edge loop regions'.format(len(linked_edges_dicts)))
        timings['find loops start'] = perf_counter()
        zipped_edge_dicts = zip(linked_edges_dicts, non_ngon_linked_edges_dicts)
        # TODO: Try hiding all but the active region, maybe it will speed up bpy.ops.mesh.loop_multi_select(ring=False)
        for bm_edges_left_to_select_dict, non_ngon_bm_edges_left_to_select_dict in zipped_edge_dicts:
            reselected_edges = 0
            region_timings = {'start': perf_counter()}
            # select an edge we haven't select yet, select edge loop, remove all selected edges from the set of all edges
            # and add to the edge loops
            non_ngon_bm_edges_left_to_select = non_ngon_bm_edges_left_to_select_dict.values()
            bm_edges_left_to_select = bm_edges_left_to_select_dict.values()
            all_bm_edges_in_region = bm_edges_left_to_select_dict.copy().values()
            region_timings['initial loop start'] = perf_counter()
            # TODO: This loop is still by far the slowest part and for some reason, it gets slower when the mesh is
            #  bigger, even if the size of this region (and therefore the number of iterations) is the same???
            while non_ngon_bm_edges_left_to_select_dict:
                edge = next(iter(non_ngon_bm_edges_left_to_select))
                edge.select = True
                bpy.ops.mesh.loop_multi_select(ring=False)
                tentative_edge_loop = [edge for edge in bm_edges_left_to_select if edge.select]
                # TODO: Try and figure out if it's possible to reselect an edge, I haven't found it happen once
                # It might be possible to reselect an edge, I'm not sure
                all_edges_are_new = len(tentative_edge_loop) == me.total_edge_sel
                if all_edges_are_new:
                    edge_loop = tentative_edge_loop
                    edge_loop_i = tuple(edge.index for edge in edge_loop)
                    for idx in edge_loop_i:
                        del bm_edges_left_to_select_dict[idx]
                        if idx in non_ngon_bm_edges_left_to_select_dict:
                            del non_ngon_bm_edges_left_to_select_dict[idx]
                else:
                    reselected_edges += me.total_edge_sel - len(tentative_edge_loop)
                    edge_loop = [edge for edge in all_bm_edges_in_region if edge.select]
                    edge_loop_i = tuple(edge.index for edge in edge_loop)
                    for idx in edge_loop_i:
                        if idx in bm_edges_left_to_select_dict:
                            del bm_edges_left_to_select_dict[idx]
                        if idx in non_ngon_bm_edges_left_to_select_dict:
                            del non_ngon_bm_edges_left_to_select_dict[idx]
                edge_loops.append(edge_loop_i)
                # print("Found edge loop: {}".format(edge_loop_i))
                # And deselect the edges that were selected
                for edge_in_loop in edge_loop:
                    edge_in_loop.select = False
            # Now the only edges left are those in ngons. When selecting edge loops from these, we make extra checks that
            # each selected edge hasn't been selected before.
            # Ensure access by index is available
            region_timings['ngon loop start'] = perf_counter()
            while bm_edges_left_to_select_dict:
                edges_iter = iter(bm_edges_left_to_select_dict.values())
                start_edge = next(edges_iter)
                start_edge.select = True
                bpy.ops.mesh.loop_multi_select(ring=False)
                tentative_edge_loop = [bm_edge for bm_edge in edges_iter if bm_edge.select]
                tentative_edge_loop.append(start_edge)
                all_edges_are_new = len(tentative_edge_loop) == me.total_edge_sel
                # If we haven't reselected any edges, then we can add the edges as a full loop
                if all_edges_are_new:
                    edge_loop_i = tuple(edge.index for edge in tentative_edge_loop)
                    edge_loops.append(edge_loop_i)
                    # print("Found ngon originating edge loop: {}".format(edge_loop_i))
                    for idx in edge_loop_i:
                        del bm_edges_left_to_select_dict[idx]
                    for edge_in_loop in tentative_edge_loop:
                        edge_in_loop.select = False
                # Some edges in the edge loop are already part of another edge loop, so we'll break each of the newly
                # selected edges into loops on their own.
                else:
                    # We need to deselect all edges, we know the edges in tentative_edge_loop are selected, but there
                    # are other edges that have been selected that we don't know about
                    bpy.ops.mesh.select_all(action='DESELECT')
                    # print("Found ngon edge loop containing repeats originating from: {}".format(start_edge.index))
                    # Add each edge as a loop on its own
                    for edge in tentative_edge_loop:
                        idx = edge.index
                        edge_loop_i = (idx, )
                        edge_loops.append(edge_loop_i)
                        # print("Found edge loop from single ngon edge: {}".format(edge_loop_i))
                        if idx in bm_edges_left_to_select_dict:
                            del bm_edges_left_to_select_dict[idx]
            region_timings['finished'] = perf_counter()
            last_name = None
            last_seconds = None
            region_len = len(all_bm_edges_in_region)
            print('Region of length {}:'.format(region_len))
            for name, seconds in region_timings.items():
                if last_name is not None and last_seconds is not None:
                    print('\t{} took {} seconds from {}'.format(name, seconds - last_seconds, last_name))
                last_name = name
                last_seconds = seconds
            print('Took {} from start to finish'.format(region_timings['finished'] - region_timings['start']))
            if reselected_edges != 0:
                print('Warning, {} edges were reselected. Lots of reselected edges can slow down finding edge loops.'.format(reselected_edges))
        timings['edit mode finished'] = perf_counter()
        last_name = None
        last_seconds = None
        for name, seconds in timings.items():
            if last_name is not None and last_seconds is not None:
                print('{} took {} seconds from {}'.format(name, seconds - last_seconds, last_name))
            last_name = name
            last_seconds = seconds
        bpy.ops.object.mode_set(mode="OBJECT")
        return edge_loops

    def decimate(self, context):
        print('START DECIMATION')
        Common.set_default_stage()

        custom_decimation = context.scene.decimation_mode == 'CUSTOM'
        full_decimation = context.scene.decimation_mode == 'FULL'
        half_decimation = context.scene.decimation_mode == 'HALF'
        safe_decimation = context.scene.decimation_mode == 'SAFE'
        smart_decimation = context.scene.decimation_mode == 'SMART'
        loop_decimation = context.scene.decimation_mode == "LOOP"
        save_fingers = context.scene.decimate_fingers
        animation_weighting = context.scene.decimation_animation_weighting
        animation_weighting_factor = context.scene.decimation_animation_weighting_factor
        max_tris = context.scene.max_tris
        current_tris_count = 0
        tris_count = 0

        cats_basis_shape_key_name = "Cats Basis"

        meshes_obj = Common.get_meshes_objects(armature_name=self.armature_name)
        armature_obj = bpy.data.object.get(self.armature_name)
        if armature_obj.type != 'ARMATURE':
            armature_obj = None

        for mesh in meshes_obj:
            Common.set_active(mesh)
            if not loop_decimation:
                Common.switch('EDIT')
                # TODO: Does this do anything normally? Isn't the entire mesh possibly deselected at this point?
                # TODO: Add a raise RuntimeError in the original code and have a look. For now, added a select_all.
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
                Common.switch('OBJECT')
            else:
                Common.switch('EDIT')
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.tris_convert_to_quads()
                Common.switch('OBJECT')
            if context.scene.decimation_remove_doubles:
                Common.remove_doubles(mesh, 0.00001, save_shapes=True)
            current_tris_count += Common.get_tricount(mesh.data.polygons)
        #
        Common.unselect_all()

        if animation_weighting and not loop_decimation:
            AutoDecimateButton._create_cats_animation_vertex_groups(meshes_obj, armature_obj)

        if save_fingers:
            AutoDecimateButton._separate_fingers(meshes_obj)

        meshes = []
        for mesh in meshes_obj:
            if custom_decimation and mesh.name in ignore_meshes:
                continue

            tris = Common.get_tricount(mesh)

            if Common.has_shapekeys(mesh):
                if full_decimation:
                    Common.shape_key_remove_all(mesh)
                    meshes.append((mesh, tris))
                    tris_count += tris
                elif smart_decimation:
                    if len(mesh.data.shape_keys.key_blocks) == 1:
                        Common.shape_key_remove_all(mesh)
                    else:
                        # TODO: Do we care about this sanity check? It is just generally a good thing to always enforce.
                        # Sanity check, make sure basis isn't against something weird
                        basis_shape_key = mesh.data.shape_keys.key_blocks[0]
                        basis_shape_key.relative_key = basis_shape_key
                        # Add a duplicate basis key which we un-apply to fix shape keys
                        mesh.shape_key_add(name=cats_basis_shape_key_name, from_mix=False)
                        mesh.active_shape_key_index = 0
                    meshes.append((mesh, tris))
                    tris_count += tris
                elif loop_decimation:
                    meshes.append((mesh, tris))
                    tris_count += tris
                elif custom_decimation:
                    found = False
                    for shape in ignore_shapes:
                        if shape in mesh.data.shape_keys.key_blocks:
                            found = True
                            break
                    if found:
                        continue
                    Common.shape_key_remove_all(mesh)
                    meshes.append((mesh, tris))
                    tris_count += tris
                elif half_decimation and len(mesh.data.shape_keys.key_blocks) < 4:
                    Common.shape_key_remove_all(mesh)
                    meshes.append((mesh, tris))
                    tris_count += tris
                elif len(mesh.data.shape_keys.key_blocks) == 1:
                    Common.shape_key_remove_all(mesh)
                    meshes.append((mesh, tris))
                    tris_count += tris
            else:
                meshes.append((mesh, tris))
                tris_count += tris

        print(current_tris_count)
        print(tris_count)

        print((current_tris_count - tris_count), '>', max_tris)

        if (current_tris_count - tris_count) > max_tris:
            message = [t('decimate.cantDecimateWithSettings', number=str(max_tris))]
            if safe_decimation:
                message.append(t('decimate.safeTryOptions'))
            elif half_decimation:
                message.append(t('decimate.halfTryOptions'))
            elif custom_decimation:
                message.append(t('decimate.customTryOptions'))
            if save_fingers:
                if full_decimation or smart_decimation:
                    message.append(t('decimate.disableFingersOrIncrease'))
                else:
                    message[1] = message[1][:-1]
                    message.append(t('decimate.disableFingers'))
            Common.show_error(6, message)
            return

        try:
            decimation = (max_tris - current_tris_count + tris_count) / tris_count
        except ZeroDivisionError:
            decimation = 1
        if decimation >= 1:
            Common.show_error(6, [t('decimate.noDecimationNeeded', number=str(max_tris))])
            return
        elif decimation <= 0:
            Common.show_error(4.5, [t('decimate.cantDecimate1', number=str(max_tris)),
                                    t('decimate.cantDecimate2')])

        meshes.sort(key=lambda x: x[1])

        Common.unselect_all()
        for mesh in reversed(meshes):
            mesh_obj = mesh[0]
            tris = mesh[1]

            Common.set_active(mesh_obj)
            print(mesh_obj.name)

            # Calculate new decimation ratio
            try:
                decimation = (max_tris - current_tris_count + tris_count) / tris_count
            except ZeroDivisionError:
                decimation = 1
            print(decimation)

            # Apply decimation mod
            if not smart_decimation and not loop_decimation:
                # Original
                mod = mesh_obj.modifiers.new("Decimate", 'DECIMATE')
                mod.ratio = decimation
                mod.use_collapse_triangulate = True
                if animation_weighting:
                    mod.vertex_group = AutoDecimateButton.cats_animation_vg_name
                    mod.vertex_group_factor = animation_weighting_factor
                    mod.invert_vertex_group = True
                Common.apply_modifier(mod)
            elif not loop_decimation:
                # Smart
                Common.switch('EDIT')
                bpy.ops.mesh.select_mode(type="VERT")
                bpy.ops.mesh.select_all(action="SELECT")
                # TODO: Fix decimation calculation when pinning seams
                if self.preserve_seams:
                    bpy.ops.mesh.select_all(action="DESELECT")
                    bpy.ops.uv.seams_from_islands()

                    # select all seams
                    Common.switch('OBJECT')
                    me = mesh_obj.data
                    for edge in me.edges:
                        if edge.use_seam:
                            edge.select = True

                    Common.switch('EDIT')
                    bpy.ops.mesh.select_all(action="INVERT")

                bpy.ops.mesh.decimate(ratio=decimation,
                                      use_vertex_group=animation_weighting,
                                      vertex_group_factor=animation_weighting_factor,
                                      invert_vertex_group=True,
                                      use_symmetry=True,
                                      symmetry_axis='X')
                Common.switch('OBJECT')
            else:
                # Loop
                weights = self._get_loop_decimate_weights(context, mesh_obj)
                print(weights)
                print(len(weights))

                if animation_weighting:
                    newweights = self._get_animation_weighting(mesh_obj, armature_obj)
                    for idx, _ in newweights.items():
                        weights[idx] = max(weights[idx], newweights[idx])

                edge_loops = self._get_loop_decimate_edge_loops(mesh_obj)

                # from the new decimation vertex group, create a dict() of loops to sum of shape-importance (loops which contain texture edges put at the end)
                # TODO: order needs to be usual -> texture boundaries -> mesh boundaries
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.uv.seams_from_islands()
                bpy.ops.object.mode_set(mode="OBJECT")
                edge_loops_weighted = [l for l in sorted([
                                         (max(weights[mesh_obj.data.edges[edge].vertices[0]] +
                                              weights[mesh_obj.data.edges[edge].vertices[1]]
                                              for edge in edge_loop),
                                         edge_loop)
                                         for edge_loop in edge_loops
                                         if not any(mesh_obj.data.edges[edge].use_seam for edge in edge_loop)
                                      ], key=lambda v: v[0])]
                edge_loops_weighted+= [l for l in sorted([
                                         (max(weights[mesh_obj.data.edges[edge].vertices[0]] +
                                              weights[mesh_obj.data.edges[edge].vertices[1]]
                                              for edge in edge_loop),
                                         edge_loop)
                                         for edge_loop in edge_loops
                                         if any(mesh_obj.data.edges[edge].use_seam for edge in edge_loop)
                                      ], key=lambda v: v[0])]
                # TODO: Meshes bordering the edge should be lowest decimatability
                print(edge_loops_weighted)

                # dissolve from the bottom up until target decimation is met
                selected_edges = set()
                while len(selected_edges) <= ((1-decimation) * Common.get_tricount(mesh_obj)/2):
                    loop = edge_loops_weighted.pop()
                    selected_edges.update(loop[1])
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_mode(type='EDGE')
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")
                for edge in selected_edges:
                    mesh_obj.data.edges[edge].select = True
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.delete_edgeloop()
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")

            # Needed? This has been moved up from below the smart_decimation repair which no longer selects objects
            Common.unselect_all()

            tris_after = len(mesh_obj.data.polygons)
            print(tris)
            print(tris_after)

            current_tris_count = current_tris_count - tris + tris_after
            tris_count = tris_count - tris
            # Repair shape keys if SMART mode is enabled
            if smart_decimation and Common.has_shapekeys(mesh_obj):
                key_idx = mesh_obj.data.shape_keys.key_blocks.find(cats_basis_shape_key_name)
                cats_basis_shape_key = mesh_obj.data.shape_keys.key_blocks[key_idx]
                cats_basis_shape_key.slider_min = -1
                cats_basis_shape_key.value = -1
                orig_key_index = mesh_obj.active_shape_key_index
                mesh_obj.active_shape_key_index = key_idx
                bpy.ops.cats_shapekey({'object': mesh_obj})
                # Note that after applying the operator, the cats_basis_shape_key's name will have changed
                mesh_obj.shape_key_remove(cats_basis_shape_key)
                mesh_obj.active_shape_key_index = orig_key_index

        # # Check if decimated correctly
        # if decimation < 0:
        #     print('')
        #     print('RECHECK!')
        #
        #     current_tris_count = 0
        #     tris_count = 0
        #
        #     for mesh in Common.get_meshes_objects():
        #         Common.select(mesh)
        #         tris = len(bpy.context.active_object.data.polygons)
        #         tris_count += tris
        #         print(tris_count)
        #
        #     for mesh in reversed(meshes):
        #         mesh_obj = mesh[0]
        #         Common.select(mesh_obj)
        #
        #         # Calculate new decimation ratio
        #         decimation = (max_tris - tris_count) / tris_count
        #         print(decimation)
        #
        #         # Apply decimation mod
        #         mod = mesh_obj.modifiers.new("Decimate", 'DECIMATE')
        #         mod.ratio = decimation
        #         mod.use_collapse_triangulate = True
        #         Common.apply_modifier(mod)
        #
        #         Common.unselect_all()
        #         break


@register_wrap
class AutoDecimatePresetGood(bpy.types.Operator):
    bl_idname = 'cats_decimation.preset_good'
    bl_label = t('DecimationPanel.preset.good.label')
    bl_description = t('DecimationPanel.preset.good.description')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        bpy.context.scene.max_tris = 70000
        return {'FINISHED'}


@register_wrap
class AutoDecimatePresetExcellent(bpy.types.Operator):
    bl_idname = 'cats_decimation.preset_excellent'
    bl_label = t('DecimationPanel.preset.excellent.label')
    bl_description = t('DecimationPanel.preset.excellent.description')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        bpy.context.scene.max_tris = 32000
        return {'FINISHED'}


@register_wrap
class AutoDecimatePresetQuest(bpy.types.Operator):
    bl_idname = 'cats_decimation.preset_quest'
    bl_label = t('DecimationPanel.preset.quest.label')
    bl_description = t('DecimationPanel.preset.quest.description')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        bpy.context.scene.max_tris = 5000
        return {'FINISHED'}
