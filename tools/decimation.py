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
    def get_animation_weighting(mesh, armature=None):
        bone_weights = dict()

        def calc_weight_pairs(v_idx, groups):
            # This gives a time complexity of O((m^2 - m)/2), which is still the same class as O(m^2) for iterating
            # groups in its entirety in two loops, but will result in just under half the number of iterations.
            #
            # We want to skip the case when w1 == w2, and we know that (w1, w2) will result in the same values as
            # (w2, w1), so we want to skip those as well.
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
                    # w1_group and w2_group can be in any order so order them when making the key
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
        meshes = []
        current_tris_count = 0
        tris_count = 0

        cats_animation_vg_name = "CATS Animation"
        cats_basis_shape_key_name = "Cats Basis"

        meshes_obj = Common.get_meshes_objects(armature_name=self.armature_name)
        armature_obj = bpy.data.object.get(self.armature_name)
        if armature_obj.type != 'ARMATURE':
            armature_obj = None

        for mesh in meshes_obj:
            Common.set_active(mesh)
            if not loop_decimation:
                Common.switch('EDIT')
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

        if animation_weighting and not loop_decimation:
            for mesh in meshes_obj:
                newweights = self.get_animation_weighting(mesh, armature_obj)

                # Vertices weight painted to a single bone are likely to end up with the same newweight
                # Mirrored meshes are likely to have the same newweight for each pair of mirrored vertices
                # All vertices with the same weight can be updated at the same time, so we'll flip the keys and values
                # to get all the vertices which need to be set to each unique weight
                weight_to_indices = {}
                for idx, weight in newweights.items():
                    vertex_indices = weight_to_indices.setdefault(weight, [])
                    vertex_indices.append(idx)

                cats_animation_vg = mesh.vertex_groups.new(name=cats_animation_vg_name)

                for weight, vertex_indices in weight_to_indices.items():
                    cats_animation_vg.add(vertex_indices, weight, "REPLACE")

        if save_fingers:
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
            # And unselect all the mesh objects
            Common.unselect_all()

        for mesh in meshes_obj:
            Common.set_active(mesh)
            tris = Common.get_tricount(mesh)

            if custom_decimation and mesh.name in ignore_meshes:
                Common.unselect_all()
                continue

            if Common.has_shapekeys(mesh):
                if full_decimation:
                    bpy.ops.object.shape_key_remove(all=True)
                    meshes.append((mesh, tris))
                    tris_count += tris
                elif smart_decimation:
                    if len(mesh.data.shape_keys.key_blocks) == 1:
                        bpy.ops.object.shape_key_remove(all=True)
                    else:
                        mesh.active_shape_key_index = 0
                        # Sanity check, make sure basis isn't against something weird
                        mesh.active_shape_key.relative_key = mesh.active_shape_key
                        # Add a duplicate basis key which we un-apply to fix shape keys
                        bpy.ops.object.shape_key_add(from_mix=False)
                        mesh.active_shape_key.name = cats_basis_shape_key_name
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
                        Common.unselect_all()
                        continue
                    bpy.ops.object.shape_key_remove(all=True)
                    meshes.append((mesh, tris))
                    tris_count += tris
                elif half_decimation and len(mesh.data.shape_keys.key_blocks) < 4:
                    bpy.ops.object.shape_key_remove(all=True)
                    meshes.append((mesh, tris))
                    tris_count += tris
                elif len(mesh.data.shape_keys.key_blocks) == 1:
                    bpy.ops.object.shape_key_remove(all=True)
                    meshes.append((mesh, tris))
                    tris_count += tris
            else:
                meshes.append((mesh, tris))
                tris_count += tris

            Common.unselect_all()

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
                    mod.vertex_group = cats_animation_vg_name
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
                depsgraph = context.evaluated_depsgraph_get()

                # create a dict() of vert cordinate to index
                mod = mesh_obj.modifiers.new("Decimate", 'DECIMATE')
                mod.use_symmetry = True
                mod.symmetry_axis = 'X'
                # Dark magic... encode the vert index into the 'red' channel of a vertex color map
                # col is a float in range [0.0, 1.0], so we can encode the idx into the lower 23b
                mesh_obj.data.vertex_colors.new(name='CATS Vert', do_init=False)
                for vertex in mesh_obj.data.vertices:
                    mesh_obj.data.vertex_colors['CATS Vert'].data[vertex.index].color[0] = struct.unpack('f', struct.pack('I', vertex.index))[0]

                # decimate N times, n/N% each time, and observe the result to get a list of leftover verts (the ones decimated)
                iterations = 100
                weights = dict()

                for i in range(1, iterations):
                    mod.ratio = (i/iterations)
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.object.mode_set(mode="OBJECT")
                    mesh_decimated = mesh_obj.evaluated_get(depsgraph)
                    for vert in mesh_decimated.data.vertices:
                        idx = struct.unpack('I', struct.pack('f',
                                mesh_obj.data.vertex_colors['CATS Vert'].data[vert.index].color[0]))[0]
                        if not idx in weights:
                            weights[idx] = 1 - (i/iterations)
                for i in range(0,len(mesh_obj.data.vertices)):
                    if not i in weights:
                        weights[i] = 0.0

                print(weights)
                print(len(weights))

                if animation_weighting:
                    newweights = self.get_animation_weighting(mesh_obj, armature_obj)
                    for idx, _ in newweights.items():
                        weights[idx] = max(weights[idx], newweights[idx])

                all_edges = set(edge.index for edge in mesh_obj.data.edges[:])
                edge_loops = []

                # pop one edge out, select it, select edge loop, remove all selected edges from the set of all edges and add to the edge loops
                # ugly ugly, and very slow (though scalable)
                while(len(all_edges) > 0):
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.select_mode(type='EDGE')
                    bpy.ops.mesh.select_all(action="DESELECT")
                    bpy.ops.object.mode_set(mode="OBJECT")
                    selected_edge = next(edge for edge in all_edges)
                    mesh_obj.data.edges[selected_edge].select = True
                    bpy.ops.object.mode_set(mode="OBJECT")
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.loop_multi_select(ring=False)
                    bpy.ops.object.mode_set(mode="OBJECT")
                    edge_loop = set(edge.index for edge in mesh_obj.data.edges if edge.select)
                    edge_loop.add(selected_edge)
                    edge_loops.append(edge_loop)
                    print("Found edge loop: {}".format(edge_loop))
                    all_edges.difference_update(edge_loop)
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

            tris_after = len(mesh_obj.data.polygons)
            print(tris)
            print(tris_after)

            current_tris_count = current_tris_count - tris + tris_after
            tris_count = tris_count - tris
            # Repair shape keys if SMART mode is enabled
            if smart_decimation and Common.has_shapekeys(mesh_obj):
                for idx in range(1, len(mesh_obj.data.shape_keys.key_blocks) - 1):
                    mesh_obj.active_shape_key_index = idx
                    Common.switch('EDIT')
                    bpy.ops.mesh.blend_from_shape(shape=cats_basis_shape_key_name, blend=-1.0, add=True)
                    Common.switch('OBJECT')
                mesh_obj.shape_key_remove(key=mesh_obj.data.shape_keys.key_blocks[cats_basis_shape_key_name])
                mesh_obj.active_shape_key_index = 0

            Common.unselect_all()

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
