# GPL License

import bpy
from bpy.types import (
    Context,
    Operator,
    Object,
    Mesh,
    DecimateModifier,
)
from bpy.props import (
    BoolProperty,
    StringProperty,
    IntProperty,
    EnumProperty,
)

import numpy as np
from typing import Optional, Literal

from . import common as Common
from . import armature_bones as Bones
from .register import register_wrap
from .translations import t

_DecimationMode = Literal['SAFE', 'HALF', 'FULL', 'CUSTOM']
_FINGER_VG_NAMES = {finger_side for finger in Bones.bone_finger_list for finger_side in (finger + "L", finger + "R")}

# TODO: Make these properties. Can add them to a property group on the WindowManager to make sure they don't get saved
#  in the .blend.
ignore_shapes: list[str] = []
ignore_meshes: list[str] = []


def _calc_vertex_selection_triangles(me: Mesh, vertex_selection_mask: np.ndarray) -> int:
    """Calculate the number of triangles in a vertex selection mask.

    This function requires that a mesh's polygons and loops are in the same order. This is forced as of Blender 3.6.
    Prior to Blender 3.6, having polygons and loops in different orders is supposed to be supported by Blender, but is
    extremely rare to come across (it should only be 3rd-party scripts that create such meshes and entering and exiting
    Edit mode sets polygons and loops into the same order) and has visual bugs.

    This is an optimized, math heavy function."""
    if bpy.app.version >= (3, 6):
        loop_vertex_index = np.empty(len(me.loops), dtype=np.intc)
        me.attributes[".corner_vert"].data.foreach_get("value", loop_vertex_index)
    else:
        loop_vertex_index = np.empty(len(me.loops), dtype=np.uintc)
        me.loops.foreach_get("vertex_index", loop_vertex_index)

    # For a mesh made of 1 quad and 1 triangle, where one vert of the quad is not selected:
    # [T, F, T, T, T, T, T] (Quad: [T, F, T, T], Tri: [T, T, T])
    loop_selection_mask = vertex_selection_mask[loop_vertex_index]
    # [0, 4]
    loop_starts = np.empty(len(me.polygons), dtype=np.uintc)
    me.polygons.foreach_get("loop_start", loop_starts)
    # Equivalent to "loop_total".
    # Equivalent to the number of sides or corners of the polygon.
    # [4, 3]
    num_loops_per_polygon = np.diff(loop_starts, append=len(me.loops))

    # Start by performing a cumulative sum across the loop_selection_mask (where False is 0 and True is 1).
    # If all of a polygon's loops are selected, the difference in the cumulative sum from the end of the current polygon
    # to the end of the previous polygon will equal the differences between the start (or end) indices of each polygon
    # (which is the same as the number of sides/loops (corners) of the polygon.
    # Then, using the mask of polygons whose loops are all selected, get the number of loops of all selected polygons.
    # Subtracting 2 from each number-of-loops gets the number of triangles that each of the polygons is made of.
    # Then return the sum of the numbers of triangles.

    # [T, F, T, T, T, T, T]
    # [1, 0, 1, 1, 1, 1, 1] -> [1, 1, 2, 3, 4, 5, 6] (Quad: [1, 1, 2, 3], Tri: [4, 5, 6])
    c_sum = np.cumsum(loop_selection_mask)

    # Get the index of the last loop of each polygon.
    # The index of the last loop of each polygon is the loop start of the next polygon minus 1.
    # For the very last polygon, it's last index will be the number of loops minus 1.
    # This is barely faster than:
    # `loop_ends = loop_starts + num_loops_per_polygon - 1`
    # [e0, e1]
    loop_ends = np.empty_like(loop_starts)
    # [e0, e1][:-1] = [e0] -> [0, 4][1:] - 1 = [4] - 1 = [3]
    # [e0, e1][-1:] = [e1] -> 7 - 1 = 6
    # [3, 6]
    loop_ends[:-1] = loop_starts[1:] - 1
    loop_ends[-1:] = len(me.loops) - 1

    # Get the values of the cumulative sum at the end of each polygon.
    # [1, 1, 2, 3, 4, 5, 6][[2, 6]] -> [3, 6]
    indexed = c_sum[loop_ends]

    # Prepend 0 and calculate the difference between the value for each polygon and its previous polygon.
    # [3, 6] -> [0, 3, 6] -> [3-0, 6-3] -> [3, 3]
    num_selected_loops_per_polygon = np.diff(indexed, prepend=0)

    # Wherever the differences equal the number of loops (corners)/sides per polygon, that polygon would be selected.
    # [3, 3] == [4, 3] -> [F, T]
    polygon_selection_mask = num_selected_loops_per_polygon == num_loops_per_polygon

    # The number of triangles of a polygon is equal to its number of loops (corners) or sides minus 2.
    # Index num_loops_per_polygon by the polygon_selection_mask to get an array of the number of loops of all selected
    # polygons.
    # Subtract 2 from every element and sum up the entire array to get the number of selected triangles.
    # sum([4, 3][[False, True]] - 2) -> sum([3] - 2) -> sum([1]) -> 1
    selected_tris = np.sum(num_loops_per_polygon[polygon_selection_mask] - 2)

    return selected_tris


def _verts_in_groups_gen(mesh: Mesh, vg_idx_set: set[int]) -> bool:
    for v in mesh.vertices:
        for g in v.groups:
            if g.group in vg_idx_set:
                # TODO: We might want to check that the weight is greater than zero, but it will make this loop slower.
                yield True
                break
        else:
            yield False


_ALL_FINGERS = -1


def _save_fingers_prep(mesh_obj: Object) -> tuple[int, Optional[np.ndarray]]:
    """Given a mesh Object, find the vertices belonging to fingers and return the tri count of the fingers and a bool
    ndarray that is a mask of the vertices belonging to fingers."""
    finger_vg_indices = set()
    for i, vg in enumerate(mesh_obj.vertex_groups):
        if vg.name in _FINGER_VG_NAMES:
            finger_vg_indices.add(i)

    if not finger_vg_indices:
        return 0, None

    mesh: Mesh = mesh_obj.data
    # Iterate through vertices and produce a bool np.ndarray of vertices in at least one 'fingers' vertex
    # group.
    #  The advantage of having a numpy bool array is that we should be able to use it to figure out how many
    #  polygons would be selected if we selected those vertices and use that to figure out how many polygons
    #  will actually be available for decimation.
    finger_selection = np.fromiter(_verts_in_groups_gen(mesh, finger_vg_indices), dtype=bool, count=len(mesh.vertices))
    if not finger_selection.any():
        return 0, None
    if finger_selection.all():
        return _ALL_FINGERS, None
    # Before actually adding the vertex group, figure out how many polygons the finger selection is and
    # modify the tricount by those which are part of fingers.
    finger_tris = _calc_vertex_selection_triangles(mesh, finger_selection)
    return finger_tris, finger_selection


def _do_save_fingers(mesh_objects_to_decimate: list[tuple[Object, int]],
                     decimation_mode: _DecimationMode,
                     total_start_tris: int,
                     max_tris: int,
                     ) -> tuple[bool, list[Object], dict[Object: str]]:
    mesh_objects_and_finger_selections: list[tuple[Object, Optional[np.ndarray]]] = []
    updated_decimatable_tris_count = 0
    for mesh_obj, entire_mesh_tri_count in mesh_objects_to_decimate:
        finger_tris, finger_selection = _save_fingers_prep(mesh_obj)
        if finger_tris == 0:
            updated_decimatable_tris_count += entire_mesh_tri_count
            mesh_objects_and_finger_selections.append((mesh_obj, None))
        elif finger_tris == _ALL_FINGERS:
            # The entire mesh is fingers, it can't be decimated, so don't add it to the new list.
            continue
        else:
            # Fingers will not be included in the decimation, so calculate the new tris count for this mesh that
            # can be decimated
            updated_decimatable_tris_count += (entire_mesh_tri_count - finger_tris)
            mesh_objects_and_finger_selections.append((mesh_obj, finger_selection))

    # Re-check that the model can be decimated to the desired triangle count now that we figured out how many
    # tris the fingers are.
    updated_min_tris_count = total_start_tris - updated_decimatable_tris_count

    print(f"After Save Fingers decimation check: total model tris: {total_start_tris}")
    print(f"After Save Fingers decimation check: target max tris: {max_tris}")
    print(f"After Save Fingers decimation check: total decimatable tris: {updated_decimatable_tris_count}")
    print(f"After Save Fingers decimation check: minimum tris after decimation: {updated_min_tris_count}")

    if updated_min_tris_count > max_tris:
        message = [t('decimate.cantDecimateWithSettings', number=str(max_tris))]
        if decimation_mode == 'SAFE':
            message.append(t('decimate.safeTryOptions'))
        elif decimation_mode == 'HALF':
            message.append(t('decimate.halfTryOptions'))
        elif decimation_mode == 'CUSTOM':
            message.append(t('decimate.customTryOptions'))

        if decimation_mode == 'FULL':
            message.append(t('decimate.disableFingersOrIncrease'))
        else:
            message[1] = message[1][:-1]
            message.append(t('decimate.disableFingers'))
        Common.show_error(6, message)
        return False, [], {}

    # Now create the final list
    # Now add the vertex group to each mesh_obj that has fingers saved. This vertex group will be set in the
    # Object's Decimate modifier.
    saved_fingers_vertex_groups: dict[Object, str] = {}
    updated_mesh_objects_to_decimate = []
    for mesh_obj, finger_selection in mesh_objects_and_finger_selections:
        if finger_selection is None:
            updated_mesh_objects_to_decimate.append(mesh_obj)
            continue
        save_fingers_vg = mesh_obj.vertex_groups.new(name="Cats Save Fingers")
        saved_fingers_vertex_groups[mesh_obj] = save_fingers_vg.name
        save_fingers_vg.add(np.flatnonzero(finger_selection).data, 1.0, 'REPLACE')
        updated_mesh_objects_to_decimate.append(mesh_obj)

    return True, updated_mesh_objects_to_decimate, saved_fingers_vertex_groups


def _apply_decimate_mod(context: Context,
                        decimation_ratio: float,
                        mesh_obj: Object,
                        finger_saver_vg_name: str,
                        triangulate: bool):
    # Modifiers can't be applied to multi-user data.
    # As of Blender 3.2, we could set single_user=True in the modifier_apply operator call instead.
    mesh = mesh_obj.data
    if mesh.users > 1:
        # Set the mesh of the mesh_obj to a copy, guaranteeing that it is not multi-user data.
        mesh_obj.data = mesh.copy()

    # Remove all shape keys. The Decimate modifier cannot be applied to meshes with shape keys.
    mesh_obj.shape_key_clear()

    # Create and set up the Decimate modifier.
    modifiers = mesh_obj.modifiers
    decimate_modifier: DecimateModifier = modifiers.new(name="Cats Decimation", type='DECIMATE')
    if finger_saver_vg_name:
        decimate_modifier.vertex_group = finger_saver_vg_name
        # The vertex group has vertices that should not be decimated assigned to 1.0 in the vertex group.
        # All other vertices are not in the vertex group.
        decimate_modifier.invert_vertex_group = True
    decimate_modifier.ratio = decimation_ratio
    decimate_modifier.use_collapse_triangulate = triangulate

    # Create a temporary context override so that we can apply modifiers to specific Objects without making them active.
    with Common.temp_context_override(context, object=mesh_obj) as override:
        # Modifiers can't be applied to multi-user data.
        # As of Blender 3.2, we could set single_user=True in the modifier_apply operator call instead.
        mesh = mesh_obj.data
        if mesh.users > 1:
            # Set the mesh of the mesh_obj to a copy, guaranteeing that it is not multi-user data.
            mesh_obj.data = mesh.copy()

        # Move the modifier to the top before applying it to suppress a warning printed to the console about the
        # modifier not being at the top and potentially having unexpected results.
        if bpy.app.version >= (3, 5):
            modifiers.move(len(modifiers) - 1, 0)
        else:
            bpy.ops.object.modifier_move_to_index(*override, modifier=decimate_modifier.name)

        # Apply the Decimate modifier
        bpy.ops.object.modifier_apply(*override, modifier=decimate_modifier.name)

    # Clean up by removing the finger-saver vertex group.
    if finger_saver_vg_name:
        vertex_groups = mesh_obj.vertex_groups
        vertex_groups.remove(vertex_groups[finger_saver_vg_name])


@register_wrap
class ScanButton(Operator):
    bl_idname = 'cats_decimation.auto_scan'
    bl_label = t('ScanButton.label')
    bl_description = t('ScanButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return Common.is_enum_non_empty(context.scene.add_shape_key)

    def execute(self, context):
        scene = context.scene
        shape = scene.add_shape_key
        shapes = Common.get_shapekeys_decimation_list(self, context)
        count = len(shapes)

        if count > 1 and shapes.index(shape) == count - 1:
            scene.add_shape_key = shapes[count - 2]

        ignore_shapes.append(shape)

        return {'FINISHED'}


@register_wrap
class AddShapeButton(Operator):
    bl_idname = 'cats_decimation.add_shape'
    bl_label = t('AddShapeButton.label')
    bl_description = t('AddShapeButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return Common.is_enum_non_empty(context.scene.add_shape_key)

    def execute(self, context):
        scene = context.scene
        shape = scene.add_shape_key
        shapes = [x[0] for x in Common.get_shapekeys_decimation_list(self, context)]
        count = len(shapes)

        if count > 1 and shapes.index(shape) == count - 1:
            scene.add_shape_key = shapes[count - 2]

        ignore_shapes.append(shape)

        return {'FINISHED'}


@register_wrap
class AddMeshButton(Operator):
    bl_idname = 'cats_decimation.add_mesh'
    bl_label = t('AddMeshButton.label')
    bl_description = t('AddMeshButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return Common.is_enum_non_empty(context.scene.add_mesh)

    def execute(self, context):
        scene = context.scene
        ignore_meshes.append(scene.add_mesh)

        ignore_meshes_set = set(ignore_meshes)
        armature = scene.armature
        if Common.is_enum_non_empty(armature):
            for obj in scene.objects:
                if (
                        obj.name not in ignore_meshes_set
                        and obj.type == 'MESH'
                        and (parent := obj.parent)
                        and parent.type == 'ARMATURE'
                        and parent.name == armature
                ):
                    context.scene.add_mesh = obj.name
                    break

        return {'FINISHED'}


@register_wrap
class RemoveShapeButton(Operator):
    bl_idname = 'cats_decimation.remove_shape'
    bl_label = t('RemoveShapeButton.label')
    bl_description = t('RemoveShapeButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    shape_name: bpy.props.StringProperty()

    def execute(self, context):
        ignore_shapes.remove(self.shape_name)

        return {'FINISHED'}


@register_wrap
class RemoveMeshButton(Operator):
    bl_idname = 'cats_decimation.remove_mesh'
    bl_label = t('RemoveMeshButton.label')
    bl_description = t('RemoveMeshButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    mesh_name: bpy.props.StringProperty()

    def execute(self, context):
        ignore_meshes.remove(self.mesh_name)

        return {'FINISHED'}


@register_wrap
class AutoDecimateButton(Operator):
    bl_idname = 'cats_decimation.auto_decimate'
    bl_label = t('AutoDecimateButton.label')
    bl_description = t('AutoDecimateButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    armature_name: StringProperty(
        name=t("AutoDecimateButton.armature.label"),
        description=t("AutoDecimateButton.armature.desc"),
    )

    decimation_mode: EnumProperty(
        name=t("Scene.decimation_mode.label"),
        description=t("Scene.decimation_mode.desc"),
        items=[
            ('SAFE', t("Scene.decimation_mode.safe.label"), t("Scene.decimation_mode.safe.desc")),
            ('HALF', t("Scene.decimation_mode.half.label"), t("Scene.decimation_mode.half.desc")),
            ('FULL', t("Scene.decimation_mode.full.label"), t("Scene.decimation_mode.full.desc")),
            ('CUSTOM', t("Scene.decimation_mode.custom.label"), t("Scene.decimation_mode.custom.desc"))
        ],
        default='SAFE',
    )

    save_fingers: BoolProperty(
        name=t("Scene.decimate_fingers.label"),
        description=t("Scene.decimate_fingers.desc"),
        default=False,
    )

    triangulate: BoolProperty(
        name=t("AutoDecimateButton.triangulate.label"),
        description=t("AutoDecimateButton.triangulate.desc"),
        default=True,
    )

    max_tris: IntProperty(
        name=t("Scene.max_tris.label"),
        description=t("Scene.max_tris.desc"),
        min=1,
        soft_max=500_000,
        default=70_000,  # More than 70_000 is considered "Very Poor" by VRChat
    )

    @classmethod
    def poll(cls, context: Context) -> bool:
        if context.mode.startswith("EDIT"):
            cls.poll_message_set("Must not be in Edit mode")
            return False

    def execute(self, context) -> set[str]:
        armature_name = self.armature_name

        if armature_name not in bpy.data.objects:
            self.report({'ERROR'}, t('AutoDecimateButton.error.noArmature'))
            return {'FINISHED'}

        meshes = Common.get_meshes_objects(self.armature_name)
        if not meshes or len(meshes) == 0:
            self.report({'ERROR'}, t('AutoDecimateButton.error.noMesh'))
            return {'FINISHED'}

        self.decimate(context)

        return {'FINISHED'}

    def decimate(self, context: Context):
        print('START DECIMATION')

        save_fingers = self.save_fingers
        max_tris = self.max_tris
        decimation_mode = self.decimation_mode
        triangulate = self.triangulate

        meshes_obj = Common.get_meshes_objects(armature_name=self.armature_name)

        if not meshes_obj:
            # There are no meshes in the first-place, so we're done.
            return True

        # Start by filtering out meshes based on decimation_mode and doing a first check on the total tris.
        total_start_tris = 0
        total_to_decimate_tris = 0
        ignore_shapes_set = set(ignore_shapes)
        ignore_meshes_set = set(ignore_meshes)
        mesh_objects_to_decimate: list[tuple[Object, int]] = []
        for mesh_obj in meshes_obj:
            tri_count = Common.get_tricount(mesh_obj)
            total_start_tris += tri_count

            if decimation_mode == 'CUSTOM':
                # Meshes in ignore_meshes and meshes with a shape key in ignore_shapes will be excluded.
                if mesh_obj.name in ignore_meshes_set:
                    continue
                if Common.has_shapekeys(mesh_obj):
                    for shape in mesh_obj.data.shape_keys.key_blocks:
                        if shape.name in ignore_shapes_set:
                            continue
            elif decimation_mode == 'HALF':
                # Meshes with 4 of more shape keys will be excluded
                if Common.has_shapekeys(mesh_obj) and len(mesh_obj.data.shape_keys.key_blocks) > 3:
                    continue
            elif decimation_mode == 'SAFE':
                # Meshes with shape keys will be excluded
                if Common.has_shapekeys(mesh_obj) and len(mesh_obj.data.shape_keys.key_blocks) > 1:
                    continue
            elif decimation_mode != 'FULL':
                # All meshes will be included and will have their shape keys removed.
                pass
            else:
                raise RuntimeError(f"Unexpected decimation mode '{decimation_mode}'")

            total_to_decimate_tris += tri_count
            mesh_objects_to_decimate.append((mesh_obj, tri_count))

        print(f"Initial decimation check: total model tris: {total_start_tris}")
        print(f"Initial decimation check: target max tris: {max_tris}")
        if total_start_tris <= max_tris:
            Common.show_error(6, [t('decimate.noDecimationNeeded', number=str(max_tris))])
            return True

        print(f"Initial decimation check: total decimatable tris: {total_to_decimate_tris}")

        # The tris remaining if every mesh being decimated is completely deleted.
        # This count is before we further increase the minimum number of tris due to the save_fingers option.
        initial_min_tris = total_start_tris - total_to_decimate_tris

        print(f"Initial decimation check: minimum tris after decimation: {initial_min_tris}")

        if initial_min_tris > max_tris:
            message = [t('decimate.cantDecimateWithSettings', number=str(max_tris))]
            if decimation_mode == 'SAFE':
                message.append(t('decimate.safeTryOptions'))
            elif decimation_mode == 'HALF':
                message.append(t('decimate.halfTryOptions'))
            elif decimation_mode == 'CUSTOM':
                message.append(t('decimate.customTryOptions'))
            elif decimation_mode == 'FULL':
                pass
            Common.show_error(6, message)
            return False

        if save_fingers:
            decimation_possible, to_decimate, finger_saver_vertex_groups = _do_save_fingers(
                mesh_objects_to_decimate, decimation_mode, total_start_tris, max_tris
            )
            if not decimation_possible:
                return False
        else:
            to_decimate = [mesh_obj for mesh_obj, *_ in mesh_objects_to_decimate]
            finger_saver_vertex_groups = {}

        # Calculate the decimation ratio. This is not affected by the save_fingers option.
        decimation_ratio = (max_tris - total_start_tris)/total_to_decimate_tris + 1

        # Now add the Decimate modifier and apply it, deleting shape keys in the process.
        for mesh_obj in to_decimate:
            finger_saver_vg_name = finger_saver_vertex_groups.get(mesh_obj)
            _apply_decimate_mod(context, decimation_ratio, mesh_obj, finger_saver_vg_name, triangulate)

        # Decimation successful!
        return True


@register_wrap
class AutoDecimatePresetGood(Operator):
    bl_idname = 'cats_decimation.preset_good'
    bl_label = t('DecimationPanel.preset.good.label')
    bl_description = t('DecimationPanel.preset.good.description')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        context.scene.max_tris = 70000
        return {'FINISHED'}


@register_wrap
class AutoDecimatePresetExcellent(Operator):
    bl_idname = 'cats_decimation.preset_excellent'
    bl_label = t('DecimationPanel.preset.excellent.label')
    bl_description = t('DecimationPanel.preset.excellent.description')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        context.scene.max_tris = 32000
        return {'FINISHED'}


@register_wrap
class AutoDecimatePresetQuest(Operator):
    bl_idname = 'cats_decimation.preset_quest'
    bl_label = t('DecimationPanel.preset.quest.label')
    bl_description = t('DecimationPanel.preset.quest.description')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        context.scene.max_tris = 5000
        return {'FINISHED'}
