# GPL License

import bpy
from bpy.types import (
    Context,
    Mesh,
)

from . import common as Common
from .register import register_wrap

from .translations import t


SHAPE_A = 'a'
SHAPE_O = 'o'
SHAPE_CH = 'ch'

# Set up the shape keys.
SHAPE_KEY_DATA = {
    'vrc.v_aa': {SHAPE_A: 1.0},
    'vrc.v_ch': {SHAPE_CH: 1.0},
    'vrc.v_dd': {SHAPE_A: 0.3, SHAPE_CH: 0.7},
    # Note 2022-07-16: 'ih' and 'e' were originally swapped, due to early VRChat viseme
    # refs being slightly incorrect. See:
    # https://github.com/absolute-quantum/cats-blender-plugin/issues/505
    # https://developer.oculus.com/documentation/unreal/audio-ovrlipsync-viseme-reference/
    'vrc.v_ih': {SHAPE_CH: 0.7, SHAPE_O: 0.3},
    'vrc.v_ff': {SHAPE_A: 0.2, SHAPE_CH: 0.4},
    'vrc.v_e': {SHAPE_A: 0.5, SHAPE_CH: 0.2},
    'vrc.v_kk': {SHAPE_A: 0.7, SHAPE_CH: 0.4},
    'vrc.v_nn': {SHAPE_A: 0.2, SHAPE_CH: 0.7},
    'vrc.v_oh': {SHAPE_A: 0.2, SHAPE_O: 0.8},
    'vrc.v_ou': {SHAPE_O: 1.0},
    'vrc.v_pp': {},
    'vrc.v_rr': {SHAPE_CH: 0.5, SHAPE_O: 0.3},
    'vrc.v_sil': {},
    'vrc.v_ss': {SHAPE_CH: 0.8},
    'vrc.v_th': {SHAPE_A: 0.4, SHAPE_O: 0.15},
}

@register_wrap
class AutoVisemeButton(bpy.types.Operator):
    bl_idname = 'cats_viseme.create'
    bl_label = t('AutoVisemeButton.label')
    bl_description = t('AutoVisemeButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        viseme_mesh_object_name = context.scene.mesh_name_viseme
        return (
                Common.is_enum_non_empty(viseme_mesh_object_name)
                # Must not be in Edit mode
                and bpy.data.objects[viseme_mesh_object_name].mode != 'EDIT'
        )

    def execute(self, context: Context):
        scene = context.scene
        mesh_obj = bpy.data.objects[scene.mesh_name_viseme]

        if not Common.has_shapekeys(mesh_obj):
            self.report({'ERROR'}, t('AutoVisemeButton.error.noShapekeys'))
            return {'CANCELLED'}

        mesh: Mesh = mesh_obj.data
        shape_keys = mesh.shape_keys
        key_blocks = shape_keys.key_blocks

        reference_shape_key_name = shape_keys.reference_key.name
        shape_a = scene.mouth_a
        shape_o = scene.mouth_o
        shape_ch = scene.mouth_ch
        if (
                shape_a == reference_shape_key_name
                or shape_o == reference_shape_key_name
                or shape_ch == reference_shape_key_name
        ):
            self.report({'ERROR'}, t('AutoVisemeButton.error.selectShapekeys'))
            return {'CANCELLED'}

        # We will be changing the active shape key throughout this function and may re-order the shape keys, so store
        # the name of the currently active shape key so that we can set it as active again at the end.
        orig_shape_key_name = mesh_obj.active_shape_key.name

        # Disable shape key pinning (always shows only the active shape key), so that new shape keys can be created from
        # the mix.
        orig_shape_key_pinning = mesh_obj.show_only_shape_key
        mesh_obj.show_only_shape_key = False

        # Get the shape keys
        shape_key_a = key_blocks.get(shape_a)
        shape_key_o = key_blocks.get(shape_o)
        shape_key_ch = key_blocks.get(shape_ch)

        # Temporarily rename the selected shapes to avoid the case where one of them already has the name of one of the
        # visemes.
        shape_key_a.name = shape_a + "_old"
        shape_a_renamed = shape_key_a.name

        # Currently, renaming a shape key to the name of an existing shape key does not rename the existing shape key,
        # however this is not documented behaviour. For safety, after renaming shape_key_a, we will always get the name
        # of the 'o' and 'ch' shape keys from the shape keys themselves to avoid the possibility that renaming
        # shape_key_a could have resulted in the 'o' and 'ch' shape keys being renamed.
        if shape_key_o == shape_key_a:
            shape_o_renamed = shape_a_renamed
        else:
            shape_key_o.name = shape_key_o.name + "_old"
            shape_o_renamed = shape_key_o.name

        if shape_key_ch == shape_key_a:
            shape_ch_renamed = shape_a_renamed
        elif shape_key_ch == shape_key_o:
            shape_ch_renamed = shape_o_renamed
        else:
            shape_key_ch.name = shape_key_ch.name + "_old"
            shape_ch_renamed = shape_key_ch.name

        total_fors = len(SHAPE_KEY_DATA)

        wm = context.window_manager
        wm.progress_begin(0, total_fors)

        # TODO: If there are non-viseme shape keys that are relative-to one of the shape keys we are deleting, those
        #  shape keys will have their relative key set to the reference key, which may produce undesired results. To
        #  cleanly fix this, we would have to adjust the shape keys that were relative-to shape keys we're deleting by
        #  the difference between the reference key and the old relative-to shape key being deleted.
        # Remove existing vrc. shape keys
        # Deleting shape keys does not seem to affect existing references to specific shape keys, but this is not a
        # documented feature and should not be relied upon. Any existing references to shape keys will be assumed to be
        # invalid after we have deleted shape keys.
        for key_name in SHAPE_KEY_DATA:
            shape_key = key_blocks.get(key_name)
            if shape_key:
                mesh_obj.shape_key_remove(shape_key)

        # Re-get the a, o and ch shape keys to ensure that the references are valid.
        shape_key_a = key_blocks[shape_a_renamed]
        shape_key_o = key_blocks[shape_o_renamed]
        shape_key_ch = key_blocks[shape_ch_renamed]

        # Store existing shape key values and slider_min/slider_max.
        # Temporarily set each shape key's value to 0.0, slider_min to 0.0 and slider_max to 1.0.
        orig_shape_key_data = {}
        new_min = 0.0
        new_max = 1.0  # Must be greater than new_min
        # Each shape key will have its value set to 0.0 so that it doesn't influence the creation of new shape keys.
        new_value = 0.0  # Must be in the range [new_min, new_max]
        for shape_key in key_blocks:
            orig_min = shape_key.slider_min
            orig_max = shape_key.slider_max
            orig_shape_key_data[shape_key] = (shape_key.value, orig_min, orig_max)
            if orig_max < new_min:
                # orig_max is smaller than new_min, so set slider_max first otherwise it's impossible to set new_min.
                shape_key.slider_max = new_max
                shape_key.slider_min = new_min
            else:
                shape_key.slider_min = new_min
                shape_key.slider_max = new_max
            shape_key.value = new_value

        # Add the shape keys into a dict using the same key identifiers used by SHAPE_KEY_DATA
        component_keys = {
            SHAPE_A: shape_key_a,
            SHAPE_O: shape_key_o,
            SHAPE_CH: shape_key_ch,
        }
        shape_intensity = scene.shape_intensity
        for index, (viseme_shape_name, new_shape_components) in enumerate(SHAPE_KEY_DATA.items()):
            wm.progress_update(index)
            # Activate the shape keys that when combined create the new viseme shape key.
            used_component_keys = []
            for component_key_name, value in new_shape_components.items():
                component_key = component_keys[component_key_name]
                used_component_keys.append(used_component_keys)
                component_key.value = value * shape_intensity

            # Create the new shape key.
            mesh_obj.shape_key_add(name=viseme_shape_name, from_mix=True)

            # Reset the values of the component keys in preparation for the next viseme shape key.
            for component_key in used_component_keys:
                component_key.value = 0.0

        # Restore shape key values and slider min/max.
        for shape_key, (orig_value, orig_min, orig_max) in orig_shape_key_data.items():
            if shape_key.slider_max < orig_min:
                # slider_max is smaller than new_min, so set slider_max first otherwise it's impossible to set orig_min.
                shape_key.slider_max = orig_max
                shape_key.slider_min = orig_min
            else:
                shape_key.slider_min = orig_min
                shape_key.slider_max = orig_max
            shape_key.value = orig_value

        # Rename shapes back
        if shape_a not in key_blocks:
            shape_key_a.name = shape_a
        if shape_o not in key_blocks:
            shape_key_o.name = shape_o
        if shape_ch not in key_blocks:
            shape_key_ch.name = shape_ch

        # Set shape key pinning back to its original value
        mesh_obj.show_only_shape_key = orig_shape_key_pinning

        # Re-order visemes to the top, just below the reference key and the blinking shape keys.
        # This is only strictly necessary for VRChat Avatars 2.0.
        Common.sort_shape_keys(mesh_obj.name)

        # Restore the originally active shape key
        mesh_obj.active_shape_key_index = key_blocks.find(orig_shape_key_name)

        # Since we've changed the shape keys, set the shape key names in the scene to the same shape keys as before.
        scene.mouth_a = shape_a
        scene.mouth_o = shape_o
        scene.mouth_ch = shape_ch

        wm.progress_end()

        self.report({'INFO'}, t('AutoVisemeButton.success'))

        return {'FINISHED'}
