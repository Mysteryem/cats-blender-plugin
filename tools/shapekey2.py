import bpy
import numpy as np
from .register import register_wrap

# Issues with CATS' implementation
#   1. Doesn't account for keys that have a different relative key to the first shape key, causing Basis to be set as the relative key of all shape keys
#   2. Operator is only well-defined for shape keys that are all immediately relative to Basis
#   3. Messes with shape key order
#   Bug: Applying two shape keys to basis and then trying to revert them both is broken
#   Bug: Requires Object mode for some Operators it uses, but doesn't include a check for Object mode in its poll method
#
# Extra benefits of this implementation:
#   1.   Maintains relative structure of all shape keys
#   1.2. Shape keys that aren't relative to the basis or new basis remain untouched
#   3.   Only one temporary shape key is created and then deleted
#   3.   Shape key order remains unchanged
#   Reverting shape keys can be done in any order
#   The vertex group of the new basis shape key is kept even after the new basis shape key becomes the reverted shape key of itself
# Extension notes:
#   This should be fairly simple to extend to curves, nurbs and lattices. I'm not sure if there are any other object types with shape keys
#   This should be easy to extend to applying the active shape key to any other 'basis-like' shape key (a shape key relative to itself),
#     simply assign old_basis_shapekey to any other 'basis-like' shape key
# Performance differences:
#   Faster for smaller number of shape keys on meshes with more vertices (individual_add)
#   Faster for smaller meshes with more shape keys (multi_add)
#   Slower for more shape keys on meshes with fewer vertices
bl_info = {
    "name": "Fast Apply Selected Shape Key To Relative Key",
    "blender": (2, 80, 0),
    "category": "Mesh",
}


@register_wrap
class ShapeKeyApplier(bpy.types.Operator):
    # tooltip
    """Applies the selected shape key to its relative key at its current value (or 1 if its value is 0) and reverses the selected shape key"""

    bl_idname = "mysteryem.fast_apply_selected_shape_key_to_relative_key"
    bl_label = "Apply Selected Shape Key To Basis"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    # TODO: Figure out how properties can be defined with Cats such that 2.7x and 2.8x+ can be supported
    use_multi_add: bpy.props.BoolProperty(name="Use multi-shape add")

    @classmethod
    def poll(cls, context):
        # Note that context.object.active_shape_key_index is 0 if there are no shape keys
        # So context.object.active_shape_key_index > 0 simultaneously checks that there are shape keys and that the active shape key isn't the first one
        # TODO: Should we also check context.object.data.shape_keys.use_relative == True?
        return (context.mode == 'OBJECT' and
                context.object and
                # Could be extended to other types that have shape keys, the only part which references something mesh specific is data.vertices, which would need
                # to be changed to data.points in the case of a lattice object. Curves and nurbs don't support vertex groups so wouldn't ever hit that code.
                context.object.type == 'MESH' and
                # If the active shape key is the basis, nothing should be done
                context.object.active_shape_key_index > 0 and
                # If the active shape key is relative to itself, nothing would be changed
                context.object.active_shape_key.relative_key != context.object.active_shape_key)

    def execute(self, context):
        # I will assume most meshes are going to have more than 600 vertices (and for small meshes, both options are very fast)
        # From the benchmark data, it is generally a good idea to pick individual_add when there are less than 8 shape keys to be affected
        # I think this number can be calculated from len(keys_relative_recursive_to_old_basis | keys_relative_recursive_to_new_basis)
        # TODO: See if the speed changes when there are also a bunch of shape keys that remain unaffected (e.g. not relative to either old_basis
        #       or new_basis
        ################
        ################
        ################
        use_multi_add = self.use_multi_add
        # use_multi_add = True
        ################
        ################
        ################

        # debug_timings = {}
        # debug_timings['start'] = time.perf_counter()
        # TODO: Try and figure out why the Cats operator is using this instead of context.object
        #       typically context.object should be used so that context overrides can be used with the operator (e.g. you want to use the operator on
        #       an object that isn't the active object. Also, the poll(...) method uses context.object anyway, there could be problems if Common.get_active()
        #       gets a different object that hasn't been checked against poll(...)
        # mesh = Common.get_active()
        mesh = context.object

        # Get shapekey which will be the new basis
        new_basis_shapekey = mesh.active_shape_key

        # Create a map of key : [keys relative to key]
        # Effectively the reverse of the key.relative_key relation
        reverse_relative_map2 = ShapeKeyApplier.ReverseRelativeMap(mesh)

        # new_basis_shapekey will only be included if it's relative to itself (new_basis_shapekey cannot be the first shape key as poll() ensures
        # that the index of the active shape key is greater than 0)
        keys_relative_recursive_to_new_basis = reverse_relative_map2.get_relative_recursive_keys(new_basis_shapekey)

        # Cancel execution if the new basis shape key is relative to itself (via a loop, since poll already returns false for being immediately relative to itself since that will always do nothing)
        # If the relative keys loop back around, then if the key is turned into its reverse after applying, it would affect all keys that it's relative to
        # Key1 relative -> Key2
        # Key2 relative -> Key1
        # If Key1 is applied to Basis, Key1 should be changed to a reverted key in order to undo the application.
        # Since Key2 is relative to Key1, it has to be modified to account for the change in Key1 so that its relative movement to Key1 stays the same.
        # Since Key1 is relative to Key2, it has to be modified to account for the change in Key2 so that its relative movement to Key2 stays the same, but that creates an infinite loop
        #
        # Another way of looking at it is if Key1 moves a vertex by +1, then Key2 MUST move that same vertex by -1 since they are relative to each other
        # If Key1 is applied to the basis, it should become a reverted key that moves a vertex by -1 instead so that when it's re-applied, it undoes initial application
        # But that would mean that Key2 would have to become a key that moves a vertex by +1, and we want the key to keep its original relative movement of -1
        if new_basis_shapekey in keys_relative_recursive_to_new_basis:
            self.report({'ERROR_INVALID_INPUT'}, 'The shape key {} forms a relative loop with itself and at least one other shape key so cannot be applied to the basis'.format(new_basis_shapekey.name))
            return {'CANCELLED'}

        # It should work to pick a different key as a basis, so long as that key is immediately relative to itself (key.relative_key == key)
        # On the off chance that old_basis_shapekey is not relative to itself, reverse_relative_map(mesh) has special handling that treats it as if it always is
        old_basis_shapekey = mesh.data.shape_keys.key_blocks[0]

        # old_basis_shapekey will only be included if it's relative to itself or if it's the first shape key
        keys_relative_recursive_to_old_basis = reverse_relative_map2.get_relative_recursive_keys(old_basis_shapekey)

        # 0.0 would have no effect, so set to 1.0
        if new_basis_shapekey.value == 0.0:
            new_basis_shapekey.value = 1.0

        if use_multi_add:
            # Optimised for a large number of affected shape keys by using the blend_from_shape operator twice, regardless of how many shape keys there are
            # As blend_from_shape is an edit mode operator, it requires some extra work to set the mesh up so that all the vertices are selected
            # and visible and some extra work to restore the vertex/edge/face selection/visibility afterwards.
            ShapeKeyApplier.multi_add(mesh, new_basis_shapekey, keys_relative_recursive_to_new_basis, old_basis_shapekey, keys_relative_recursive_to_old_basis)
        else:
            ShapeKeyApplier.individual_add(mesh, new_basis_shapekey, keys_relative_recursive_to_new_basis, keys_relative_recursive_to_old_basis)

        # The active key is now a key that reverts to the old relative key so rename it as such
        reverted_string = ' - Reverted'
        reverted_string_len = len(reverted_string)
        old_name = new_basis_shapekey.name

        if new_basis_shapekey.name[-reverted_string_len:] == reverted_string:
            # If the last letters of the name are the reverted_string, remove them
            new_basis_shapekey.name = new_basis_shapekey.name[:-reverted_string_len]
        else:
            # Add the reverted_string to the end of the name, so it's clear that this shape key now reverts
            new_basis_shapekey.name = new_basis_shapekey.name + reverted_string

        # Setting the value to zero will make the mesh appear unchanged in overall shape and help to show that the operator has worked correctly
        new_basis_shapekey.value = 0.0
        new_basis_shapekey.slider_min = 0.0
        # Regardless of what the max was before, 1.0 will now fully undo the applied shape key
        new_basis_shapekey.slider_max = 1.0
        self.report({'INFO'}, 'Applied \'{}\' to Basis shape key \'{}\''.format(old_name, old_basis_shapekey.name))
        return {'FINISHED'}

    class ReverseRelativeMap:
        def __init__(self, obj):
            reverse_relative_map = {}

            basis_key = obj.data.shape_keys.key_blocks[0]
            for key in obj.data.shape_keys.key_blocks:
                # Special handling for basis shape key to treat it as if its always relative to itself
                relative_key = basis_key if key == basis_key else key.relative_key
                keys_relative_to_relative_key = reverse_relative_map.get(relative_key)
                if keys_relative_to_relative_key is None:
                    keys_relative_to_relative_key = {key}
                    reverse_relative_map[relative_key] = keys_relative_to_relative_key
                else:
                    keys_relative_to_relative_key.add(key)
            self.reverse_relative_map = reverse_relative_map

        #
        def get_relative_recursive_keys(self, shape_key):
            shape_set = set()

            # Pretty much a depth-first search, but with loop prevention
            def inner_recursive_loop(key, checked_set):
                # Prevent infinite loops by maintaining a set of shapes that we've checked
                if key not in checked_set:
                    # Need to add the current key to the set of shapes we've checked before the recursive call
                    checked_set.add(key)
                    keys_relative_to_shape_key_inner = self.reverse_relative_map.get(key)
                    if keys_relative_to_shape_key_inner:
                        for relative_to_inner in keys_relative_to_shape_key_inner:
                            shape_set.add(relative_to_inner)
                            inner_recursive_loop(relative_to_inner, checked_set)

            inner_recursive_loop(shape_key, set())
            return shape_set

    @staticmethod
    # Isolate the active shape key such that afterwards, creating a new shape from mix will create a shape key that at
    # a value of 1.0 is the same movement as the active shape key at its current value and vertex group
    # Returns a function that restores the data that got affected due to the isolation
    def isolate_active_shape(obj_with_shapes):
        active_shape = obj_with_shapes.active_shape_key
        restore_data = {}

        # When the value is 1.0, we can simply enable show_only_shape_key on the object
        if active_shape.value == 1.0:
            if obj_with_shapes.show_only_shape_key:
                # Don't need to do anything, it's already isolated
                pass
            else:
                # Store the current .show_only_shape_key value, so it can be restored later
                restore_data['show_only_shape_key'] = False
                obj_with_shapes.show_only_shape_key = True
        # When the value is not 1.0, the next simplest method is to mute all the other shapes on the object
        else:
            # Mute all shapes and save their current .mute value, so it can be restored later
            shapekey_mutes = []
            for key_block in obj_with_shapes.data.shape_keys.key_blocks:
                shapekey_mutes.append(key_block.mute)
                key_block.mute = True
            # Unmute the active shape key
            active_shape.mute = False

            restore_data['mutes'] = shapekey_mutes

            # show_only_shape_key acts as if active_shape.value is always 1.0, so it needs to be disabled if it's enabled
            if obj_with_shapes.show_only_shape_key:
                # store the current value so it can be restored
                restore_data['show_only_shape_key'] = True
                obj_with_shapes.show_only_shape_key = False

        # closure to restore
        def restore_function():
            if restore_data:
                mutes = restore_data.get('mutes')
                if mutes:
                    # Restore shape key mutes
                    for mute, shape in zip(mutes, obj_with_shapes.data.shape_keys.key_blocks):
                        shape.mute = mute
                show_only_shape_key = restore_data.get('show_only_shape_key')
                # show_only_shape_key can be False so need to explicitly check for None
                if show_only_shape_key is not None:
                    # Restore show_only_shape_key
                    obj_with_shapes.show_only_shape_key = show_only_shape_key

        return restore_function

    # Figures out what needs to be added to each affected key, then iterates through all the affected keys, getting the current shape,
    # adding the corresponding amount to it and then setting that as the new shape.
    # This is very fast when the number of vertices is very small or there are only a few shape keys.
    # This avoids EDIT mode entirely, instead getting and setting shape key positions manually with foreach_get and foreach_set
    @staticmethod
    def individual_add(mesh, new_basis_shapekey, keys_relative_recursive_to_new_basis, keys_relative_recursive_to_basis):
        data = mesh.data
        num_verts = len(data.vertices)

        new_basis_shapekey_vertex_group = new_basis_shapekey.vertex_group

        new_basis_affected_by_own_application = new_basis_shapekey in keys_relative_recursive_to_basis

        # Array of Vector type is flattened by foreach_get into a sequence so the length needs to be multiplied by 3
        flattened_co_length = num_verts * 3

        # Store shape key vertex positions for new_basis
        # There's no need to initialise the elements to anything since they will all be overwritten
        # Faster version of new_basis_co_flat = [None]*flattened_co_length
        new_basis_co_flat = np.empty(flattened_co_length, dtype=float)
        # Faster version of new_basis_relative_co_flat = [None]*flattened_co_length
        new_basis_relative_co_flat = np.empty(flattened_co_length, dtype=float)

        new_basis_shapekey.data.foreach_get('co', new_basis_co_flat)
        new_basis_shapekey.relative_key.data.foreach_get('co', new_basis_relative_co_flat)

        # This is movement of the active shape key at a value of 1.0
        # Probably faster version of difference_co_flat = [new_basis_co - relative_co for new_basis_co, relative_co in zip(new_basis_co_flat, new_basis_relative_co_flat)]
        difference_co_flat = np.subtract(new_basis_co_flat, new_basis_relative_co_flat)

        # Scale the difference based on the value of the active key
        # Probably faster version of difference_co_flat_value_scaled = [difference * new_basis_shapekey.value for difference in difference_co_flat]
        difference_co_flat_value_scaled = np.multiply(difference_co_flat, new_basis_shapekey.value)

        # Scale the difference based on the vertex group of the active key
        #   An alternative would be to scale difference_co_flat by the weight of each vertex in new_basis_shapekey.vertex_group
        #   unfortunately, Blender has no efficient way to get all the weights for a particular vertex group, so it's
        #   pretty much always a few times faster to create a new shape from mix and get its 'co' with foreach_get(...)
        #   Tiny meshes, think <1000 vertices, are the exception.
        #
        #   For reference, the ways to get all vertex weights that you can find on stackoverflow:
        #       Weights from vertices:
        #           This scales really poorly when lots of vertices are in multiple vertex groups, especially when the vertices are not the vertex group we want to check,
        #           because for every vertex v, v.groups has to be iterated until either the vertex group is found or iteration finishes without finding the vertex group
        #               vertex_weights = [next((g.weight for g in v.groups if g.group == vertex_group_index), 0) for v in data.vertices]
        #
        #       Weights from vertex group:
        #           This doesn't scale poorly with lots of vertex groups like the other way does, but, if most of the vertices aren't in the vertex group, relying on catching
        #           the exception is really slow. If Blender had a similar method that returned a default value or even just None instead of throwing an exception, this would
        #           be much faster, though maybe still a little slower than creating a new key from mix (ideally we'd want a fast access method like foreach_get(...)) instead
        #           of having to iterate through all the vertices individually
        #               vertex_weights = []
        #               for i in range(num_verts):
        #                   try:
        #                       weight = vertex_group.weight(i)
        #                   except:
        #                       weight = 0
        #                   vertex_weights.append(weight)
        if new_basis_shapekey_vertex_group:
            # Need to isolate the active shape key, so that when a new shape is created from mix, it's only the active shape key
            restore_function = ShapeKeyApplier.isolate_active_shape(mesh)
            # This new shape key has the effect of new_basis.value and new_basis.vertex_group applied
            new_basis_mixed = mesh.shape_key_add(from_mix=True)
            # Restore whatever got changed in order to isolate the active shape key
            restore_function()

            # Reuse the existing arrays, new names for convenience
            temp_shape_co_flat = new_basis_co_flat
            temp_shape_relative_co_flat = new_basis_relative_co_flat

            new_basis_mixed.data.foreach_get('co', temp_shape_co_flat)
            # Often, the relative keys are the same, e.g. they're both the 'basis', but if they're not we'll need to get its data
            if new_basis_mixed.relative_key != new_basis_shapekey.relative_key:
                new_basis_mixed.relative_key.data.foreach_get('co', temp_shape_relative_co_flat)

            # TODO: Reuse an existing array instead of creating a new one?
            difference_co_flat_scaled = np.subtract(temp_shape_co_flat, temp_shape_relative_co_flat)

            # Remove new_basis_mixed
            active_index = mesh.active_shape_key_index
            mesh.shape_key_remove(new_basis_mixed)
            mesh.active_shape_key_index = active_index
        else:
            difference_co_flat_scaled = difference_co_flat_value_scaled

        # We can reuse the same arrays over and over instead of creating new ones each time
        # Faster version of temp_co_array = [None]*flattened_co_length
        temp_co_array = np.empty(flattened_co_length, dtype=float)
        temp_co_array2 = np.empty(flattened_co_length, dtype=float)

        if new_basis_affected_by_own_application:
            # All keys in keys_recursive_relative_to_new_basis must also be in keys_recursive_relative_to_basis
            keys_not_relative_recursive_to_new_basis_and_not_new_basis = (keys_relative_recursive_to_basis - keys_relative_recursive_to_new_basis) - {new_basis_shapekey}

            # TODO: This for loop is where most of the execution will happen for 'normal' setups of lots of shape keys relative to the first shape
            #  Maybe multiprocessing could speed this up, if the overhead isn't too much, maybe to even faster than the multi_add function?
            #  see https://docs.python.org/3/library/multiprocessing.html and https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool
            # from multiprocessing import Pool
            #
            # def multiprocess_add_difference_co_flat_scaled(key_block):
            #     # The issue with using multiprocess is that we can't share the same array for speed any more
            #     co = np.empty(flattened_co_length, dtype=float)
            #     key_block.data.foreach_get('co', co)
            #     key_block.data.foreach_set('co', np.add(co, difference_co_flat_scaled, out=co))
            # None gets count based on
            # with Pool(None) as p:
            #     # automatically picks size of chunks to break keys_not_relative_recursive_to_new_basis into
            #     p.map(multiprocess_add_difference_co_flat_scaled, keys_not_relative_recursive_to_new_basis)
            for key_block in keys_not_relative_recursive_to_new_basis_and_not_new_basis:
                if key_block == new_basis_shapekey:
                    print("no")
                # DEBUG
                # print('Adding shape key difference to {}'.format(key_block.name))
                # Add difference between new_basis_shapekey and new_basis_shapekey.relative_key (scaled according to the value and vertex_group of new_basis_shapekey)
                key_block.data.foreach_get('co', temp_co_array)
                # Faster version of [key_co + difference_co for key_co, difference_co in zip(temp_co_array, difference_co_flat_scaled)]
                key_block.data.foreach_set('co', np.add(temp_co_array, difference_co_flat_scaled, out=temp_co_array))

            # We need the difference between r(NB) and r(NB).r to be the negative of
            #   (r(NB) - r(NB).r) * NB.vg = -((NB - NB.r) * NB.v * NB.vg)
            #                             = -(NB - NB.r) * NB.v * NB.vg
            # NB.vg cancels on both sides, leaving:
            #   r(NB) - r(NB).r = -(NB - NB.r) * NB.v
            # Rearranging for r(NB) gives:
            #   r(NB) = r(NB).r - (NB - NB.r) * NB.v
            # Note that (NB - NB.r) * NB.v = difference_co_flat_value_scaled so:
            #   r(NB) = r(NB).r - difference_co_flat_value_scaled
            # Note that r(NB).r = NB.r + difference_co_flat_scaled as we've added that to it
            #   r(NB) = NB.r + difference_co_flat_scaled - difference_co_flat_value_scaled
            # Note that r(NB) = NB + X where X is what we want to find to add to NB (and all keys relative to it
            # so that their relative differences remain the same)
            #   NB + X = NB.r + difference_co_flat_scaled - difference_co_flat_value_scaled
            #   X = NB.r - NB + difference_co_flat_scaled - difference_co_flat_value_scaled
            #   X = -(NB - NB.r) + difference_co_flat_scaled - difference_co_flat_value_scaled
            # Fully expanding out would give:
            #   X = -(NB - NB.r) + (NB - NB.r) * NB.v * NB.vg - (NB - NB.r) * NB.v
            #
            # In the case of there being a vertex group, it's too costly to calculate NB.vg on its own, so we'll leave it at
            #   X = -(NB - NB.r) + difference_co_flat_scaled - (NB - NB.r) * NB.v
            #   Which we can either factor to
            #       X = (NB - NB.r)(-1 - NB.v) + difference_co_flat_scaled
            #       X = difference_co_flat * (-1 - NB.v) + difference_co_flat_scaled
            #   Or, as NB - NB.r = difference_co_flat, calculate as, which may be faster since it only uses addition/subtraction
            #       X = -difference_co_flat + difference_co_flat_scaled - difference_co_flat_value_scaled
            #
            # From my own benchmarks, np.multiply(array1, scalar, out=output_array) starts to scale slightly better than
            # np.add(array1, array2, out=output_array) once array1 gets to around 9000 elements or more
            # I guess this is due to the fact that the add option needs to do 2 array accesses per element, and that
            # eventually this surpasses the effect of the multiply operation being more expensive than the add operation
            # In this case, the array length is 3*num_verts, meaning the multiplication option gets better at around
            # 3000 vertices, so we'll use the multiplication option
            if new_basis_shapekey_vertex_group:
                np.multiply(difference_co_flat, -1 - new_basis_shapekey.value, out=temp_co_array2)
                np.add(temp_co_array2, difference_co_flat_scaled, out=temp_co_array2)
                for key_block in keys_relative_recursive_to_new_basis | {new_basis_shapekey}:
                    key_block.data.foreach_get('co', temp_co_array)
                    key_block.data.foreach_set('co', np.add(temp_co_array, temp_co_array2, out=temp_co_array))
            # But for there not being a vertex group, the NB.vg term can be eliminated as it becomes effectively 1.0
            #   X = -(NB - NB.r) + (NB - NB.r) * NB.v - (NB - NB.r) * NB.v
            # Then the last part cancels out
            #   X = -(NB - NB.r)
            # Giving X = -difference_co_flat
            else:
                # If there wasn't a vertex group on new_basis:
                # Instead of adding the difference_co_flat_scaled to each key it will be subtracted from each key instead
                for key_block in keys_relative_recursive_to_new_basis | {new_basis_shapekey}:
                    key_block.data.foreach_get('co', temp_co_array)
                    key_block.data.foreach_set('co', np.subtract(temp_co_array, difference_co_flat, out=temp_co_array))

                # it will be affected by its own application, so we can simply set it to the old positions of its relative key
                new_basis_shapekey.data.foreach_set('co', new_basis_relative_co_flat)
        else:
            # New basis isn't relative to Basis so keys New basis is recursively relative to will remain unchanged
            # Keys recursively relative to Basis and Keys recursively relative to new basis will be mutually exclusive
            # Typical user setups have all the shape keys immediately relative to Basis, so this won't be used much

            for key_block in keys_relative_recursive_to_basis:
                # Add difference between new_basis_shapekey and new_basis_shapekey.relative_key (scaled according to the value and vertex_group of new_basis_shapekey)
                key_block.data.foreach_get('co', temp_co_array)
                # Faster version of [key_co + difference_co for key_co, difference_co in zip(temp_co_array, difference_co_flat_scaled)]
                key_block.data.foreach_set('co', np.add(temp_co_array, difference_co_flat_scaled, out=temp_co_array))

            # The difference between the reverted key and its relative key needs to equal the negative of the
            # difference between new_basis and new_basis.relative_key multiplied
            # new_basis.vertex_group should be present on both
            #   (r(NB) - r(NB).r) * NB.vg = -((NB - NB.r) * NB.v * NB.vg)
            #                             = -(NB - NB.r) * NB.v * NB.vg
            # NB.vg cancels on both sides, leaving:
            #   r(NB) - r(NB).r = -(NB - NB.r) * NB.v
            # r(NB).r is unchanged, so r(NB).r = NB.r
            #   r(NB) - NB.r = -(NB - NB.r) * NB.v
            # r(NB) = X + NB where X is what we want to find to add
            #   X + NB - NB.r = -(NB - NB.r) * NB.v
            # Rearrange for X
            #   X = -(NB - NB.r) - (NB - NB.r) * NB.v
            #
            # (NB - NB.r) can be factorised
            #   X = (NB - NB.r)(-1 - NB.v)
            # Note that (NB - NB.r) is difference_co_flat, giving
            #   X = difference_co_flat * (-1 - NB.v)
            #
            # Alternatively, instead of factorising, note that (NB - NB.r) * NB.v is difference_co_flat_value_scaled
            #   X = -(NB - NB.r) - difference_co_flat_value_scaled
            # Note that (NB - NB.r) is difference_co_flat, giving
            #   X = -difference_co_flat - difference_co_flat_value_scaled
            # Or
            #   X = -(difference_co_flat + difference_co_flat_value_scaled)
            #
            # Since NB.vg isn't present, it doesn't matter whether new_basis_shapekey has a vertex_group or not
            #
            # As with before, we'll use the multiplication option due to it scaling slightly better with a larger
            # number of vertices
            # X = difference_co_flat * (-1 - NB.v)
            np.multiply(difference_co_flat, -1 - new_basis_shapekey.value, out=temp_co_array2)

            for key_block in keys_relative_recursive_to_new_basis | {new_basis_shapekey}:
                key_block.data.foreach_get('co', temp_co_array)
                key_block.data.foreach_set('co', np.add(temp_co_array, temp_co_array2, out=temp_co_array))

    # To use the blend_from_shape modifier, all the vertices need to be visible and selected in order to affect them all (faces and edges don't matter)
    # use_shape_key_edit_mode needs to be disabled because if enabled it could be slow if other shape keys are active and if the active shape key does not have
    # a value of 1.0, it will affect how blend_from_shape gets applied, e.g., if the value is 0.0, the blend_from_shape modifier will do nothing.
    class BlendFromShapeEditModeHelper:
        is_set_up = False

        def __init__(self, mesh):
            self.mesh = mesh

        def __store__(self):
            if not self.is_set_up:
                data = self.mesh.data
                num_verts = len(data.vertices)
                num_edges = len(data.edges)
                num_polygons = len(data.polygons)

                # Store current hide values so they can be restored later
                self.vertex_hide_values = np.empty(num_verts, dtype=bool)
                self.edge_hide_values = np.empty(num_edges, dtype=bool)
                self.polygon_hide_values = np.empty(num_polygons, dtype=bool)
                data.vertices.foreach_get('hide', self.vertex_hide_values)
                data.edges.foreach_get('hide', self.edge_hide_values)
                data.polygons.foreach_get('hide', self.polygon_hide_values)

                # Store current select values so they can be restored later
                self.vertex_select_values = np.empty(num_verts, dtype=bool)
                self.edge_select_values = np.empty(num_edges, dtype=bool)
                self.polygon_select_values = np.empty(num_polygons, dtype=bool)
                data.vertices.foreach_get('select', self.vertex_select_values)
                data.edges.foreach_get('select', self.edge_select_values)
                data.polygons.foreach_get('select', self.polygon_select_values)

                # Store the current shape key edit mode setting
                self.use_shape_key_edit_mode = self.mesh.use_shape_key_edit_mode

        def pre_edit_mode_setup(self):
            if not self.is_set_up:
                self.__store__()

                data = self.mesh.data

                # Turn off use_shape_key_edit_mode
                self.mesh.use_shape_key_edit_mode = False
                # For large meshes, these get slower, but are still faster than revealing and selecting the whole mesh while in edit mode using the corresponding operators,
                #   particularly the unhiding
                num_verts = len(data.vertices)
                data.vertices.foreach_set('hide', np.full(num_verts, False, dtype=bool))
                data.vertices.foreach_set('select', np.full(num_verts, True, dtype=bool))

                self.is_set_up = True

        def restore(self):
            if self.is_set_up:
                data = self.mesh.data

                # Restore use_shape_key_edit_mode
                self.mesh.use_shape_key_edit_mode = self.use_shape_key_edit_mode

                # Restore hide attributes
                data.vertices.foreach_set('hide', self.vertex_hide_values)
                data.edges.foreach_set('hide', self.edge_hide_values)
                data.polygons.foreach_set('hide', self.polygon_hide_values)

                # Restore select attributes
                data.vertices.foreach_set('select', self.vertex_select_values)
                data.edges.foreach_set('select', self.edge_select_values)
                data.polygons.foreach_set('select', self.polygon_select_values)

                self.is_set_up = False

    @staticmethod
    def multi_add(mesh, new_basis_shapekey, keys_relative_recursive_to_new_basis, old_basis_shapekey, keys_relative_recursive_to_basis):
        # Need to isolate the active shape key, so that when a new shape is created from mix, it's only the active shape key
        isolate_active_restore_function = ShapeKeyApplier.isolate_active_shape(mesh)

        # First, apply new_basis_shapekey multiplied by its value and its vertex group to all shape keys immediately or recursively relative to the target_basis
        #   This is effectively applying the active shape key with its current value and vertex group to the basis
        #   To do this, we mute all shape keys but the active one (already done) and create a temporary new shape from mix
        #   This temporary shape will be applied to the target_basis and all shape keys immediately or recursively relative to the target_basis
        #
        #   new_shape_from_mix() -> temp_shape
        #     # true_basis is the first shape key
        #     temp_shape - true_basis = (NB - NB.r) * NB.v * NB.vg
        temp_shape = mesh.shape_key_add(name="temp shape (you shouldn't see this)", from_mix=True)

        # Restore whatever got changed in order to isolate the active shape key
        isolate_active_restore_function()

        # fast_multi_shape_add_key changes the active shape key and shape key removal sets the active shape key index to 0
        # We'll want to restore the active shape key when we're done
        new_basis_shapekey_index = mesh.active_shape_key_index

        # fast_multi_shape_add_key uses the blend_from_shape modifier, which means we need to make sure edit mode
        # is set up such that all vertices get fully affected when blend_from_shape is used
        # This helper will perform the required setup when needed and can be used afterwards to restore what got changed
        blend_from_shape_edit_mode_helper = ShapeKeyApplier.BlendFromShapeEditModeHelper(mesh)

        #   add(temp_shape, value=1) to shapes in r_relative(target_basis) | {target_basis}
        ShapeKeyApplier.multi_shape_add_key(shape_key_to_add=temp_shape,
                                            shapes_to_affect=keys_relative_recursive_to_basis | {old_basis_shapekey},
                                            mesh=mesh,
                                            edit_mode_helper=blend_from_shape_edit_mode_helper)

        # Remove the temporary key
        mesh.shape_key_remove(temp_shape)

        # Second, apply new_basis_shapekey to itself such that when it's applied a second time, it reverts its initial application
        #   For this, all keys immediately or recursively relative to new_basis_shapekey will need to be affected by the same amount
        #   The reverted(new_basis) * new_basis.vertex_group needs to equal the negative of the new_basis * new_basis.value * new_basis.vertex_group
        #   Shorthand key:
        #       NB: new_basis
        #     r(x): reverted shape of shape x
        #      x.r: relative_key of shape x
        #      x.v: value of shape x
        #     x.vg: vertex group of shape x, if there is no vertex group, x.vg = 1, though it doesn't actually matter since the vertex group cancels out
        #     (r(NB) - r(NB).r) * NB.vg = -(NB - NB.r) * NB.v * NB.vg
        #   NB.vg cancels
        #     (r(NB) - r(NB).r) = -(NB - NB.r) * NB.v
        #     r(NB) - r(NB).r = -(NB - NB.r) * NB.v
        #   Note that since NB.r isn't immediately relative or recursively relative to NB, it won't be affected, so r(NB).r = NB.r
        #     r(NB) - NB.r = -(NB - NB.r) * NB.v
        #   Note that r(NB) = X + NB where X is what we need to find to add
        #     X + NB - NB.r = -(NB - NB.r) * NB.v
        #   Rearrange
        #     X + (NB - NB.r) = -(NB - NB.r) * NB.v
        #   Isolate X
        #     X = -(NB - NB.r) - (NB - NB.r) * NB.v
        #   Factorize out (NB - NB.r)
        #     X = (NB - NB.r) * (-1 - NB.v)
        #   Note that new_basis_shapekey = (NB - NB.r)
        #     X = new_basis_shapekey * (-1 - NB.v)
        #
        #   add(NB, value=-1 - NB.v) to shapes in r_relative(NB) | {NB}
        ShapeKeyApplier.multi_shape_add_key(shape_key_to_add=new_basis_shapekey,
                                            shapes_to_affect=keys_relative_recursive_to_new_basis | {new_basis_shapekey},
                                            mesh=mesh,
                                            value=-1 - new_basis_shapekey.value,
                                            edit_mode_helper=blend_from_shape_edit_mode_helper)

        # Restore vert/edge/face hide and select status, and use_shape_key_edit_mode
        blend_from_shape_edit_mode_helper.restore()

        # Restore the active shape key index
        mesh.active_shape_key_index = new_basis_shapekey_index

    @staticmethod
    def multi_shape_add_key(*, shape_key_to_add, shapes_to_affect, mesh, value=1.0, edit_mode_helper):
        to_add_relative_key = shape_key_to_add.relative_key

        if shapes_to_affect and value != 0.0 and to_add_relative_key != shape_key_to_add:

            # Make sure it's a set
            shapes_to_affect = set(shapes_to_affect)

            # Out of all the shapes_to_affect, we need to pick one of them that all the shapes_to_affect will temporarily
            # have their relative keys set to. This will also be the shape key that gets used as the active shape key when
            # entering edit mode.
            # It's important that shape_key_to_add's relative key remains the same since the difference between those
            # two keys are what will be added in edit mode
            if shape_key_to_add in shapes_to_affect:
                if to_add_relative_key in shapes_to_affect:
                    # Both shape_key_to_add and to_add_relative_key are to be affected,
                    # make sure to_add_relative_key is picked as the temporary basis,
                    # that way shape_key_to_add's relative_key remains as to_add_relative_key
                    temporary_affected_basis = to_add_relative_key
                else:
                    # shape_key_to_add is to be affected, but to_add_relative_key isn't, make sure shape_key_to_add is picked
                    # as the temporary basis. We will restore shape_key_to_add's relative_key afterwards.
                    # Other keys cannot be picked as blend_from_shape will only affect the active shape key and all
                    # shape keys immediately relative to that active shape key.
                    # If we were to pick other_key as the temporary_affected_basis, the only way shape_key_to_add could
                    # get affected is if shape_key_to_add.relative_key == other_key, but then shape_key_to_add's relative
                    # key is no longer to_add_relative_key, so the blend_from_shape will blend the wrong amount
                    temporary_affected_basis = shape_key_to_add
            elif to_add_relative_key in shapes_to_affect:
                # If to_add_relative_key is in the set and shape_key_to_add is not:
                #   If we were to pick to_add_relative_key as temporary_affected_basis and blend shape_key_to_add into it:
                #     Since shape_key_to_add is relative to to_add_relative_key (and must remain as such as what we're adding
                #     is the difference between shape_key_to_add and to_add_relative_key), shape_key_to_add will also get updated.
                #
                #   If there are other shape keys to pick from, and we pick one of those, to_add_relative_key will get updated due to its temporary relative_key being updated
                #   but this update will not propagate to shape_key_to_add. shape_key_to_add will remain unchanged (though the difference between shape_key_to_add and
                #   to_add_relative_key will change)
                #
                #   Technically, if shape_key_to_add is a temporary key, we don't care if it gets modified even though it was in shapes_to_affect, but it would result
                #   in 1 more shape being affected than is necessary, which would be a little slower, particularly for large meshes
                #
                # Set the temporary_affected_basis as the first shape that isn't to_add_relative_key
                # The worst case is two iterations, where the first iteration was to_add_relative_key, but that the condition rejects
                # If there's only one element, then we'll have to use to_add_relative_key
                temporary_affected_basis = next((x for x in shapes_to_affect if x is not to_add_relative_key), to_add_relative_key)

                if temporary_affected_basis is to_add_relative_key:
                    # Generally, there shouldn't be only one element in shapes_to_affect since this function is designed for when there are multiple elements,
                    # so this condition shouldn't ever be true if the function is being used correctly
                    # If we do get here, what this means is that we want to add the difference between shape_key_to_add and to_add_relative_key to only to_add_relative_key,
                    #   which is the same as setting to_add_relative_key to shape_key_to_add, so we'll do just that, albeit manually, since it should be faster
                    # Create an empty array for storing the flattened 'co' vectors
                    shape_key_to_add_co = np.empty(len(shape_key_to_add.data * 3), dtype=float)
                    # Fill the array with the flattened 'co' vectors of shape_key_to_add
                    shape_key_to_add.data.foreach_get('co', shape_key_to_add_co)
                    # Set to_add_relative_key's 'co' vectors from the array
                    to_add_relative_key.foreach_set('co', shape_key_to_add_co)
                    # There's nothing more to do, so return
                    return
            else:
                # it doesn't matter which element it is, so get the first element by iterator since it's a set
                temporary_affected_basis = next(iter(shapes_to_affect))

            data = mesh.data
            all_shapes = data.shape_keys.key_blocks
            unaffected_shapes = set(all_shapes) - shapes_to_affect

            # If any of the shapes in unaffected_shapes are relative to temporary_affected_basis, they would get modified, but we don't want that
            # Pick any shape in unaffected_shapes and set that as the relative_key of all the unaffected_shapes, this will ensure the shapes are unaffected
            # We will need to restore their relative keys once we're done, so we'll put them into a list
            unaffected_relative_keys = []
            if unaffected_shapes:
                # unaffected_shapes is a set, so we'll use the first value from its iterator
                temporary_unaffected_basis = next(iter(unaffected_shapes))
                for shape in unaffected_shapes:
                    unaffected_relative_keys.append((shape, shape.relative_key))
                    shape.relative_key = temporary_unaffected_basis

            # Set temporary_affected_basis as the relative_key of all the shapes_to_affect
            # We will need to restore their relative keys once we're done, so we'll put them into a list
            affected_relative_keys = []
            for shape in shapes_to_affect:
                affected_relative_keys.append((shape, shape.relative_key))
                shape.relative_key = temporary_affected_basis

            # shape_key_to_add may have just had its relative_key changed, but it must not be changed, otherwise what's being added will be changed
            # We have already accounted for this by picking temporary_affected_basis carefully, so it is safe to restore the relative_key
            # An alternative to setting the relative key back would be to check if the iterated element is not shape_key_to_add when iterating
            #   unaffected_shapes or shapes_to_affect so that shape_key_to_add.relative_key never gets changed in the first place
            shape_key_to_add.relative_key = to_add_relative_key

            # Make temporary_affected_basis the active shape key
            # It's important to do this in 'OBJECT' mode, since changing active shape key in 'EDIT' mode is slow for large meshes
            mesh.active_shape_key_index = data.shape_keys.key_blocks.find(temporary_affected_basis.name)

            # Make sure edit mode is set up correctly for using blend from shape
            edit_mode_helper.pre_edit_mode_setup()

            # Blend from shape is an edit mode operator so the mode needs to be changed to 'EDIT'
            # Note that for meshes with a large number of vertices, going into edit mode from object mode is slow
            bpy.ops.object.mode_set(mode='EDIT')

            # Additively blend our prepared shape into temporary_affected_basis, affecting temporary_affected_basis and all keys immediately relative to it
            # add=True will blend in the difference from temp_shape and temp_shape.relative_key
            bpy.ops.mesh.blend_from_shape(shape=shape_key_to_add.name, blend=value, add=True)

            # Exiting out of object mode (or changing the active shape key) will cause all shape keys with relative_key equal to temporary_affected_basis to be updated too
            # The relative keys cannot be restored before doing this, otherwise the wrong shape keys will be updated
            bpy.ops.object.mode_set(mode='OBJECT')

            # Restore the relative keys of the unaffected shapes
            for shape, relative in unaffected_relative_keys:
                shape.relative_key = relative

            # Restore the relative keys of the affected shapes
            for shape, relative in affected_relative_keys:
                shape.relative_key = relative


# Add/remove from Shape Key Specials dropdown
def draw_menu(self, context):
    self.layout.separator()
    self.layout.operator(ShapeKeyApplier.bl_idname, icon='KEY_HLT')


# Register addon and add Operator to Shape Key Specials dropdown
def register(menu=True):
    bpy.utils.register_class(ShapeKeyApplier)
    if menu:
        if hasattr(bpy.types, 'MESH_MT_shape_key_specials'):  # pre 2.80
            bpy.types.MESH_MT_shape_key_specials.append(draw_menu)
        else:
            bpy.types.MESH_MT_shape_key_context_menu.append(draw_menu)


# Unregister addon and remove Operator from Shape Key Specials dropdown
def unregister(menu=True):
    if menu:
        if hasattr(bpy.types, 'MESH_MT_shape_key_specials'):  # pre 2.80
            bpy.types.MESH_MT_shape_key_specials.remove(draw_menu)
        else:
            bpy.types.MESH_MT_shape_key_context_menu.remove(draw_menu)
    bpy.utils.unregister_class(ShapeKeyApplier)


# Test from text editor
if __name__ == "__main__":
    register(menu=False)