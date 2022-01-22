# MIT License

# Copyright (c) 2017 GiveMeAllYourCats

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

# Code author: Mysteryem
# Repo: https://github.com/michaeldegroot/cats-blender-plugin

import unittest
import sys
import bpy

from tools.common import version_2_79_or_older


class TestAddon(unittest.TestCase):
    def set_active_by_name(self, object_name):
        if version_2_79_or_older():
            objects = bpy.context.scene.objects
        else:
            objects = bpy.context.view_layer.objects

        object_index = objects.find(object_name)
        self.assertGreaterEqual(object_index, 0, msg="Could not find object with name {name}".format(name=object_name))
        found_object = objects[object_index]
        objects.active = found_object
        return found_object

    def assertShapeKeysEqual(self, obj, expected_flattened_co_array, *shape_key_names):
        for shape_key_name in shape_key_names:
            shape_key = obj.data.shape_keys.key_blocks[shape_key_name]
            flattened_co_array = [None, ] * len(shape_key.data)
            shape_key.foreach_get('co', flattened_co_array)
            self.assertEqual(expected_flattened_co_array, flattened_co_array,
                             msg="Expected {name} to be {expected}, but it was {actual}"
                             .format(name=shape_key_name, expected=expected_flattened_co_array, actual=flattened_co_array))

    def assertShapeKeysRelativeTo(self, obj, expected_relative_key_name, *shape_key_names):
        for shape_key_name in shape_key_names:
            shape_key = obj.data.shape_keys.key_blocks[shape_key_name]
            self.assertEqual(expected_relative_key_name, shape_key.relative_key.name)

    def assertShapeKeyValueEquals(self, obj, expected_value, shape_key_name):
        shape_key = obj.data.shape_keys.key_blocks[shape_key_name]
        self.assertEqual(expected_value, shape_key.value)

    def assertShapeKeyVertexGroupNameEquals(self, obj, expected_name, shape_key_name):
        shape_key = obj.data.shape_keys.key_blocks[shape_key_name]
        self.assertEqual(expected_name, shape_key.vertex_group)

    # When there are no shape keys, poll() should return False, resulting in a RuntimeError
    def test_no_shape_keys(self):
        obj = self.set_active_by_name('01_NoShapes')
        self.assertRaises(RuntimeError, bpy.ops.cats_shapekey.shape_key_to_basis)

    # When the basis is the active shape key, poll() should return False, resulting in a RuntimeError
    def test_active_is_basis(self):
        obj = self.set_active_by_name('02_BasisActive')
        self.assertRaises(RuntimeError, bpy.ops.cats_shapekey.shape_key_to_basis)

    # When the basis is the active shape key, poll() should return False, resulting in a RuntimeError
    # This test is to show that poll() isn't returning False in test_active_is_basis just because the Basis is relative
    # to itself
    def test_active_is_basis_relative_to_key1(self):
        obj = self.set_active_by_name('03_BasisActiveRelativeToKey1')
        self.assertRaises(RuntimeError, bpy.ops.cats_shapekey.shape_key_to_basis)

    # When the active shape key is relative to itself, poll() should return False, resulting in a RuntimeError
    def test_active_relative_to_self(self):
        self.set_active_by_name('04_ActiveRelativeToSelf')
        self.assertRaises(RuntimeError, bpy.ops.cats_shapekey.shape_key_to_basis)

    # A shape key that is recursively relative to itself cannot be applied to the basis as it cannot be correctly
    # set to a shape key that reverts its initial application.
    def test_active_recursively_relative_to_self(self):
        self.set_active_by_name('05_ActiveRecursivelyRelativeToSelf')
        self.assertEqual(bpy.ops.cats_shapekey.shape_key_to_basis(), {'CANCELLED'})

    # The active shape (Key 1 [1,1,1]) is relative to the Basis
    # Basis is [0,0,0]
    # Key 2 [0,0,0] is relative to the Basis -> Another key relative to the Basis
    # Key 3 [0,0,0] is relative to Key 1     -> A key relative to the active shape
    # Key 4 [0,0,0] is relative to Key 2     -> A key relative to a key relative to the Basis
    # Key 5 [0,0,0] is relative to itself    -> A key relative to itself
    # Key 6 [0,0,0] is relative to Key 5     -> A key that is not recursively relative to the Basis
    # When applied,
    #     Basis becomes Basis + (Key 1 - Basis) = [1,1,1]
    #     Key 1 becomes Key 1 - (Key 1 - Basis) = [0,0,0]
    #     Key 2 becomes Key 2 + (Key 1 - Basis) = [1,1,1]
    #     Key 3 becomes Key 3 - (Key 1 - Basis) = [-1,-1,-1]
    #     Key 4 becomes Key 4 + (Key 1 - Basis) = [1,1,1]
    #     Key 5 remains as [0,0,0]
    #     Key 6 remains as [0,0,0]
    def immediately_relative_to_basis(self, object_name):
        obj = self.set_active_by_name(object_name)
        self.assertShapeKeysEqual(obj, [1, 1, 1], 'Key 1')
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Basis', 'Key 2', 'Key 3', 'Key 4', 'Key 5', 'Key 6')

        self.assertEqual(bpy.ops.cats_shapekey.shape_key_to_basis(), {'FINISHED'})

        self.assertShapeKeysEqual(obj, [1, 1, 1], 'Basis', 'Key 2', 'Key 4')
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Key 1 - Reverted', 'Key 5', 'Key 6')
        self.assertShapeKeysEqual(obj, [-1, -1, -1], 'Key 3')
        return obj

    def test_immediately_relative_to_basis_individual(self):
        self.immediately_relative_to_basis('06A_ImmediatelyRelative_Individual')

    # 10 extra shape keys all relative to the Basis to ensure the functions that are faster with many shape keys are used
    def test_immediately_relative_to_basis_multi(self):
        obj = self.immediately_relative_to_basis('06B_ImmediatelyRelative_Multi')
        # Keys 7 to 16 should all be [1,1,1]
        self.assertShapeKeysEqual(obj, [1, 1, 1], *['Key ' + str(i) for i in range(7, 17)])

    # The active shape (Key 1 [1,1,1]) is relative to Key 2 [0,0,0]
    # Basis is [0,0,0]
    # Key 2 [0,0,0] is relative to the Basis -> A key relative to the Basis that the active shape key is relative to
    # Key 3 [0,0,0] is relative to the Basis -> A key relative to the Basis that the active shape key is not relative to
    # Key 4 [0,0,0] is relative to Key 1     -> A key relative to the active shape key
    # Key 5 [0,0,0] is relative to Key 2     -> A key relative to the same shape key that the active shape key is relative to
    # Key 6 [0,0,0] is relative to itself    -> A key relative to itself
    # Key 7 [0,0,0] is relative to Key 6     -> A key that is not recursively relative to the Basis
    # When applied,
    #     Basis becomes Basis + (Key 1 - Key 2) = [1,1,1]
    #     Key 1 becomes Key 1 - (Key 1 - Key 2) = [0,0,0]
    #     Key 2 becomes Key 2 + (Key 1 - Key 2) = [1,1,1]
    #     Key 3 becomes Key 3 + (Key 1 - Key 2) = [1,1,1]
    #     Key 4 becomes Key 1 - (Key 1 - Key 2) = [-1,-1,-1]
    #     Key 5 becomes Key 5 + (Key 1 - Key 2) = [1,1,1]
    #     Key 6 remains as [0,0,0]
    #     Key 7 remains as [0,0,0]
    def recursively_relative_to_basis(self, object_name):
        obj = self.set_active_by_name(object_name)
        self.assertShapeKeysEqual(obj, [1, 1, 1], 'Key 1')
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Basis', 'Key 2', 'Key 3', 'Key 4', 'Key 5', 'Key 6', 'Key 7')

        self.assertEqual(bpy.ops.cats_shapekey.shape_key_to_basis(), {'FINISHED'})

        self.assertShapeKeysEqual(obj, [1, 1, 1], 'Basis', 'Key 2', 'Key 3', 'Key 5')
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Key 1 - Reverted', 'Key 6', 'Key 7')
        self.assertShapeKeysEqual(obj, [-1, -1, -1], 'Key 4')

        return obj

    def test_recursively_relative_to_basis_individual(self):
        self.recursively_relative_to_basis('07A_RecursivelyRelative_Individual')

    # 10 extra shape keys all relative to the Basis to ensure the functions that are faster with many shape keys are used
    def test_recursively_relative_to_basis_multi(self):
        obj = self.recursively_relative_to_basis('07B_RecursivelyRelative_Multi')
        # Keys 8 to 17 should all be [1,1,1]
        self.assertShapeKeysEqual(obj, [1, 1, 1], *['Key ' + str(i) for i in range(8, 18)])

    # The active shape (Key 2 [1,1,1]) isn't recursively relative to the basis and is instead relative to Key 1 [0,0,0]
    # Basis is [0,0,0]
    # Key 1 [0,0,0] is relative to itself
    # Key 3 [0,0,0] is relative to the Basis
    # Key 4 [0,0,0] is relative to Key 3
    # Key 5 [0,0,0] is relative to Key 1
    # Key 6 [0,0,0] is relative to Key 2
    # Key 7 [0,0,0] is relative to itself
    # Key 8 [0,0,0] is relative to Key 7
    # When applied,
    #     Basis becomes Basis + (Key 2 - Key 1) = [1,1,1]
    #     Key 1 remains as [0,0,0]
    #     Key 2 becomes Key 2 - (Key 2 - Key 1) - (Key 2 - Key 1) = [-1,-1,-1]
    #     Key 3 becomes Key 3 + (Key 2 - Key 1) = [1,1,1]
    #     Key 4 becomes Key 4 + (Key 2 - Key 1) = [1,1,1]
    #     Key 5 remains as [0,0,0]
    #     Key 6 becomes Key 6 - (Key 2 - Key 1) - (Key 2 - Key 1) = [-1,-1,-1]
    #     Key 7 remains as [0,0,0]
    #     Key 8 remains as [0,0,0]
    def separate_from_basis(self, object_name):
        obj = self.set_active_by_name(object_name)
        self.assertShapeKeysEqual(obj, [1, 1, 1], 'Key 2')
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Basis', 'Key 1', 'Key 3', 'Key 4', 'Key 5', 'Key 6', 'Key 7', 'Key 8')

        self.assertEqual(bpy.ops.cats_shapekey.shape_key_to_basis(), {'FINISHED'})

        self.assertShapeKeysEqual(obj, [1, 1, 1], 'Basis', 'Key 3', 'Key 4')
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Key 1', 'Key 5', 'Key 7', 'Key 8')
        self.assertShapeKeysEqual(obj, [-1, -1, -1], 'Key 2 - Reverted', 'Key 6')

        return obj

    def test_separate_from_basis_individual(self):
        self.separate_from_basis('08A_Separate_Individual')

    # 10 extra shape keys all relative to the Basis to ensure the functions that are faster with many shape keys are used
    def test_separate_from_basis_multi(self):
        obj = self.separate_from_basis('08B_Separate_Multi')
        # Keys 9 to 18 should all be [1,1,1]
        self.assertShapeKeysEqual(obj, [1, 1, 1], *['Key ' + str(i) for i in range(9, 19)])

    # The active shape key (Key 1 [1,1,1]) is relative to the Basis and the Basis is mistakenly relative to Key 1
    #     Despite the fact that would normally be cancelled due to an infinite loop, the Basis instead gets treated as
    #     if it's always relative to itself and the operator succeeds
    #     Basis is [0,0,0] and is relative to Key 1
    #     Key 1 is [1,1,1] and is relative to Basis
    #     When applied,
    #         Basis becomes Basis + (Key 1 - Basis) = [1,1,1]
    #         Key 1 becomes Key 1 - (Key 1 - Basis) = [0,0,0]
    def basis_relative_loop_special_case(self, object_name):
        obj = self.set_active_by_name(object_name)
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Basis')
        self.assertShapeKeysEqual(obj, [1, 1, 1], 'Key 1')
        self.assertShapeKeysRelativeTo(obj, 'Basis', 'Key 1')
        self.assertShapeKeysRelativeTo(obj, 'Key 1', 'Basis')

        self.assertEqual(bpy.ops.cats_shapekey.shape_key_to_basis(), {'FINISHED'})

        self.assertShapeKeysEqual(obj, [1, 1, 1], 'Basis')
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Key 1')
        self.assertShapeKeysRelativeTo(obj, 'Basis', 'Key 1 - Reverted')
        self.assertShapeKeysRelativeTo(obj, 'Key 1 - Reverted', 'Basis')

        return obj

    def test_basis_relative_loop_special_case_individual(self):
        self.basis_relative_loop_special_case('09A_BasisLoopSpecial_Individual')

    # 10 extra shape keys all relative to the Basis to ensure the functions that are faster with many shape keys are used
    def test_basis_relative_loop_special_case_multi(self):
        obj = self.basis_relative_loop_special_case('09B_BasisLoopSpecial_Multi')
        # Keys 2 to 11 should all be [1,1,1]
        self.assertShapeKeysEqual(obj, [1, 1, 1], *['Key ' + str(i) for i in range(2, 12)])

    # The active shape key (Key 1 [1,1,1]) is relative to the Basis and the value of the active shape key is 0.25
    #     Basis is [0,0,0]
    #     Key 1 is [1,1,1] and is relative to Basis
    #     When applied,
    #         Basis becomes Basis + (Key 1 - Basis) * 0.25 = [0.25,0.25,0.25]
    #         Key 1 becomes Key 1 - (Key 1 - Basis) = [0,0,0]
    def non_zero_value(self, object_name):
        obj = self.set_active_by_name(object_name)
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Basis')
        self.assertShapeKeysEqual(obj, [1, 1, 1], 'Key 1')
        self.assertShapeKeyValueEquals(obj, 0.25, 'Key 1')

        self.assertEqual(bpy.ops.cats_shapekey.shape_key_to_basis(), {'FINISHED'})

        self.assertShapeKeysEqual(obj, [0.25, 0.25, 0.25], 'Basis')
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Key 1 - Reverted')

        return obj

    def test_non_zero_value_individual(self):
        self.non_zero_value('10A_Value_Individual')

    # 10 extra shape keys all relative to the Basis to ensure the functions that are faster with many shape keys are used
    def test_non_zero_value_multi(self):
        obj = self.non_zero_value('10B_Value_Multi')
        # Keys 2 to 11 should all be [0.25,0.25,0.25]
        self.assertShapeKeysEqual(obj, [0.25, 0.25, 0.25], *['Key ' + str(i) for i in range(2, 12)])

    # The active shape key (Key 1 [1,1,1]) is relative to the Basis and has a vertex group with the single vertex of the mesh with value 0.25
    #     Basis is [0,0,0]
    #     Key 1 is [1,1,1] and is relative to Basis
    #     When applied,
    #         Basis becomes Basis + (Key 1 - Basis) * 0.25 = [0.25,0.25,0.25]
    #         Key 1 becomes Key 1 - (Key 1 - Basis) = [0,0,0]
    def vertex_group(self, object_name):
        obj = self.set_active_by_name(object_name)
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Basis')
        self.assertShapeKeysEqual(obj, [1, 1, 1], 'Key 1')
        self.assertShapeKeyVertexGroupNameEquals(obj, 'Group', 'Key 1')

        self.assertEqual(bpy.ops.cats_shapekey.shape_key_to_basis(), {'FINISHED'})

        self.assertShapeKeysEqual(obj, [0.25, 0.25, 0.25], 'Basis')
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Key 1 - Reverted')
        self.assertShapeKeyVertexGroupNameEquals(obj, 'Group', 'Key 1 - Reverted')

        return obj

    def test_vertex_group_individual(self):
        self.vertex_group('11A_VertexGroup_Individual')

    # 10 extra shape keys all relative to the Basis to ensure the functions that are faster with many shape keys are used
    def test_vertex_group_multi(self):
        obj = self.vertex_group('11B_VertexGroup_Multi')
        # Keys 2 to 11 should all be [0.25,0.25,0.25]
        self.assertShapeKeysEqual(obj, [0.25, 0.25, 0.25], *['Key ' + str(i) for i in range(2, 12)])

    # The active shape key (Key 1 [1,1,1]) is relative to the Basis and has a value of 0.25 and a vertex group with the single vertex of the mesh with value 0.25
    #     Basis is [0,0,0]
    #     Key 1 is [1,1,1] and is relative to Basis
    #     When applied,
    #         Basis becomes Basis + (Key 1 - Basis) * 0.25 * 0.25 = [0.0625,0.0625,0.0625]
    #         Key 1 becomes Key 1 - (Key 1 - Basis) = [0,0,0]
    def non_zero_value_and_vertex_group(self, object_name):
        obj = self.set_active_by_name(object_name)
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Basis')
        self.assertShapeKeysEqual(obj, [1, 1, 1], 'Key 1')
        self.assertShapeKeyValueEquals(obj, 0.25, 'Key 1')
        self.assertShapeKeyVertexGroupNameEquals(obj, 'Group', 'Key 1')

        self.assertEqual(bpy.ops.cats_shapekey.shape_key_to_basis(), {'FINISHED'})

        self.assertShapeKeysEqual(obj, [0.0625, 0.0625, 0.0625], 'Basis')
        self.assertShapeKeysEqual(obj, [0, 0, 0], 'Key 1 - Reverted')
        self.assertShapeKeyVertexGroupNameEquals(obj, 'Group', 'Key 1 - Reverted')

        return obj

    def test_non_zero_value_and_vertex_group_individual(self):
        self.non_zero_value_and_vertex_group('12A_ValueAndVertexGroup_Individual')

    # 10 extra shape keys all relative to the Basis to ensure the functions that are faster with many shape keys are used
    def test_non_zero_value_and_vertex_group_multi(self):
        obj = self.non_zero_value_and_vertex_group('12B_ValueAndVertexGroup_Multi')
        # Keys 2 to 11 should all be [0.0625,0.0625,0.0625]
        self.assertShapeKeysEqual(obj, [0.0625, 0.0625, 0.0625], *['Key ' + str(i) for i in range(2, 12)])


suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAddon)
runner = unittest.TextTestRunner()
ret = not runner.run(suite).wasSuccessful()
sys.exit(ret)
