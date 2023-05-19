# GPL License

import re
import os
import bpy
from bpy.types import (
    Operator,
)
import json
import pathlib
import traceback
import requests.exceptions

from datetime import datetime, timezone
from functools import partial
from typing import Callable, Optional, Iterable, Union

from . import common as Common
from .register import register_wrap
from .. import globs
from ..extern_tools.google_trans_new.google_trans_new import google_translator
from .translations import t

from mmd_tools_local import translations as mmd_translations


# Each character range in the regex corresponds to a Unicode block.
JP_CHARS_PATTERN = re.compile(
    "["
    "\u3000-\u303f"  # CJK Symbols and Punctuation
    "\u3040-\u3094"  # Hiragana
    "\u30a0-\u30ff"  # Katakana
    # Extra glyphs for the Ainu language and to describe precise pronunciation (not used in normal writing).
    # "\u31f0-\u31ff"  # Katakana Phonetic Extensions
    "\u3400-\u4dbf"  # CJK Unified Ideographs Extension A
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\uff00-\uff9f"  # Fullwidth and Halfwidth Forms (up to the end of the halfwidth katakana)
    # Extra glyphs that seem to contain the small forms of the remaining kana not in other Unicode blocks.
    # "\U0001b130-\U0001b16f"  # Small Kana Extension Unicode Block
    # Currently we're only using Extensions A, it would be worth investigating if the other blocks should be used too.
    # "\U00020000-\U000323AF"  # CJK Unified Ideographs Extensions B to H and CJK Compatibility Ideographs Supplement
    "]+")
"""Regex to look for japanese chars"""

# Google Translate has issues translating single characters of small katakana/hiragana.
# Characters from the "Small Kana Extension" Unicode block are not included because they are not supported by most
# applications.
SMALL_JP_CHARS_TO_FULL = {
    # Katakana Unicode Block:
    # These characters sometimes translate and sometimes raise an error because Google Translate returns a null
    # response.
    # They are usually used to indicate modified pronunciation of the preceding character.
    "ァ": "ア",  # A:      \u30A1 -> \u30A2
    "ィ": "イ",  # I:      \u30A3 -> \u30A4
    "ゥ": "ウ",  # U:      \u30A5 -> \u30A6
    "ェ": "エ",  # E:      \u30A7 -> \u30A8
    "ォ": "オ",  # O:      \u30A9 -> \u30AA
    "ッ": "ツ",  # Tu/Tsu: \u30C3 -> \u30C4
    "ャ": "ヤ",  # Ya:     \u30E3 -> \u30E4
    "ュ": "ユ",  # Yu:     \u30E5 -> \u30E6
    "ョ": "ヨ",  # Yo:     \u30E7 -> \u30E8
    "ヮ": "ワ",  # Wa:     \u30EE -> \u30EF
    "ヵ": "カ",  # Ka:     \u30F5 -> \u30AB
    "ヶ": "ケ",  # Ke:     \u30F6 -> \u30B1
    #
    # Halfwidth and Fullwidth Forms Unicode Block:
    # These appear to translate without issue when on their own, though they will be replaced with their normal
    # fullwidth forms by fix_jp_chars anyway, so there's no need to include them here.
    #
    # Katakana Phonetic Extensions Unicode Block:
    # These are not normally used in writing and do not translate. Google Translate appears to treat them like symbols.
    #
    # Hiragana Unicode Block:
    # These characters sometimes translate and sometimes raise an error because Google Translate returns a null
    # response.
    # They are usually used to indicate modified pronunciation of the preceding character.
    "ぁ": "あ",  # A:      \u3041 -> \u3042
    "ぃ": "い",  # I:      \u3043 -> \u3044
    "ぅ": "う",  # U:      \u3045 -> \u3046
    "ぇ": "え",  # E:      \u3047 -> \u3048
    "ぉ": "お",  # O:      \u3049 -> \u304A
    "っ": "つ",  # Tu/Tsu: \u3063 -> \u3064
    "ゃ": "や",  # Ya:     \u3083 -> \u3084
    "ゅ": "ゆ",  # Yu:     \u3085 -> \u3086
    "ょ": "よ",  # Yo:     \u3087 -> \u3088
    "ゎ": "わ",  # Wa:     \u308E -> \u308F
    "ゕ": "か",  # Ka:     \u3095 -> \u304B
    "ゖ": "け",  # Ke:     \u3096 -> \u3051
}
"""Mapping from small kana glyphs to their equivalent fullwidth glyphs."""


dictionary = {}
dictionary_google = {}

main_dir = pathlib.Path(os.path.dirname(__file__)).parent.resolve()
resources_dir = os.path.join(str(main_dir), "resources")
dictionary_file = os.path.join(resources_dir, "dictionary.json")
dictionary_google_file = os.path.join(resources_dir, "dictionary_google.json")


class CatsTranslationError(RuntimeError):
    """Simple exception class for wrapping errors that occur when translating."""
    pass


@register_wrap
class TranslateShapekeyButton(Operator):
    bl_idname = 'cats_translate.shapekeys'
    bl_label = t('TranslateShapekeyButton.label')
    bl_description = t('TranslateShapekeyButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        saved_data = Common.SavedData()

        to_translate = []
        for mesh in Common.get_meshes_objects(mode=2):
            if Common.has_shapekeys(mesh):
                for shapekey in mesh.data.shape_keys.key_blocks:
                    if 'vrc.' not in shapekey.name and shapekey.name not in to_translate:
                        to_translate.append(shapekey.name)

        update_dictionary(to_translate, translating_shapes=True, self=self)

        Common.update_shapekey_orders()

        i = 0
        for mesh in Common.get_meshes_objects(mode=2):
            if Common.has_shapekeys(mesh):
                for shapekey in mesh.data.shape_keys.key_blocks:
                    if 'vrc.' not in shapekey.name:
                        shapekey.name, translated = translate(shapekey.name, add_space=True, translating_shapes=True)
                        if translated:
                            i += 1

        Common.ui_refresh()

        saved_data.load()

        self.report({'INFO'}, t('TranslateShapekeyButton.success', number=str(i)))
        return {'FINISHED'}


@register_wrap
class TranslateBonesButton(Operator):
    bl_idname = 'cats_translate.bones'
    bl_label = t('TranslateBonesButton.label')
    bl_description = t('TranslateBonesButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        if not Common.get_armature():
            return False
        return True

    def execute(self, context):
        to_translate = []
        for armature in Common.get_armature_objects():
            for bone in armature.data.bones:
                to_translate.append(bone.name)

        update_dictionary(to_translate, self=self)

        count = 0
        for armature in Common.get_armature_objects():
            for bone in armature.data.bones:
                bone.name, translated = translate(bone.name)
                if translated:
                    count += 1

        self.report({'INFO'}, t('TranslateBonesButton.success', number=str(count)))
        return {'FINISHED'}


@register_wrap
class TranslateObjectsButton(Operator):
    bl_idname = 'cats_translate.objects'
    bl_label = t('TranslateObjectsButton.label')
    bl_description = t('TranslateObjectsButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        to_translate = []
        objects = context.view_layer.objects
        for obj in objects:
            if obj.name not in to_translate:
                to_translate.append(obj.name)
            if obj.type == 'ARMATURE':
                if obj.data and obj.data.name not in to_translate:
                    to_translate.append(obj.data.name)
                if obj.animation_data and obj.animation_data.action:
                    to_translate.append(obj.animation_data.action.name)

        update_dictionary(to_translate, self=self)

        i = 0
        for obj in objects:
            obj.name, translated = translate(obj.name)
            if translated:
                i += 1

            if obj.type == 'ARMATURE':
                if obj.data:
                    obj.data.name, translated = translate(obj.data.name)
                    if translated:
                        i += 1

                if obj.animation_data and obj.animation_data.action:
                    obj.animation_data.action.name, translated = translate(obj.animation_data.action.name)
                    if translated:
                        i += 1

        self.report({'INFO'}, t('TranslateObjectsButton.success', number=str(i)))
        return {'FINISHED'}


@register_wrap
class TranslateMaterialsButton(Operator):
    bl_idname = 'cats_translate.materials'
    bl_label = t('TranslateMaterialsButton.label')
    bl_description = t('TranslateMaterialsButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        saved_data = Common.SavedData()

        to_translate = []
        for mesh in Common.get_meshes_objects(mode=2):
            for matslot in mesh.material_slots:
                if matslot.name not in to_translate:
                    to_translate.append(matslot.name)

        update_dictionary(to_translate, self=self)

        i = 0
        for mesh in Common.get_meshes_objects(mode=2):
            Common.set_active(mesh)
            for index, matslot in enumerate(mesh.material_slots):
                mesh.active_material_index = index
                if bpy.context.object.active_material:
                    bpy.context.object.active_material.name, translated = translate(bpy.context.object.active_material.name)
                    if translated:
                        i += 1

        saved_data.load()
        self.report({'INFO'}, t('TranslateMaterialsButton.success', number=str(i)))
        return {'FINISHED'}


# @register_wrap
# class TranslateTexturesButton(Operator):
#     bl_idname = 'cats_translate.textures'
#     bl_label = t('TranslateTexturesButton.label')
#     bl_description = t('TranslateTexturesButton.desc')
#     bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}
#
#     def execute(self, context):
#         # It currently seems to do nothing. This should probably only added when the folder textures really get translated. Currently only the materials are important
#         self.report({'INFO'}, t('TranslateTexturesButton.success_alt'))
#         return {'FINISHED'}
#
#         translator = google_translator()
#
#         to_translate = []
#         for ob in Common.get_objects():
#             if ob.type == 'MESH':
#                 for matslot in ob.material_slots:
#                     for texslot in bpy.data.materials[matslot.name].texture_slots:
#                         if texslot:
#                             print(texslot.name)
#                             to_translate.append(texslot.name)
#
#         translated = []
#         try:
#             translations = translator.translate(to_translate, lang_tgt='en')
#         except SSLError:
#             self.report({'ERROR'}, t('TranslateTexturesButton.error.noInternet'))
#             return {'FINISHED'}
#
#         for translation in translations:
#             translated.append(translation)
#
#         i = 0
#         for ob in Common.get_objects():
#             if ob.type == 'MESH':
#                 for matslot in ob.material_slots:
#                     for texslot in bpy.data.materials[matslot.name].texture_slots:
#                         if texslot:
#                             bpy.data.textures[texslot.name].name = translated[i]
#                             i += 1
#
#         Common.unselect_all()
#
#         self.report({'INFO'}, t('TranslateTexturesButton.success', number=str(i)))
#         return {'FINISHED'}


@register_wrap
class TranslateAllButton(Operator):
    bl_idname = 'cats_translate.all'
    bl_label = t('TranslateAllButton.label')
    bl_description = t('TranslateAllButton.desc')
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        error_shown = False

        try:
            if Common.get_armature():
                bpy.ops.cats_translate.bones('INVOKE_DEFAULT')
        except RuntimeError as e:
            self.report({'ERROR'}, str(e).replace('Error: ', ''))
            error_shown = True

        try:
            bpy.ops.cats_translate.shapekeys('INVOKE_DEFAULT')
        except RuntimeError as e:
            if not error_shown:
                self.report({'ERROR'}, str(e).replace('Error: ', ''))
                error_shown = True

        try:
            bpy.ops.cats_translate.objects('INVOKE_DEFAULT')
        except RuntimeError as e:
            if not error_shown:
                self.report({'ERROR'}, str(e).replace('Error: ', ''))
                error_shown = True

        try:
            bpy.ops.cats_translate.materials('INVOKE_DEFAULT')
        except RuntimeError as e:
            if not error_shown:
                self.report({'ERROR'}, str(e).replace('Error: ', ''))
                error_shown = True

        if error_shown:
            return {'CANCELLED'}
        self.report({'INFO'}, t('TranslateAllButton.success'))
        return {'FINISHED'}


# Loads the dictionaries at the start of blender
def load_translations():
    global dictionary
    dictionary = {}
    temp_dict = {}
    dict_found = False

    # Load internal dictionary
    try:
        with open(dictionary_file, encoding="utf8") as file:
            temp_dict = json.load(file)
            dict_found = True
            # print('DICTIONARY LOADED!')
    except FileNotFoundError:
        print('DICTIONARY NOT FOUND!')
        pass
    except json.decoder.JSONDecodeError:
        print("ERROR FOUND IN DICTIONARY")
        pass

    # Load local google dictionary and add it to the temp dict
    try:
        with open(dictionary_google_file, encoding="utf8") as file:
            global dictionary_google
            dictionary_google = json.load(file)

            if 'created' not in dictionary_google \
                    or 'translations' not in dictionary_google \
                    or 'translations_full' not in dictionary_google:
                reset_google_dict()
            else:
                for name, trans in dictionary_google.get('translations').items():
                    if not name:
                        continue

                    if name in temp_dict.keys():
                        print(name, 'ALREADY IN INTERNAL DICT!')
                        continue

                    temp_dict[name] = trans

            # print('GOOGLE DICTIONARY LOADED!')
    except FileNotFoundError:
        print('GOOGLE DICTIONARY NOT FOUND!')
        reset_google_dict()
        pass
    except json.decoder.JSONDecodeError:
        print("ERROR FOUND IN GOOOGLE DICTIONARY")
        reset_google_dict()
        pass

    # Sort temp dictionary by lenght and put it into the global dict
    for key in sorted(temp_dict, key=lambda k: len(k), reverse=True):
        dictionary[key] = temp_dict[key]

    # for key, value in dictionary.items():
    #     print('"' + key + '" - "' + value + '"')

    return dict_found


class _JapaneseToEnglishTranslator:
    # All that would be needed to support other languages would be to add lang fields to this class or add lang
    # arguments to the functions.
    def __init__(self):
        self.translator = google_translator(url_suffix='com')

    def refresh_internal_translator(self):
        self.translator = google_translator(url_suffix='com')

    def translate(self, text: str, attempts: int = 3) -> Optional[str]:
        """Main translate function that makes the requests to Google Translate."""
        for token_tries in range(1, attempts + 1):
            try:
                return self.translator.translate(text, lang_src='ja', lang_tgt='en')
            except AttributeError as e:
                # If the translator wasn't able to create a stable connection to Google, just retry it again
                # This is an issue with Google since Nov 2020: https://github.com/ssut/py-googletrans/issues/234
                if token_tries < attempts:
                    print("RETRY", token_tries)
                    self.refresh_internal_translator()
                else:
                    # If it didn't work after 3 tries, just quit
                    # The response from Google was printed into "cats/resources/google-response.txt"
                    print("ERROR: GOOGLE API CHANGED!")
                    print(traceback.format_exc())
                    raise CatsTranslationError("Google API changed") from e

    def translate_gen(self, to_translate: Iterable[str], operator: Optional[Operator]) -> str:
        """Translation generator that yields translations until an error occurs."""
        try:
            for text in to_translate:
                yield self.translate(text)
        except CatsTranslationError as e:
            # If it didn't work even after retries, just quit
            # The response from Google was printed into "cats/resources/google-response.txt"
            if operator:
                operator.report({'ERROR'}, t("update_dictionary.error.apiChanged"))
            raise e
        except (requests.exceptions.ConnectionError, ConnectionRefusedError) as e:
            print("CONNECTION TO GOOGLE FAILED!")
            if operator:
                operator.report({'ERROR'}, t("update_dictionary.error.cantConnect"))
            raise CatsTranslationError("Connection to Google failed") from e
        except json.JSONDecodeError as e:
            if operator:
                print(traceback.format_exc())
                operator.report({'ERROR'}, "Either Google changed their API or you got banned from Google Translate"
                                           " temporarily!"
                                           "\nCats translated what it could with the local dictionary,"
                                           "\nbut you will have to try again later for the Google translations.")
            print("YOU GOT BANNED BY GOOGLE!")
            raise CatsTranslationError("Either Google API changed or banned from Google Translate temporarily") from e
        except RuntimeError as e:
            # TODO: Are there more specific RuntimeError subclasses we can catch other than just RuntimeError?
            error = Common.html_to_text(str(e))
            if 'Please try your request again later' in error:
                if operator:
                    operator.report({'ERROR'},
                                    t('update_dictionary.error.temporaryBan') + t('update_dictionary.error.catsTranslated'))
                print('YOU GOT BANNED BY GOOGLE!')
                raise CatsTranslationError("Banned from Google Translate temporarily") from e
            if 'Error 403' in error:
                if operator:
                    operator.report({'ERROR'},
                                    t('update_dictionary.error.cantAccess') + t('update_dictionary.error.catsTranslated'))
                print('NO PERMISSION TO USE GOOGLE TRANSLATE!')
                raise CatsTranslationError("No permission to use Google translate") from e
            if operator:
                operator.report({'ERROR'},
                                t('update_dictionary.error.errorMsg') + t('update_dictionary.error.catsTranslated')
                                + '\n' + '\nGoogle: ' + error)
            print('You got an error message from Google: ', error)
            raise CatsTranslationError from e

    def translate_until_error(self, to_translate: Iterable[str], operator: Optional[Operator]) -> list[str]:
        """Translate the input strings until an error occurs.
        Returns a list of all the successful translations."""
        translations: list[str] = []
        try:
            for translated in self.translate_gen(to_translate, operator):
                translations.append(translated.strip())
        except CatsTranslationError:
            pass
        return translations


def _save_google_dict():
    # TODO: Open and write the file on another thread, though lock until done in-case a second call to save is made
    #  while still saving. If there's already one call waiting in queue, additional calls could be rejected.
    #  Sharing the lock with functions that can modify dictionary_google may also be necessary.
    #  Should be able to do some of the threading with queue.get(), blocking until the queue has an IO function to call.
    with open(dictionary_google_file, 'w', encoding="utf8") as outfile:
        json.dump(dictionary_google, outfile, ensure_ascii=False, indent=4)


def _update_google_only_dictionary(input_strings: Iterable[str], translated_strings: Iterable[str]):
    google_dict = dictionary_google['translations_full']
    updated = False
    for input_string, translated_string in zip(input_strings, translated_strings):
        updated = True
        google_dict[input_string] = translated_string
        print(input_string, '->', translated_string)
    if updated:
        # Save the google dict locally
        _save_google_dict()


def _update_translation_dictionaries(input_strings: Iterable[str], translated_strings: Iterable[str]):
    global dictionary
    # Update the dictionaries
    updated = False
    google_dict = dictionary_google['translations']
    for input_string, translated_string in zip(input_strings, translated_strings):
        updated = True
        # Capitalize words
        translation_words = translated_string.split(' ')
        translation_words = [word.capitalize() for word in translation_words]
        translated_string = ' '.join(translation_words)

        dictionary[input_string] = translated_string
        google_dict[input_string] = translated_string

        print(input_string, '->', translated_string)
    if updated:
        # Replace dictionary with a newly sorted copy
        sorted_keys_length_decreasing = sorted(dictionary, key=len, reverse=True)
        dictionary = {key: dictionary[key] for key in sorted_keys_length_decreasing}
        # Save the google dict locally
        _save_google_dict()


def _make_fix_jp_chars() -> Callable[[str], str]:
    """Function for creating to fix_jp_chars function to avoid adding a bunch of variables to the module globals by
    making them local to the function call instead."""
    # mmd_translations.jp_half_to_full_tuples is a tuple of tuple-pairs, but we need a dict for fast lookup of the
    # replacements for matches.
    half_to_full_dict = dict(mmd_translations.jp_half_to_full_tuples)

    # Compile regex that simply matches every key in the dict
    pattern = re.compile("|".join(re.escape(key) for key in half_to_full_dict))

    def callback(match_):
        """Get the first match and look up and return its replacement."""
        return half_to_full_dict[match_[0]]

    # Return a partial function of pattern.sub with callback bound to the first argument.
    return partial(pattern.sub, callback)


fix_jp_chars = _make_fix_jp_chars()
"""Convert half-width character strings in the input string to their equivalent full-width characters"""
del _make_fix_jp_chars


def _fix_to_translate_input(to_translate: Iterable[str]) -> str:
    for s in to_translate:
        # Replace halfwidth characters with fullwidth characters.
        s = fix_jp_chars(s)
        # Google Translate has issues translating single characters of small hiragana/katakana, sometimes returning
        # a null response, but sometimes working as expected.
        if len(s) == 1 and s in SMALL_JP_CHARS_TO_FULL:
            yield SMALL_JP_CHARS_TO_FULL[s]
        else:
            yield s


def _make_google_input_google_only(to_translate_iter: Iterable[str]) -> set[str]:
    """Filter strings to be translated by Google Translate only, filtering out any that already have translations.

    Strings containing a mix of Japanese (as defined by JP_CHARS_PATTERN) and non-Japanese will not be separated into
    parts, and will be translated as one whole string if they are not already in the dictionary."""
    google_full_dict = dictionary_google['translations_full']
    google_input = set()
    for s in _fix_to_translate_input(to_translate_iter):
        # If it's not already translated and it contains some Japanese characters
        if s not in google_full_dict and s not in google_input and JP_CHARS_PATTERN.search(s):
            google_input.add(s)
    return google_input


def _make_google_input(to_translate_iter: Iterable[str]) -> set[str]:
    """Filter strings to be translated by the internal dictionary, filtering out any that already have translations.

    Strings containing a mix of Japanese (as defined by JP_CHARS_PATTERN) and non-Japanese will have the consecutive
    Japanese parts separated into individual parts to translate. Parts that are already in the dictionary will be
    excluded."""
    google_input = set()
    for s in _fix_to_translate_input(to_translate_iter):
        if s in dictionary or s in google_input:
            continue

        length = len(s)
        translated_count = 0

        for key, value in dictionary.items():
            if key in s:
                if value:
                    s = s.replace(key, value)
                    # Check if string is fully translated
                    translated_count += len(key)
                    if translated_count >= length:
                        break

        # If not fully translated, translate the rest with Google
        if translated_count < length:
            match = JP_CHARS_PATTERN.findall(s)
            if match:
                for name in match:
                    if name not in google_input and name not in dictionary:
                        google_input.add(name)
    return google_input


def update_dictionary(to_translate_list: Union[str, Iterable[str]],
                      translating_shapes: bool = False,
                      self: Optional[Operator] = None
                      ):
    use_google_only = translating_shapes and bpy.context.scene.use_google_only

    # Check if single string is given and put it into a list
    if isinstance(to_translate_list, str):
        to_translate_list = [to_translate_list]

    # Filter out any already translated strings
    if use_google_only:
        google_input = _make_google_input_google_only(to_translate_list)
    else:
        google_input = _make_google_input(to_translate_list)

    if not google_input:
        # print('NO GOOGLE TRANSLATIONS')
        return

    # Convert the google input to a list to ensure that iteration order is stable. Iterating the set multiple times
    # should iterate in the same order each time, but this is not well-defined behaviour.
    google_input = list(google_input)

    # Translate the rest with google translate
    print('GOOGLE DICT UPDATE!')
    translations = _JapaneseToEnglishTranslator().translate_until_error(google_input, self)

    if not translations:
        print('DICTIONARY UPDATE FAILED!')
        return

    # Update the dictionaries for what got translated, even if translation failed part way through.
    if use_google_only:
        _update_google_only_dictionary(google_input, translations)
    else:
        _update_translation_dictionaries(google_input, translations)

    if len(translations) < len(google_input):
        # Translation failed part way through google_input
        print('DICTIONARY UPDATE PARTIALLY SUCCEEDED!')
    else:
        print('DICTIONARY UPDATE SUCCEEDED!')


def translate(to_translate: str, add_space: bool = False, translating_shapes: bool = False) -> tuple[str, bool]:
    global dictionary

    pre_translation = to_translate
    length = len(to_translate)
    translated_count = 0

    # Figure out whether to use google only or not
    use_google_only = False
    if translating_shapes and bpy.context.scene.use_google_only:
        use_google_only = True

    # Add space for shape keys
    addition = ''
    if add_space:
        addition = ' '

    # Convert half chars into full chars
    to_translate = fix_jp_chars(to_translate)
    # Convert a single small char into its full char
    if len(to_translate) == 1 and to_translate in SMALL_JP_CHARS_TO_FULL:
        to_translate = SMALL_JP_CHARS_TO_FULL[to_translate]

    # Translate shape keys with Google Translator only, if the user chose this
    if use_google_only:
        value = dictionary_google.get('translations_full').get(to_translate)
        if value:
            to_translate = value

    # Translate with internal dictionary
    else:
        value = dictionary.get(to_translate)
        if value:
            to_translate = value
        else:
            for key, value in dictionary.items():
                if key in to_translate:
                    # If string is empty, don't replace it. This will be done at the end
                    if not value:
                        continue

                    to_translate = to_translate.replace(key, addition + value)

                    # Check if string is fully translated
                    translated_count += len(key)
                    if translated_count >= length:
                        break

    to_translate = to_translate.replace('.L', '_L').replace('.R', '_R').replace('  ', ' ').replace('し', '').replace('っ', '').strip()

    # print('"' + pre_translation + '"')
    # print('"' + to_translate + '"')

    return to_translate, pre_translation != to_translate


def reset_google_dict():
    global dictionary_google

    now_utc = datetime.now(timezone.utc).strftime(globs.time_format)

    dictionary_google = {
        'created': now_utc,
        'translations': {},
        'translations_full': {},
    }

    _save_google_dict()
    print('GOOGLE DICT RESET')
