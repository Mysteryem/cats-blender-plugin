# GPL License

# This is directly taken from the export_fbx_bin.py to change it via monkey patching
import bpy
from . import common as Common
from io_scene_fbx import fbx_utils

def start_patch_fbx_exporter_timer():
    pass

def time_patch_fbx_exporter():
    import time
    found_scene = False
    while not found_scene:
        if hasattr(bpy.context, 'scene'):
            found_scene = True
        else:
            time.sleep(0.5)

    patch_fbx_exporter()

def patch_fbx_exporter():
    fbx_utils.get_bid_name = get_bid_name

# Blender-specific key generators - monkeypatched to force name if present
def get_bid_name(bid):
    if isinstance(bid, bpy.types.ID) and 'catsForcedExportName' in bid:
        return bid['catsForcedExportName']
    library = getattr(bid, "library", None)
    if library is not None:
        return "%s_L_%s" % (bid.name, library.name)
    else:
        return bid.name
