import os


LIFT_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_ASSET_ZOO_PATH = os.path.join(LIFT_BASE_PATH, "assets")

OBJAVERSE_PATH = os.path.join(BASE_ASSET_ZOO_PATH, "objaverse")

AGOD_PATH = os.path.join(BASE_ASSET_ZOO_PATH, "AGOD")
# AGOD_PATH = "/workspace/realdex/AnyDexGrasp/agod_objects/AGOD"
AGOD_OBJECT_PATH = [os.path.join(AGOD_PATH, file) for file in os.listdir(AGOD_PATH)]
AGOD_OBJECT_PATH.sort()
