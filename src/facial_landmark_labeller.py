from typing import Dict, List

# ----------------
# Landmark mapping
# ----------------

# Index to string name mapping
# Only feature-relevant landmarks are named here, but the CNN predicts all 48
LANDMARK_INDEX_MAP: Dict[int, str] = {
    24: "left_ear_tip",
    25: "left_ear_inner_middle",
    23: "left_ear_outer_middle",
    26: "left_ear_inner_base",
    22: "left_ear_outer_base",
    29: "right_ear_tip",
    28: "right_ear_inner_middle",
    30: "right_ear_outer_middle",
    27: "right_ear_inner_base",
    31: "right_ear_outer_base",
    6: "left_eye_middle_top",
    7: "left_eye_middle_bottom",
    5: "left_eye_inner_corner",
    4: "left_eye_outer_corner",
    10: "right_eye_middle_top",
    11: "right_eye_middle_bottom",
    9: "right_eye_inner_corner",
    8: "right_eye_outer_corner",
}

# String to index mapping
LANDMARK_NAME_MAP: Dict[str,int] = {}
for index, name in LANDMARK_INDEX_MAP.items():
    LANDMARK_NAME_MAP[name] = index

# --------
# functions 
# --------
def get_landmark_name(index:int) -> str:
    return LANDMARK_INDEX_MAP[index]

def get_landmark_index(name:str) -> int:
    return LANDMARK_NAME_MAP[name]

def get_defined_indices() -> List[int]:
    return list(LANDMARK_INDEX_MAP.keys())