from typing import Dict, List

# ----------------
# Landmark mapping
# ----------------

# Index to string name mapping
LANDMARK_INDEX_MAP = Dict[int, str] = {
    24: "left_ear_tip",
    29: "right_ear_tip",
}

# String to index mapping
LANDMARK_NAME_MAP = Dict[int, str] = {}
for index, name in LANDMARK_INDEX_MAP.items():
    LANDMARK_NAME_MAP[name] = index