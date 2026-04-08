from src.facial_landmark_labeller import get_landmark_index
from src.tools import compute_angle
import math

EVANGELISTA_STATS = {
    "ear_tips_bases_ratio":{
        "control_mean": 2.85, "control_sd": 0.3,
        "painful_mean": 2.34, "painful_sd": 0.3,
    },
    "eye_height_width_ratio": {
        "control_mean": 0.79, "control_sd": 0.1,
        "painful_mean": 0.50, "painful_sd": 0.2,
    },
    "medial_ear_angle":{
        "control_mean": 126.5, "control_sd": 4.7,
        "painful_mean": 140.4, "painful_sd": 6.5,
    },
    "lateral_ear_angle": {
        "control_mean": 78.9, "control_sd": 3.1,
        "painful_mean": 68.5, "painful_sd": 5.9,
    },
}

'''
Note: Maybe consider combining these into 1 function?
'''
def compute_ear_tip_distance(points):
    left_index = get_landmark_index("left_ear_tip")
    right_index = get_landmark_index("right_ear_tip")

    # remember there are 2 numbers per point - x & y
    left_x, left_y = points[left_index]
    right_x, right_y = points[right_index]

    distance_x = left_x - right_x
    distance_y = left_y - right_y

    # distance formula
    total_distance = (distance_x ** 2 + distance_y ** 2) ** 0.5

    return total_distance

def compute_ear_base_distance(points):
    left_index = get_landmark_index("left_ear_inner_base")
    right_index = get_landmark_index("right_ear_inner_base")

    left_x, left_y = points[left_index]
    right_x, right_y = points[right_index]

    distance_x = left_x - right_x
    distance_y = left_y - right_y

    total_distance = (distance_x ** 2 + distance_y ** 2) ** 0.5

    return total_distance

def compute_ear_tips_bases_ratio(points):
    tip_distance = compute_ear_tip_distance(points)
    base_distance = compute_ear_base_distance(points)
    if base_distance == 0:
        return float("nan")
    return tip_distance / base_distance

def compute_eye_heights(points):

    # indexes
    left_top_index = get_landmark_index("left_eye_middle_top")
    right_top_index = get_landmark_index("right_eye_middle_top")
    
    left_bottom_index = get_landmark_index("left_eye_middle_bottom")
    right_bottom_index = get_landmark_index("right_eye_middle_bottom")

    # x & y

    left_top_x, left_top_y = points[left_top_index]
    left_bottom_x, left_bottom_y = points[left_bottom_index]

    right_top_x, right_top_y = points[right_top_index]
    right_bottom_x, right_bottom_y = points[right_bottom_index]

    # distances

    left_x = left_bottom_x - left_top_x
    left_y = left_bottom_y - left_top_y
    left_height = (left_x**2 + left_y**2) ** 0.5

    right_x = right_bottom_x - right_top_x
    right_y = right_bottom_y - right_top_y
    right_height = (right_x**2 + right_y**2) ** 0.5

    return left_height, right_height

def compute_eye_lengths(points):

    # indexes
    left_inner_index = get_landmark_index("left_eye_inner_corner")
    right_inner_index = get_landmark_index("right_eye_inner_corner")
    
    left_outer_index = get_landmark_index("left_eye_outer_corner")
    right_outer_index = get_landmark_index("right_eye_outer_corner")

    # x & y

    left_outer_x, left_outer_y = points[left_outer_index]
    left_inner_x, left_inner_y = points[left_inner_index]

    right_outer_x, right_outer_y = points[right_outer_index]
    right_inner_x, right_inner_y = points[right_inner_index]

    # distances

    left_x = left_inner_x - left_outer_x
    left_y = left_inner_y - left_outer_y
    left_length = (left_x**2 + left_y**2) ** 0.5

    right_x = right_inner_x - right_outer_x
    right_y = right_inner_y - right_outer_y
    right_length = (right_x**2 + right_y**2) ** 0.5

    return left_length, right_length

def compute_eye_height_width_ratio(points):
    left_height, right_height = compute_eye_heights(points)
    left_width, right_width = compute_eye_lengths(points)

    if left_width == 0 or right_width == 0:
        return float("nan")
    
    left_ratio = left_height / left_width
    right_ratio = right_height / right_width
    return (left_ratio + right_ratio) / 2.0

def compute_medial_angle(points):
    left_inner_base = points[get_landmark_index("left_ear_inner_base")]
    left_inner_middle = points[get_landmark_index("left_ear_inner_middle")]
    right_inner_base = points[get_landmark_index("right_ear_inner_base")]
    right_inner_middle = points[get_landmark_index("right_ear_inner_middle")]

    left_angle = compute_angle(
        vertex = left_inner_base,
        point_a = left_inner_middle,
        point_b = right_inner_base,
    )
    right_angle = compute_angle(
        vertex = right_inner_base,
        point_a = right_inner_middle,
        point_b = left_inner_base,
    )
    return (left_angle + right_angle) / 2.0

def compute_lateral_angle(points):

    left_outer_base = points[get_landmark_index("left_ear_outer_base")]
    left_outer_middle = points[get_landmark_index("left_ear_outer_middle")]
    right_outer_base = points[get_landmark_index("right_ear_outer_base")]
    right_outer_middle = points[get_landmark_index("right_ear_outer_middle")]

    left_angle = compute_angle(
        vertex=left_outer_base,
        point_a=left_outer_middle,
        point_b=right_outer_base,
    )

    right_angle = compute_angle(
        vertex=right_outer_base,
        point_a=right_outer_middle,
        point_b=left_outer_base,
    )
    
    return 180.0 - ((left_angle + right_angle) / 2.0)

def compute_all_features(points):
    return {
        "ear_tips_bases_ratio": compute_ear_tips_bases_ratio(points),
        "eye_height_width_ratio":compute_eye_height_width_ratio(points),
        "medial_ear_angle": compute_medial_angle(points),
        "lateral_ear_angle": compute_lateral_angle(points),
    }