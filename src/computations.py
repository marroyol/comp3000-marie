from src.facial_landmark_labeller import get_landmark_index

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
    left_index = get_landmark_index("left_ear_base")
    right_index = get_landmark_index("right_ear_base")

    left_x, left_y = points[left_index]
    right_x, right_y = points[right_index]

    distance_x = left_x - right_x
    distance_y = left_y - right_y

    total_distance = (distance_x ** 2 + distance_y ** 2) ** 0.5

    return total_distance

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