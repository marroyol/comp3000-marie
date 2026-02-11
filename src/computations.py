from src.facial_landmark_labeller import get_landmark_index

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