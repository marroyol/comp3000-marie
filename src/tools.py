import os
import math

def find_matching_image(label_filename, image_dir):
    """The json labels match the image labels so we can match them together using this function"""
    base_name = os.path.splitext(label_filename)[0]
    possible_extensions = [
        ".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"
    ]
        
    for ext in possible_extensions:
        image_path = os.path.join(image_dir, base_name + ext)
        if os.path.exists(image_path):
            return image_path
    return None

def compute_angle(vertex, point_a, point_b):
    """Attempting to get angles using dot product"""

    vx, vy = vertex
    ax, ay = point_a
    bx, by = point_b

    vector_a = [ax - vx, ay - vy]
    vector_b = [bx - vx, by - vy]

    dot_product = (vector_a[0] * vector_b[0]) + (vector_a[1]*vector_b[1])

    norm_a = math.sqrt(vector_a[0]**2 + vector_a[1]**2)
    norm_b = math.sqrt(vector_b[0]**2 + vector_b[1]**2)

    if norm_a == 0 or norm_b == 0:
        return float("nan")

    cos_theta = dot_product / (norm_a * norm_b)
    cos_theta = max(-1.0, min(1.0, cos_theta)) # to prevent cos_theta crashing
    # it's in degrees to match the literature (Evangelista)
    theta = math.degrees(math.acos(cos_theta))

    return theta

def midpoint(point_a, point_b):
    ax, ay = point_a
    bx, by = point_b
    return ((ax+bx)/2.0, (ay+by/2.0))