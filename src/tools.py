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

def compute_angle(ax, ay, bx, by):
    """Attempting to get angles using dot product"""

    vector_a = [ax, ay]
    vector_b = [bx, by]

    dot_product = (vector_a[0] * vector_b[0]) + (vector_a[1]*vector_b[1])

    norm_a = math.sqrt(vector_a[0]**2 + vector_a[1]**2)
    norm_b = math.sqrt(vector_b[0]**2 + vector_b[1]**2)

    cos_theta = dot_product / (norm_a * norm_b)
    # it's in degrees to match the literature (Evangelista)
    theta = math.degrees(math.acos(cos_theta))

    return theta