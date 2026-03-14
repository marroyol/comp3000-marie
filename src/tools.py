import os

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