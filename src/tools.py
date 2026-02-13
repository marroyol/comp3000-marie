
'''
To do:
quick tool that matches image to json file (the file names match)
'''

import os

def find_matching_image(label_file_name, img_dir):
    base = os.path.splitext(label_file_name)[0]
    matching_image = os.path.join(img_dir, base + ".png")

    return matching_image