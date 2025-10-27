import torch
import numpy as np
import random
import os
import shutil
from PIL import Image, ImageFont

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_and_empty_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)


def get_font(font_size, font_type=None):
    """
    return font with random size
    """
    if font_type:
        try:
            font = ImageFont.truetype(
                font_type, size=random.randint(*font_size)
            )
        except IOError:
            raise ValueError("Font file not found, using default font.")
    else:
        font = ImageFont.load_default(size=random.randint(*font_size))
    return font


def merge_images(images, grid_size=(4, 4)):
    
    # final grid size
    size = images[0].size
    final_image_width = size[0] * grid_size[0]
    final_image_height = size[1] * grid_size[1]

    new_im = Image.new("RGB", (final_image_width, final_image_height))

    for index, im in enumerate(images):
        im = im.resize(size)
        x = index % grid_size[0] * size[0]
        y = index // grid_size[0] * size[1]
        new_im.paste(im, (x, y))

    return new_im