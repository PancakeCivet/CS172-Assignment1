import os
import random
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw
import string

from cs172.utils import create_and_empty_folder, get_font


def generate_verification(
    min_spacing=5,
    font_size=(30, 40),
    font_type=None,
    save_folder=None,
    image_size=(260, 80),
    need_rotate=True,
):
    num = 5  # generate 5 digits
    width, height = image_size
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # random size font list for {num} digits
    fonts = [get_font(font_size, font_type) for _ in range(num)]

    # generate random {num} characters
    characters = random.choices(string.digits, k=num)

    # ====================== TO DO START ========================
    # generate random color list for each character
    # char_colors = [(r1, g1, b1), ..., (rk, gk, bk)]
    # RGB should be int and in (0, 200) in case it is too similar to white
    # ===========================================================
    char_colors = ...
    # ======================= TO DO END =========================

    # to set random char location
    # you may need char_width, char_height of each char
    char_width = [0] * num
    char_height = [0] * num
    for i, char in enumerate(characters):
        char_bbox = draw.textbbox((0, 0), char, font=fonts[i])
        char_width[i] = char_bbox[2] - char_bbox[0]
        char_height[i] = char_bbox[3] - char_bbox[1]

    # consider random rotate
    if need_rotate:
        char_images = ["_"] * num
        for i, char in enumerate(characters):
            # ====================== TO DO START ========================
            # you may need to create a new image to rotate the character
            # you may need to update char_width and char_height
            # Then paste the rotated character onto the main image (already implemented)
            # If you have other ways, it's fine
            # This may be a little difficult, the rotated sample only occur in last test data
            # ===========================================================
            char_images[i] = ...
            
            char_width[i] = ...
            char_height[i] = ...
            # ======================= TO DO END =========================

    
    # Get random character position (x, y)
    remaining_spacing = width - sum(char_width) - min_spacing * (num + 1)
    points = [0] + sorted([random.randint(0, remaining_spacing) for _ in range(num)])
    x = 0
    for i, char in enumerate(characters):
        x += min_spacing + points[i + 1] - points[i]
        y = random.randint(2*min_spacing, height - char_height[i] - 2*min_spacing)

        if need_rotate:
            # paste the rotated character onto the main image
            # If you have other ways, it's fine
            image.paste(char_images[i], (x, y), char_images[i])
            x += char_width[i]
        else:
            # put the char on the picture
            draw.text((x, y), char, fill=char_colors[i], font=fonts[i])
            x += char_width[i]

    # add random colored lines
    for _ in range(random.randint(0, 5)):  # random number of lines (0, 5)
        # ====================== TO DO START ========================
        # lines should have random color and random start, end position
        # ===========================================================
        line_color = ...
        start = ...
        end = ...
        # ======================= TO DO END =========================
        draw.line([start, end], fill=line_color, width=random.randint(1, 2))

    # add salt and pepper noise
    noise_raito = 0.02  # 2% of the pixels
    for _ in range(int(width * height * noise_raito)):
        # ====================== TO DO START ========================
        # noise should have random color and random position
        # ===========================================================
        x, y = ...
        color = ...
        # ======================= TO DO END =========================
        image.putpixel((x, y), color)

    # Add a black border
    # If you don't like it, it's okay to comment it
    border_width = 2
    draw.rectangle(
        [0, 0, width - 1, height - 1], outline="black", width=border_width
    )

    # save image if save_folder is given
    if save_folder:
        if not os.path.exists(save_folder):
            raise FileNotFoundError(
                f"The directory '{save_folder}' does not exist."
            )

        extension = ".jpg"
        base_filename = "".join(characters)
        counter = 0

        new_filename = f"{base_filename}#{counter}{extension}"
        filename = os.path.join(save_folder, new_filename)

        # in case same name
        while os.path.exists(filename):
            counter += 1
            new_filename = f"{base_filename}#{counter}{extension}"
            filename = os.path.join(save_folder, new_filename)

        image.save(filename)

    return image, characters


def save_certification_data(image_size, data_num, save_folder, need_rotate):
    
    create_and_empty_folder(save_folder)

    for _ in tqdm(range(data_num)):
        generate_verification(
            image_size=image_size,
            need_rotate=need_rotate,
            save_folder=save_folder,
        )
