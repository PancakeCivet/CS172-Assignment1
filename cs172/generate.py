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
    char_colors = [
        (random.randint(0, 199), random.randint(0, 199), random.randint(0, 199))
        for _ in range(num)
    ]
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
            tmp_w = max(char_width[i] + 8, fonts[i].size + 8)
            tmp_h = max(char_height[i] + 8, fonts[i].size + 8)
            tmp_img = Image.new("RGBA", (tmp_w, tmp_h), (255, 255, 255, 0))
            tmp_draw = ImageDraw.Draw(tmp_img)
            offset_x = (tmp_w - char_width[i]) // 2
            offset_y = (tmp_h - char_height[i]) // 2
            tmp_draw.text(
                (offset_x, offset_y), char, fill=char_colors[i], font=fonts[i]
            )

            angle = random.uniform(-30, 30)
            rot_img = tmp_img.rotate(angle, expand=True, resample=Image.BICUBIC)

            max_w_per_char = max(1, (width - min_spacing * (num + 1)) // num)
            max_h_allowed = max(1, height - 4 * min_spacing)

            w, h = rot_img.size
            scale_w = min(1.0, max_w_per_char / w)
            scale_h = min(1.0, max_h_allowed / h)
            scale = min(scale_w, scale_h, 1.0)

            if scale < 1.0:
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                rot_img = rot_img.resize((new_w, new_h), Image.BICUBIC)

            char_images[i] = rot_img
            char_width[i], char_height[i] = rot_img.size
            # ======================= TO DO END =========================

    # Get random character position (x, y)
    remaining_spacing = width - sum(char_width) - min_spacing * (num + 1)
    remaining_spacing = max(0, remaining_spacing)
    points = [0] + sorted([random.randint(0, remaining_spacing) for _ in range(num)])
    x = 0
    for i, char in enumerate(characters):
        x += min_spacing + points[i + 1] - points[i]
        y = random.randint(
            2 * min_spacing,
            max(2 * min_spacing, height - char_height[i] - 2 * min_spacing),
        )

        if need_rotate:
            # paste the rotated character onto the main image
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
        line_color = (
            random.randint(0, 199),
            random.randint(0, 199),
            random.randint(0, 199),
        )
        start = (random.randint(0, width - 1), random.randint(0, height - 1))
        end = (random.randint(0, width - 1), random.randint(0, height - 1))
        # ======================= TO DO END =========================
        draw.line([start, end], fill=line_color, width=random.randint(1, 2))

    # add salt and pepper noise
    noise_raito = 0.02  # 2% of the pixels
    for _ in range(int(width * height * noise_raito)):
        # ====================== TO DO START ========================
        # noise should have random color and random position
        # ===========================================================
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        # ======================= TO DO END =========================
        image.putpixel((x, y), color)

    # Add a black border
    # If you don't like it, it's okay to comment it
    border_width = 2
    draw.rectangle([0, 0, width - 1, height - 1], outline="black", width=border_width)

    # save image if save_folder is given
    if save_folder:
        if not os.path.exists(save_folder):
            raise FileNotFoundError(f"The directory '{save_folder}' does not exist.")

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
