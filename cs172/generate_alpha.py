import os
import random
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw
import string

from cs172.utils import create_and_empty_folder, get_font


def generate_verification_alpha(
    min_spacing=5,
    font_size=(30, 40),
    font_type=None,
    save_folder=None,
    image_size=(260, 80),
    need_rotate=True,
    enable_color=True,
    base_color=(0, 0, 0),
):
    num = 5  # 固定5个字符
    width, height = image_size
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 为每个字符随机选择字体大小
    fonts = [get_font(font_size, font_type) for _ in range(num)]

    # ============ 修改处：生成随机字母（大小写混合） ============
    # string.ascii_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    characters = random.choices(string.ascii_letters, k=num)
    # ===========================================================

    # 随机颜色
    if enable_color:
        char_colors = [
            (
                random.randint(0, 199),
                random.randint(0, 199),
                random.randint(0, 199),
            )
            for _ in range(num)
        ]
    else:
        char_colors = [base_color for _ in range(num)]

    # 计算每个字符的宽高
    char_width = [0] * num
    char_height = [0] * num
    for i, char in enumerate(characters):
        char_bbox = draw.textbbox((0, 0), char, font=fonts[i])
        char_width[i] = char_bbox[2] - char_bbox[0]
        char_height[i] = char_bbox[3] - char_bbox[1]

    # 旋转字符
    if need_rotate:
        char_images = ["_"] * num
        for i, char in enumerate(characters):
            tmp_w = max(char_width[i] + 20, fonts[i].size + 20)
            tmp_h = max(char_height[i] + 20, fonts[i].size + 20)
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

    # 计算字符位置
    remaining_spacing = width - sum(char_width) - min_spacing * (num + 1)
    points = [0] + sorted([random.randint(0, remaining_spacing) for _ in range(num)])
    x = 0
    for i, char in enumerate(characters):
        x += min_spacing + points[i + 1] - points[i]
        y = random.randint(2 * min_spacing, height - char_height[i] - 2 * min_spacing)

        if need_rotate:
            image.paste(char_images[i], (x, y), char_images[i])
            x += char_width[i]
        else:
            draw.text((x, y), char, fill=char_colors[i], font=fonts[i])
            x += char_width[i]

    # 随机干扰线
    for _ in range(random.randint(0, 5)):
        line_color = (
            random.randint(0, 199),
            random.randint(0, 199),
            random.randint(0, 199),
        )
        start = (random.randint(0, width - 1), random.randint(0, height - 1))
        end = (random.randint(0, width - 1), random.randint(0, height - 1))
        draw.line([start, end], fill=line_color, width=random.randint(1, 2))

    # 随机噪点
    noise_ratio = 0.02
    for _ in range(int(width * height * noise_ratio)):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        image.putpixel((x, y), color)

    # 边框
    border_width = 2
    draw.rectangle([0, 0, width - 1, height - 1], outline="black", width=border_width)

    # 保存
    if save_folder:
        if not os.path.exists(save_folder):
            raise FileNotFoundError(f"The directory '{save_folder}' does not exist.")

        extension = ".jpg"
        base_filename = "".join(characters)
        counter = 0
        new_filename = f"{base_filename}#{counter}{extension}"
        filename = os.path.join(save_folder, new_filename)

        while os.path.exists(filename):
            counter += 1
            new_filename = f"{base_filename}#{counter}{extension}"
            filename = os.path.join(save_folder, new_filename)

        image.save(filename)

    return image, characters


def save_certification_data_alpha(image_size, data_num, save_folder, need_rotate):
    create_and_empty_folder(save_folder)
    for _ in tqdm(range(data_num)):
        generate_verification_alpha(
            image_size=image_size,
            need_rotate=need_rotate,
            save_folder=save_folder,
        )
