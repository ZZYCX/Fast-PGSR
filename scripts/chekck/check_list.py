import os

# ============================================================
# 配置区域
# ============================================================

# data 根目录
DATA_ROOT = "/media/ubuntu2/A/LWL/code/Train/Fast-PGSR/data"

# images 文件夹中的图片数量小于这个值时，输出场景名称
MIN_IMAGE_COUNT = 20

# 场景文件夹名称中必须包含
SCENE_KEYWORD = "_soybean"

# 支持统计的图片后缀
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# 主程序
# ============================================================

def main():
    for scene_name in sorted(os.listdir(DATA_ROOT)):
        scene_path = os.path.join(DATA_ROOT, scene_name)

        # 只处理文件夹
        if not os.path.isdir(scene_path):
            continue

        # 只处理 *_soybean* 场景
        if SCENE_KEYWORD not in scene_name:
            continue

        images_dir = os.path.join(scene_path, "images")

        # images 不存在时，按 0 张处理
        if not os.path.isdir(images_dir):
            image_count = 0
        else:
            image_count = sum(
                1
                for filename in os.listdir(images_dir)
                if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
            )

        # 图片数量不足时输出
        if image_count < MIN_IMAGE_COUNT:

            # 例如：
            # 2026_06_25_soybean31_0804
            # ->
            # soybean31_0804
            soybean_pos = scene_name.find("soybean")

            if soybean_pos != -1:
                output_name = scene_name[soybean_pos:]
            else:
                output_name = scene_name

            print(output_name)


if __name__ == "__main__":
    main()