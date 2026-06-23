import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# DATASET PATH
# =========================================================

# Dataset folder containing:
# data/
#    images/
#    labels/

# dataset_path = BASE_DIR / "dataset" / "detection" / "dataset"
dataset_path = BASE_DIR / "segmentation" / "train"

images_path = dataset_path
labels_path = dataset_path

# =========================================================
# OUTPUT SPLIT FOLDER
# =========================================================

# output_path = BASE_DIR / "dataset" / "detection"
output_path = BASE_DIR / "segmentation_split"

# =========================================================
# SPLIT RATIOS
# =========================================================

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

splits = ["train", "val", "test"]

for split in splits:

    (output_path / split / "images").mkdir(parents=True, exist_ok=True)

    (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

# =========================================================
# GET IMAGE FILES
# =========================================================

image_extensions = [".jpg", ".jpeg", ".png"]

image_files = [
    f for f in images_path.iterdir()
    if f.suffix.lower() in image_extensions
]

# =========================================================
# SHUFFLE DATA
# =========================================================

random.shuffle(image_files)

total_images = len(image_files)

print(f"\nTotal Images Found: {total_images}")

# =========================================================
# CALCULATE SPLITS
# =========================================================

train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)

train_files = image_files[:train_count]

val_files = image_files[
    train_count:train_count + val_count
]

test_files = image_files[
    train_count + val_count:
]

# =========================================================
# FUNCTION TO COPY FILES
# =========================================================

def copy_files(file_list, split_name):

    saved = 0

    for image_file in file_list:

        image_name = image_file.stem

        # Image path
        src_image = image_file

        # Corresponding label
        src_label = labels_path / (image_name + ".txt")

        # Destination image
        dst_image = output_path / split_name / "images" / image_file.name

        # Destination label
        dst_label = output_path / split_name / "labels" / (image_name + ".txt")

        # Copy image
        shutil.copy2(src_image, dst_image)

        # Copy label if exists
        if src_label.exists():
            shutil.copy2(src_label, dst_label)

        saved += 1

    print(f"{split_name.upper()} -> {saved} files copied")


# =========================================================
# COPY DATA
# =========================================================

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

# =========================================================
# SUMMARY
# =========================================================

print("\n===================================")
print("DATASET SPLIT COMPLETED")
print("===================================")

print(f"Train Images : {len(train_files)}")
print(f"Val Images   : {len(val_files)}")
print(f"Test Images  : {len(test_files)}")

print("\nSaved at:")
print(output_path)