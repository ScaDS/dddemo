import os
import cv2
import random
from tqdm import tqdm

# Configuration
RAW_DIR = "../dataset/raw"
OUTPUT_DIR = "../dataset"
IMG_SIZE = (128, 128)
CATEGORIES = ['real', 'cartoon', 'sketch', 'blur', 'edge']
ANIMALS = ['cat', 'dog']
TRAIN_PER_CATEGORY = 500
TEST_PER_CATEGORY = 100
TOTAL_PER_CATEGORY = TRAIN_PER_CATEGORY + TEST_PER_CATEGORY

# Ensure output directories exist
for split in ['train', 'test']:
    for category in CATEGORIES:
        for animal in ANIMALS:
            os.makedirs(os.path.join(OUTPUT_DIR, split, category, animal), exist_ok=True)

# Helper functions for filters
def apply_cartoon(img):
    img_color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, blockSize=9, C=2)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(img_color, img_edge)

def apply_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_blur(img):
    return cv2.GaussianBlur(img, (15, 15), 0)

def apply_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

FILTERS = {
    'cartoon': apply_cartoon,
    'sketch': apply_sketch,
    'blur': apply_blur,
    'edge': apply_edge,
}

def process_images():
    for animal in ANIMALS:
        # All available files for this animal
        all_files = [f for f in os.listdir(RAW_DIR) if f.startswith(animal)]
        assert len(all_files) >= TOTAL_PER_CATEGORY * len(CATEGORIES), \
            f"Not enough images for {animal} in raw folder."

        # Shuffle once and divide a unique chunk per category
        random.shuffle(all_files)
        chunks = [all_files[i * TOTAL_PER_CATEGORY:(i + 1) * TOTAL_PER_CATEGORY] for i in range(len(CATEGORIES))]

        for category, files in zip(CATEGORIES, chunks):
            for idx, filename in enumerate(tqdm(files, desc=f"{animal}-{category}")):
                img_path = os.path.join(RAW_DIR, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)

                # Apply filter if not real
                if category in FILTERS:
                    img = FILTERS[category](img)

                # Decide split
                split = 'train' if idx < TRAIN_PER_CATEGORY else 'test'
                out_name = f"{animal}_{category}_{idx}.jpg"
                out_path = os.path.join(OUTPUT_DIR, split, category, animal, out_name)
                cv2.imwrite(out_path, img)

if __name__ == "__main__":
    process_images()
