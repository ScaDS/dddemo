import os
import itertools
import warnings
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")

# Configuration
STYLES = ['real', 'cartoon', 'sketch', 'edge', 'blur']
IMG_SIZE = (224, 224)
BATCH_SIZE = 50
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# Resolve paths assuming we run from inside backend/
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)
MODEL_DIR = os.path.join(BACKEND_DIR, "models")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUT_XLSX = os.path.join(MODEL_DIR, "models_performances.xlsx")

os.makedirs(MODEL_DIR, exist_ok=True)

from model import CNNModel

# Transform: (224 + ImageNet normalization)
print("Preparing test loaders (TRAIN-matched: 224 + ImageNet normalization)...")
eval_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Helpers
def combo_name(styles):
    return "_".join(sorted(styles))

def model_path_for(styles):
    return os.path.join(MODEL_DIR, f"model_{combo_name(styles)}.pt")

def iter_all_combinations(items):
    for r in range(1, len(items) + 1):
        for combo in itertools.combinations(items, r):
            yield list(combo)

def make_style_loader(style):
    style_dir = os.path.join(TEST_DIR, style)
    if not os.path.isdir(style_dir):
        raise FileNotFoundError(f"Missing test directory for style: {style_dir}")
    ds = datasets.ImageFolder(style_dir, transform=eval_transform)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return (correct / total) if total > 0 else 0.0

# Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        try:
            print("CUDA device:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    # Preload loaders (one per style)
    test_loaders = {style: make_style_loader(style) for style in STYLES}

    # Results table: columns = 5 styles + avg_accuracy
    results = pd.DataFrame(index=[], columns=STYLES + ["avg_accuracy"], dtype=float)
    results.index.name = "model_trained_on"

    # Evaluate all non-empty combinations (31)
    for styles in iter_all_combinations(STYLES):
        name = combo_name(styles)
        path = model_path_for(styles)

        if not os.path.exists(path):
            print(f"Missing model file, skipping: {os.path.basename(path)}")
            continue

        print(f"\nEvaluating model: {os.path.basename(path)}")
        model = CNNModel().to(device)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        row = {}
        for style in STYLES:
            acc = accuracy(model, test_loaders[style], device)
            row[style] = acc
            print(f"  - test on {style:7s}: {acc:.4f}")

        row["avg_accuracy"] = sum(row[s] for s in STYLES) / len(STYLES)
        results.loc[name] = row

    # Sort rows for readability
    results = results.reindex(
        sorted(results.index, key=lambda s: (s.count('_') + 1, s))
    )

    print(f"\nWriting Excel to: {OUT_XLSX}")
    results.to_excel(OUT_XLSX, sheet_name="accuracy")

    print("\nPreview:")
    with pd.option_context("display.max_rows", 10, "display.max_columns", None):
        print(results.head())

if __name__ == "__main__":
    main()
