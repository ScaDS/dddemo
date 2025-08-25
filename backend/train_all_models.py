import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

# Configuration
STYLES = ['real', 'cartoon', 'sketch', 'edge', 'blur']
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
SUMMARY_FILE = os.path.join(MODEL_DIR, "training_summary.txt")
BATCH_SIZE = 50
EPOCHS = 20
IMG_SIZE = (224, 224) 
LEARNING_RATE = 0.0001

os.makedirs(MODEL_DIR, exist_ok=True)

# Model Definition: ResNet18 + Custom Head
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base.fc = nn.Linear(self.base.fc.in_features, 2)

    def forward(self, x):
        return self.base(x)

# Image Transform for ResNet
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Utility Functions
def get_combination_name(styles):
    return "_".join(sorted(styles))

def get_dataset_for_styles(styles, base_dir):
    datasets_list = []
    for style in styles:
        path = os.path.join(base_dir, style)
        if os.path.exists(path):
            datasets_list.append(datasets.ImageFolder(path, transform=transform))
    if not datasets_list:
        raise ValueError(f"No datasets found in {base_dir} for styles: {styles}")
    return ConcatDataset(datasets_list)

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

def train_model(styles):
    name = get_combination_name(styles)
    print(f"\n Training model for: {name}")

    # Load training and validation datasets
    train_dataset = get_dataset_for_styles(styles, TRAIN_DIR)
    val_dataset = get_dataset_for_styles(styles, TEST_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model and optimizer
    model = CNNModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Using device:", device)
    if device.type == 'cuda':
        print("Device name:", torch.cuda.get_device_name(0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"{name} | Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, f"model_{name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Evaluate
    accuracy = evaluate_model(model, val_loader, device)
    print(f"Validation Accuracy for {name}: {accuracy:.2%}")

    # Log accuracy
    with open(SUMMARY_FILE, "a") as f:
        f.write(f"{name}: {accuracy:.4f}\n")

def all_style_combinations():
    for r in range(1, len(STYLES) + 1):
        for combo in itertools.combinations(STYLES, r):
            yield combo

if __name__ == "__main__":
    # Clear log before new run
    with open(SUMMARY_FILE, "w") as f:
        f.write("Validation Accuracy per Style Combination\n\n")

    for style_combo in all_style_combinations():
        train_model(style_combo)
