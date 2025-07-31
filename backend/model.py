import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

# Base directory resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "backend", "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# === CNN Model Definition ===
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = resnet18(pretrained=False)  # Will load pretrained weights manually
        self.base.fc = nn.Linear(self.base.fc.in_features, 2)

    def forward(self, x):
        return self.base(x)

# === BabyModel Wrapper ===
class BabyModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNModel().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.loaded_styles = []  # Track currently loaded styles

    def load_pretrained_model(self, styles):
        if not styles:
            raise ValueError("At least one style must be provided.")

        # Sort styles to match saved model naming convention
        sorted_styles = sorted(styles)
        model_name = f"model_{'_'.join(sorted_styles)}.pt"
        model_path = os.path.join(MODEL_DIR, model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_name}' not found in '{MODEL_DIR}'.")
        
        print(f"📦 Loading model for styles: {sorted_styles}")
        print(f"📂 Model path: {model_path}")

        # Reinitialize model and load weights
        self.model = CNNModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.loaded_styles = sorted_styles
        print(f"✅ Loaded model: {model_name}")

    def predict(self, tensor):
        self.model.eval()
        with torch.no_grad():
            inputs = tensor.unsqueeze(0).to(self.device)
            outputs = self.model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            return ['cat', 'dog'][predicted.item()]

    def reset(self):
        self.__init__()  # Reinitialize everything
