import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 40 * 40, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class BabyModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ])
        self.learned_styles = set()  # Store trained styles

    def train(self, style='real'):
        self.learned_styles.add(style)  # accumulate styles

        # Combine datasets for all learned styles
        datasets = []
        for s in self.learned_styles:
            path = os.path.join(DATASET_DIR, "train", s)
            if os.path.exists(path):
                datasets.append(ImageFolder(path, transform=self.transform))

        if not datasets:
            print("No valid datasets found.")
            return

        combined = ConcatDataset(datasets)
        loader = DataLoader(combined, batch_size=16, shuffle=True)

        # Reinitialize model and optimizer for fresh training on all styles
        self.model = SimpleCNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(3):  # Adjust for performance
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    # def predict(self, tensor):
    #     self.model.eval()
    #     with torch.no_grad():
    #         inputs = tensor.unsqueeze(0).to(self.device)
    #         outputs = self.model(inputs)
    #         probs = torch.softmax(outputs, dim=1)
    #         confidence, predicted = torch.max(probs, 1)
    #         return ['cat', 'dog'][predicted.item()], round(confidence.item() * 100, 2)

    def predict(self, tensor):
        self.model.eval()
        with torch.no_grad():
            inputs = tensor.unsqueeze(0).to(self.device)
            outputs = self.model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            return ['cat', 'dog'][predicted.item()]

    def reset(self):
        self.__init__()  # clear all state
