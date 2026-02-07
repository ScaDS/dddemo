import os
import hashlib
import torch
from backend.model import CNNModel

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def get_file_checksum(filepath):
    """Calculate MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()

def get_model_summary(model):
    """Get a summary of model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "layers": len(list(model.modules()))
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")])
    
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        checksum = get_file_checksum(model_path)
        
        print(f"{'='*60}")
        print(f"Model: {model_file}")
        print(f"MD5 Checksum: {checksum}")
        
        try:
            model = CNNModel().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            summary = get_model_summary(model)
            print(f"Total Parameters: {summary['total_params']:,}")
            print(f"Trainable Parameters: {summary['trainable_params']:,}")
            print(f"Number of Modules: {summary['layers']}")
        except Exception as e:
            print(f"Error loading model: {e}")
        
        print()

if __name__ == "__main__":
    main()
