from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from backend.model import BabyModel
from backend.utils import read_imagefile, preprocess_image

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static images from the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
app.mount("/dataset", StaticFiles(directory=DATASET_DIR), name="dataset")

# Initialize model
model = BabyModel()

@app.post("/train/")
def train_model(style: str):
    """
    Train the model using images from dataset/train/<style>/
    """
    style_path = os.path.join(TRAIN_DIR, style)
    model.train(style_path)
    return {"message": f"Model trained on {style} images."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict the label of the uploaded image using the trained model.
    """
    img = await read_imagefile(file)
    tensor = preprocess_image(img)
    # label, confidence = model.predict(tensor)
    # return {"label": label, "confidence": confidence}
    label = model.predict(tensor)
    return {"label": label}

@app.post("/reset")
def reset_model():
    """
    Reset the model to its untrained state.
    """
    model.reset()
    return {"message": "Model reset."}

@app.get("/trained_styles")
def get_styles():
    return {"trained_styles": sorted(model.learned_styles)}
