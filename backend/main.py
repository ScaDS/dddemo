from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
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

# Serve static dataset images to frontend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
app.mount("/dataset", StaticFiles(directory=DATASET_DIR), name="dataset")

# Initialize global model
model = BabyModel()

@app.post("/train/")
def load_model(styles: List[str] = Query(...)):
    """
    Load a pretrained model for the selected styles.
    Example: /train/?styles=real&styles=cartoon
    """
    try:
        model.load_pretrained_model(styles)
        return {"message": f"Loaded model for: {sorted(styles)}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict the label of the uploaded image using the currently loaded model.
    """
    img = await read_imagefile(file)
    tensor = preprocess_image(img)
    label = model.predict(tensor)
    return {"label": label}

@app.post("/reset")
def reset_model():
    """
    Reset the model to untrained state.
    """
    model.reset()
    return {"message": "Model reset."}

@app.get("/trained_styles")
def get_loaded_styles():
    """
    Return the currently loaded style list (alphabetical).
    """
    return {"loaded_styles": model.loaded_styles}
