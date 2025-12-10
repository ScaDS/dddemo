import numpy as np
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

IMG_SIZE = (224, 224)

async def read_imagefile(file) -> Image.Image:
    image = await file.read()
    return Image.open(BytesIO(image)).convert("RGB")

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]) 
    return transform(image)