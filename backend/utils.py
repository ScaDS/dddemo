import numpy as np
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

async def read_imagefile(file) -> Image.Image:
    image = await file.read()
    return Image.open(BytesIO(image)).convert("RGB")

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    return transform(image)