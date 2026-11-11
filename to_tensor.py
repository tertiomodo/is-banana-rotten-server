import io
from PIL import Image
import numpy as np


def image_to_tensor(image_bytes):
    file_like_object = io.BytesIO(image_bytes)

    img = Image.open(file_like_object)
    img = img.convert(mode="RGB")
    img = img.resize((224, 224), Image.Resampling.BILINEAR)

    arr = np.array(img)
    x = arr / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    x_norm = (x - mean) / std
    x_norm = x_norm.astype(np.float32)

    x_norm = np.expand_dims(x_norm, axis=0)
    x_norm = x_norm.transpose(0, 3, 1, 2)

    return x_norm
