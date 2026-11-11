import os
from fastapi import FastAPI, UploadFile, HTTPException
from PIL import UnidentifiedImageError as ImgErr
from fastapi.middleware.cors import CORSMiddleware
from model import predict

app = FastAPI()

env = os.getenv("ALLOWED_ORIGINS", "https://example.com")
origins = env.split(",")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "OK"}


@app.post("/predict")
async def get_file_and_predict(file: UploadFile):
    # 5000000 byte = 5 mb
    if file.size is not None and file.size > 5000000:
        raise HTTPException(status_code=413, detail="Your image must be less than 5mb")

    image_bytes = await file.read()

    try:
        pred, prob = predict(image_bytes)
    except ImgErr:
        raise HTTPException(
            status_code=415, detail="You can send only jpg and png images"
        )

    response = {
        "pred": pred,
        "prob": prob,
    }

    return response
