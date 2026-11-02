from typing import Annotated
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.get("/")
def root():
    return {"status": "OK"}

@app.post("/predict")
async def get_file(file: Annotated[bytes, File()]):
    response = {
        "file-size": len(file),
    }

    return response 
