from __future__ import annotations
from fastapi import FastAPI, File, UploadFile
from fastai.vision import *
import uvicorn

app = FastAPI()

path = os.path.dirname(__file__)
#check if file exist
clf = load_learner(path, 'hedhika-classifier.pkl')


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    img = open_image(file.file)
    preds, idx, output = clf.predict(img)
    prediction = dict({clf.data.classes[i]: round(to_np(p) * 100, 2) for i, p in enumerate(output) if p > 0.2})
    return {"prediction": prediction}


if __name__ == "__main__":
 uvicorn.run(app, host="0.0.0.0", port=8000)