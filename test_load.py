import os
import json
import joblib
import numpy as np
import keras
from PIL import Image

ART = 'app_artifacts'

image_model = keras.models.load_model(f'{ART}/image_model.keras')
tfidf = joblib.load(f'{ART}/tfidf.joblib')
text_model = joblib.load(f'{ART}/text_model.joblib')
with open(f'{ART}/class_names.json') as f:
    class_names = json.load(f)

# Image model on one sample photo
sample = os.listdir(f'{ART}/images')[0]
img = Image.open(f'{ART}/images/{sample}').convert('RGB').resize((96, 96), Image.BILINEAR)
x = np.array(img, dtype='float32')[None, ...]
probs = image_model.predict(x, verbose=0)[0]
print("Image:", sample, "->", class_names[probs.argmax()], f"({probs.max():.2f})")

# Text model on one title
print("Text: 'Puma Men Grey T-shirt' ->", text_model.predict(tfidf.transform(['Puma Men Grey T-shirt']))[0])