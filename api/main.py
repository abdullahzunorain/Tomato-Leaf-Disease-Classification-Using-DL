# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf

# app = FastAPI()

# # Enable CORS for frontend interaction
# origins = [
#     "http://localhost",
#     "http://localhost:3000"
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# # Load your trained model
# MODEL_PATH = r"C:\Users\lenovo\OneDrive\Desktop\potato disease detection\Z_tomato project__02\best_model.h5"
# MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

# # Compile the model manually (if needed)
# MODEL.compile(
#     optimizer=tf.keras.optimizers.Adam(),
#     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#     metrics=['accuracy']
# )

# # Define class names
# CLASS_NAMES = {
#     0: "Tomato_Bacterial_spot",
#     1: "Tomato_Early_blight",
#     2: "Tomato_Late_blight",
#     3: "Tomato_Leaf_Mold",
#     4: "Tomato_Septoria_leaf_spot",
#     5: "Tomato_Spider_mites_Two_spotted_spider_mite",
#     6: "Tomato__Target_Spot",
#     7: "Tomato__Tomato_YellowLeaf__Curl_Virus",
#     8: "Tomato__Tomato_mosaic_virus",
#     9: "Tomato_healthy"
# }

# @app.get("/ping")
# async def ping():
#     return {"message": "Hello, I am alive"}

# # Utility to convert uploaded image to model input format
# def read_file_as_image(data: bytes) -> np.ndarray:
#     try:
#         image = Image.open(BytesIO(data)).convert("RGB")
#         image = image.resize((256, 256))  # Resize as expected by model
#         image_array = np.array(image) / 255.0  # Normalize
#         return image_array
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # File type check
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid file type. Please upload an image file (JPG/PNG)."
#         )

#     try:
#         image = read_file_as_image(await file.read())
#         img_batch = np.expand_dims(image, axis=0)

#         predictions = MODEL.predict(img_batch)
#         predicted_index = int(np.argmax(predictions[0]))
#         predicted_class = CLASS_NAMES[predicted_index]
#         confidence = float(np.max(predictions[0]))

#         return {
#             "class": predicted_class,
#             "confidence": round(confidence, 4)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)







from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Enable CORS for frontend interaction
origins = [
    "http://localhost",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load your trained model
MODEL_PATH = r"C:\Users\lenovo\OneDrive\Desktop\potato disease detection\Z_tomato project__02\best_model.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Compile the model manually to avoid legacy issues
MODEL.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Define class names (index corresponds to model prediction output)
CLASS_NAMES = {
    0: "Tomato_Bacterial_spot",
    1: "Tomato_Early_blight",
    2: "Tomato_Late_blight",
    3: "Tomato_Leaf_Mold",
    4: "Tomato_Septoria_leaf_spot",
    5: "Tomato_Spider_mites_Two_spotted_spider_mite",
    6: "Tomato__Target_Spot",
    7: "Tomato__Tomato_YellowLeaf__Curl_Virus",
    8: "Tomato__Tomato_mosaic_virus",
    9: "Tomato_healthy"
}

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

# Utility to convert uploaded image to model input format
def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))  # Match input shape expected by model
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

    predictions = MODEL.predict(img_batch)
    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": round(confidence, 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
