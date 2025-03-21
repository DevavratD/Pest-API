import tensorflow as tf
import numpy as np
import io
import json
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

app = FastAPI(title="Plant Disease Recognition API")

# Enable CORS middleware for all origins (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model (this will happen once when the API starts)
model = None
def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model('trained_plant_disease_model.keras')
    return model

@app.on_event("startup")
async def startup_event():
    load_model()

# Class names for prediction
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# OpenRouter API details
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HTTP_REFERER = "<YOUR_SITE_URL>"   # Optional: Your website URL
X_TITLE = "<YOUR_SITE_NAME>"       # Optional: Your site name

@app.get("/")
def home():
    return {
        "message": "Plant Disease Recognition API",
        "endpoints": {
            "predict": "/predict",
            "predict/llm": "/predict/llm",
            "health": "/health"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and validate the uploaded image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Preprocess the image
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    
    # Get prediction
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = float(prediction[0][result_index])
    
    return {
        "disease": class_names[result_index],
        "confidence": confidence,
        "index": int(result_index)
    }

@app.post("/predict/llm")
async def predict_with_llm(file: UploadFile = File(...)):
    # Get basic prediction
    predict_response = await predict(file)
    if "error" in predict_response:
        raise HTTPException(status_code=400, detail="Prediction error")
    
    disease_name = predict_response["disease"]  # e.g., 'Tomato___Early_blight'
    confidence = predict_response["confidence"]
    
    # Build prompt for OpenRouter API: all details in English; translate disease name to Hindi
    prompt = f"""
**Predicted Disease:** {disease_name.replace('_', ' ')}
**Confidence:** {confidence:.2f}

Please provide a concise Markdown summary about this disease. The response must explicitly include and label the following sections:

- **Description:** A brief overview of the disease in two lines.
- **Symptoms:** List three common symptoms.
- **Prevention:** List three key prevention methods.
- **Recommendations:** List three actionable management steps.

Ensure that each section name (Description, Symptoms, Prevention, Recommendations) is clearly mentioned and formatted as shown.
"""


    
    try:
        # Call the OpenRouter API
        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": HTTP_REFERER,
                "X-Title": X_TITLE,
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": "mistralai/mistral-small-3.1-24b-instruct:free",  # Adjust model if needed
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")
    
    try:
        llm_response = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error decoding LLM response: {str(e)}")
    
    details = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "No details available.")
    
    return {
        "disease": disease_name,
        "confidence": confidence,
        "details": details
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8069)
