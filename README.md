# Deploying-a-personalized-image-classifier

# CIFAR-10 Image Classifier with Streamlit

This project deploys a VGG16-based image classification model using Streamlit. Users can upload an image, preview it, and get a real-time prediction with a confidence score.

## Features
- File upload for image input.
- Image preview before prediction.
- Preprocessing for VGG16 input.
- Model prediction and class label output.
- Confidence score display.

## Files
- `streamlit_app.py` — Streamlit app code.
- `model.h5` — Trained model file.
- `requirements.txt` — Required Python packages.

## Run the App
```bash
streamlit run streamlit_app.py
```

## Note
The model must be trained and saved as `model.h5` before running the app.
