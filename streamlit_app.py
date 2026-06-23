import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️", layout="centered")

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

IMG_SIZE = (224, 224)
MODEL_PATH = "model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32")
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title("CIFAR-10 Image Classifier")
    st.write("Upload an image and get a prediction from your trained VGG16 model.")

    model = load_model()

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            with st.spinner("Making prediction..."):
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)
                pred_index = np.argmax(predictions[0])
                pred_class = CLASS_NAMES[pred_index]
                confidence = float(np.max(predictions[0]))

            st.success(f"Prediction: {pred_class}")
            st.write(f"Confidence: {confidence:.2f}")

            prob_dict = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
            st.bar_chart(prob_dict)

if __name__ == "__main__":
    main()
