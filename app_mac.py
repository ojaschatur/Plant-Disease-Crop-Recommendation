import streamlit as st
import os
import sys
import pickle
import numpy as np
import pandas as pd
from PIL import Image

# Only this env var; others can cause issues!
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import TensorFlow with error handling
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    # Do not use GPU/CPU device code here; let Apple metal handle it
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    st.error(f"TensorFlow import failed: {e}")

# Set page config
st.set_page_config(
    page_title="Plant Disease & Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def load_models():
    plant_model = None
    crop_model = None
    
    if TENSORFLOW_AVAILABLE:
        try:
            plant_model = tf.keras.models.load_model('plant_disease_model.h5', compile=False)
            plant_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            st.success("✅ Plant disease model loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load plant disease model: {str(e)}")
            plant_model = None
    else:
        st.warning("⚠️ TensorFlow not available - Plant disease classification disabled")
    
    try:
        with open('crop_recommendation_model.pkl', 'rb') as file:
            crop_model = pickle.load(file)
        st.success("✅ Crop recommendation model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading crop recommendation model: {str(e)}")
        crop_model = None
    
    return plant_model, crop_model

PLANT_DISEASE_CLASSES = [
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

def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(model, image):
    if not TENSORFLOW_AVAILABLE or model is None:
        return "TensorFlow not available", 0.0
    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        if predicted_class_idx < len(PLANT_DISEASE_CLASSES):
            predicted_class = PLANT_DISEASE_CLASSES[predicted_class_idx]
        else:
            predicted_class = f"Class_{predicted_class_idx}"
        return predicted_class, confidence
    except Exception as e:
        return f"Prediction error: {str(e)}", 0.0

def predict_crop(model, features):
    prediction = model.predict([features])
    return prediction[0]

def main():
    st.title("🌱 Plant Disease Classification & Crop Recommendation System")
    st.markdown("---")
    plant_model, crop_model = load_models()
    if plant_model is None and crop_model is None:
        st.error("❌ Failed to load any models. Please ensure the model files are in the correct directory.")
        return
    if plant_model is None:
        st.warning("⚠️ Plant disease classification is disabled due to TensorFlow issues.")
    if crop_model is None:
        st.warning("⚠️ Crop recommendation is disabled due to model loading issues.")

    tab1, tab2 = st.tabs(["🔍 Plant Disease Classification", "🌾 Crop Recommendation"])
    with tab1:
        st.header("Plant Disease Classification")
        if plant_model is None:
            st.error("🚫 **Plant Disease Classification is currently unavailable**")
            st.info("TensorFlow may be incompatible with your system. Use Google Colab or a compatible machine, or use crop recommendation below.")
        else:
            st.write("Upload an image of a plant leaf to detect diseases")
            uploaded_file = st.file_uploader(
                "Choose a plant image...",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear image of a plant leaf"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                col1, col2 = st.columns(2)
                with col1:
                    # st.image(image, caption="Uploaded Image", use_column_width=True)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                with col2:
                    if st.button("🔍 Analyze Disease", type="primary"):
                        with st.spinner("Analyzing image..."):
                            try:
                                predicted_class, confidence = predict_disease(plant_model, image)
                                if confidence > 0:
                                    st.success("Analysis Complete!")
                                    clean_class = predicted_class.replace('___', ' - ').replace('_', ' ')
                                    st.markdown(f"### 📊 Results:")
                                    st.markdown(f"**Predicted Class:** {clean_class}")
                                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                                    st.progress(confidence / 100)
                                    if 'healthy' in predicted_class.lower():
                                        st.success("✅ Plant appears to be healthy!")
                                    else:
                                        st.warning("⚠️ Disease detected. Consider consulting an agricultural expert.")
                                else:
                                    st.error(f"❌ {predicted_class}")
                            except Exception as e:
                                st.error(f"Error during prediction: {str(e)}")
    with tab2:
        st.header("Crop Recommendation System")
        if crop_model is None:
            st.error("🚫 **Crop Recommendation is currently unavailable**")
            st.info("Could not load the crop recommendation model file.")
            st.markdown("**Please ensure:**")
            st.markdown("- `crop_recommendation_model.pkl` is in the same directory as this app")
            st.markdown("- The file is not corrupted")
        else:
            st.write("Enter soil and climate parameters to get crop recommendations")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Soil Parameters")
                nitrogen = st.slider("Nitrogen (N)", 0, 140, 50)
                phosphorus = st.slider("Phosphorus (P)", 5, 145, 50)
                potassium = st.slider("Potassium (K)", 5, 205, 50)
                ph = st.slider("pH Level", 3.5, 10.0, 6.5, step=0.1)
            with col2:
                st.subheader("🌤️ Climate Parameters")
                temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0, step=0.5)
                humidity = st.slider("Humidity (%)", 14.0, 100.0, 65.0, step=0.5)
                rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, step=5.0)
            features = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🌾 Get Crop Recommendation", type="primary", use_container_width=True):
                    with st.spinner("Analyzing soil and climate conditions..."):
                        try:
                            recommended_crop = predict_crop(crop_model, features)
                            st.success("Analysis Complete!")
                            st.markdown("### 🎯 Recommended Crop:")
                            st.markdown(f"## **{recommended_crop.title()}**")
                            st.markdown("### 📋 Input Summary:")
                            input_df = pd.DataFrame({
                                'Parameter': ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
                                'Value': [f"{nitrogen}", f"{phosphorus}", f"{potassium}", f"{temperature}°C", f"{humidity}%", f"{ph}", f"{rainfall}mm"],
                                'Unit': ['', '', '', '°C', '%', '', 'mm']
                            })
                            st.table(input_df[['Parameter', 'Value']])
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🌱 Built with Streamlit | Plant Disease Classification & Crop Recommendation System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
