import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json


# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

# Preprocess Test Image
def preprocess_image(image_path):
    # Load and process image using TensorFlow utilities
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    return input_arr

# Display Test Image
def display_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Test Image")
    plt.xticks([])  # No x-axis ticks
    plt.yticks([])  # No y-axis ticks
    plt.show()

# Prediction Logic
def model_prediction(model, input_arr):
    predictions = model.predict(input_arr)
    return np.argmax(predictions), predictions

# Disease Insights
def get_disease_insights(disease_name):
    disease_info = {
        'Apple___Apple_scab': {
            'Symptoms': 'Dark, scabby lesions on leaves, fruit, and twigs.',
            'Treatment': 'Prune infected areas and apply fungicides like Captan or Mancozeb.',
            'Prevention': 'Plant resistant varieties and maintain proper spacing for airflow.'
        },
        'Apple___Black_rot': {
            'Symptoms': 'Dark, rotten spots on fruit and bark lesions on twigs.',
            'Treatment': 'Remove infected fruit and apply copper-based fungicides.',
            'Prevention': 'Prune regularly and ensure good drainage.'
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'Symptoms': 'Yellowing of leaves, curling of leaves upwards, stunted growth, and reduced yield.',
        'Treatment': 'There is no direct treatment for the virus. Remove infected plants and control whiteflies.',
        'Prevention': 'Use resistant tomato varieties, control whiteflies with insecticides, and practice good field sanitation.'
    }
        # Add more disease details here
    }
    return disease_info.get(disease_name, None)

# Actionable Recommendations
def get_actionable_recommendations(disease_name):
    recommendations = {
        'Apple___Apple_scab': 'Apply fungicide every 7-10 days to protect healthy plants.',
        'Apple___Black_rot': 'Remove all affected leaves and ensure proper irrigation to reduce disease spread.'
    }
    return recommendations.get(disease_name, "No recommendations available.")

# Community Reporting
def handle_community_report():
    st.subheader("Community Reporting")
    reported_disease = st.text_input("Enter the disease name:")
    additional_info = st.text_area("Additional Information (Optional):")
    
    if st.button("Submit Report"):
        if reported_disease:
            st.success("Thank you for your report! We will review it shortly.")
            # In real scenario, save the data to a database or file
            report_data = {
                "disease": reported_disease,
                "info": additional_info
            }
            with open("community_reports.json", "a") as file:
                json.dump(report_data, file)
                file.write("\n")
        else:
            st.error("Please enter a disease name to report.")

# Data Integration (Weather, Risk, etc.)
def get_weather_data():
    # Simulate fetching weather data (You can integrate a weather API like OpenWeather)
    weather_data = {
        'temperature': 28,  # Example data
        'humidity': 85,     # Example data
        'rainfall': 10,     # Example data
    }
    return weather_data

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Community Reporting", "Weather Data"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_container_width=True)
    st.markdown("""
        Welcome to the Plant Disease Recognition System! 🌿🔍
        This system helps identify plant diseases efficiently using advanced AI models.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
        ### Dataset Information
        This dataset contains over 87K RGB images of crop leaves categorized into 38 classes.
        - Train: 70,295 images
        - Validation: 17,572 images
        - Test: 33 images
        
        [Dataset Link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    uploaded_file = st.file_uploader("Choose a test image:")

    if uploaded_file is not None:
        # Save uploaded image to temporary path
        test_image_path = "uploaded_image.jpg"
        with open(test_image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.image(test_image_path, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            st.snow()  # Show loading animation
            model = load_model()

            # Preprocess image and make predictions
            input_arr = preprocess_image(test_image_path)
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            predicted_index, predictions = model_prediction(model, input_arr)
            predicted_class = class_names[predicted_index]
            st.success(f"Prediction: {predicted_class}")
            
            # Display disease insights and recommendations
            disease_details = get_disease_insights(predicted_class)
            if disease_details:
                st.subheader("Disease Insights")
                st.markdown(f"**Symptoms:** {disease_details['Symptoms']}")
                st.markdown(f"**Treatment:** {disease_details['Treatment']}")
                st.markdown(f"**Prevention:** {disease_details['Prevention']}")
            else:
                st.warning("No detailed insights available.")

            recommendations = get_actionable_recommendations(predicted_class)
            st.subheader("Actionable Recommendations")
            st.markdown(recommendations)

            st.bar_chart(predictions[0])  # Display prediction confidence

# Community Reporting Page
elif app_mode == "Community Reporting":
    handle_community_report()

# Weather Data Page
elif app_mode == "Weather Data":
    weather = get_weather_data()
    st.subheader("Weather Data")
    st.write(f"Temperature: {weather['temperature']}°C")
    st.write(f"Humidity: {weather['humidity']}%")
    st.write(f"Rainfall: {weather['rainfall']} mm")
