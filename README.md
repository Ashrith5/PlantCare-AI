# Plant Disease Recognition System

## Description
This project is a web-based application for recognizing plant diseases using machine learning. It allows users to upload images of plant leaves and receive predictions about potential diseases, along with insights and recommendations.

## Features
- AI-powered plant disease classification
- Image upload and prediction
- Disease insights and recommendations
- Community reporting functionality
- Weather data integration


## Project Structure
- `plant_disease.py`: Main Streamlit application
- `trained_plant_disease_model.keras`: Pre-trained model for disease classification
- `requirements.txt`: List of required Python packages

## Pages
1. Home: Introduction to the system
2. About: Dataset information
3. Disease Recognition: Upload images and get predictions
4. Community Reporting: Submit reports about observed plant diseases
5. Weather Data: Display current weather information

## Technologies Used
- Streamlit
- TensorFlow
- OpenCV
- Matplotlib
- NumPy

## Dataset
- Over 87,000 RGB images of crop leaves
- 38 different plant disease classes
- Train set: 70,295 images
- Validation set: 17,572 images

## Installation
1. Clone the repository:
git clone <repository-url>
cd plant-disease-recognition

## 2. Install dependencies:
pip install -r requirements.txt

## 3. Usage
Run the Streamlit app:
streamlit run plant_disease.py
