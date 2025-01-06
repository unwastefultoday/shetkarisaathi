import streamlit as st
import numpy as np
import requests
import pandas as pd
import pickle
import os
import random
import google.generativeai as genai
genai.configure(api_key="AIzaSyC7EMGMD40KoLyLg_DbgHpPUdJ0VeJ2ZMg")
from tensorflow.keras.models import load_model
from PIL import Image
os.chdir(r"C:\Users\HP\Desktop\shetisahayak")

def get_weather(city, api_key):
    """
    Fetch weather data from an external API.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"  # For temperature in Celsius
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_weather_forecast(city, api_key):
    """
    Fetches a 5-day weather forecast for the given city using OpenWeatherMap API.

    Parameters:
        city (str): The city for which the weather forecast is required.
        api_key (str): Your OpenWeatherMap API key.

    Returns:
        pd.DataFrame: A DataFrame containing the weather forecast.
    """
    # OpenWeatherMap API endpoint
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"

    # Fetch data from API
    response = requests.get(url)

    # Check if the response is successful
    if response.status_code == 200:
        data = response.json()
        
        # Extracting required data
        weather_data = [
            {
                "date": entry["dt_txt"],
                "temperature": entry["main"]["temp"],
                "perceived_temperature": entry["main"]["feels_like"],
                "humidity": entry["main"]["humidity"],
                "pressure": entry["main"]["pressure"],
                "weather_description": entry["weather"][0]["description"],
            }
            for entry in data["list"]
        ]

        # Convert to DataFrame
        forecast_df = pd.DataFrame(weather_data)
        return forecast_df
    else:
        print(f"Error: Unable to fetch data for city '{city}', status code {response.status_code}")
        return None

def preprocess_image(image_path):
  """
  Preprocesses the image for model prediction.

  Args:
    image_path: Path to the uploaded image file.

  Returns:
    Preprocessed image as a NumPy array.
  """
  img = Image.open(image_path)
  img = img.resize((64, 64))  # Resize to match training data
  img_array = np.array(img) / 255.0  # Normalize pixel values
  img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
  return img_array

def predict_disease(model, image_array, class_labels):
  """
  Predicts the disease using the given model.

  Args:
    model: Trained model.
    image_array: Preprocessed image array.
    class_labels: List of class labels.

  Returns:
    Predicted disease class label.
  """
  predictions = model.predict(image_array)
  predicted_class_index = np.argmax(predictions)
  predicted_class = class_labels[predicted_class_index] 
  return predicted_class
  

def main():
    st.title("Sheti Sahayak")

    # Tabs
    tab1, tab2,tab3,tab4, tab5, tab6 = st.tabs(["Weather Forecast", "Which Crops to Plant?","Which Fertilizers to Use?", "Crop Disease Identifier", "Govt. Schemes", "Chat with Sheti Sahayak" ])

    with tab1:
        st.header("Weather Forecast")
    
        # API key for OpenWeatherMap
        api_key = "2deccccbe8ae3f8affd1dda7de033aee"  
        # User input for city
        district = ["Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana", "Chandrapur",
            "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur",
            "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad",
            "Palghar", "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara", "Sindhudurg",
            "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"]
        city = st.selectbox("Select a district:", district)
        if st.button("Get Weather"):
            if city:
                weather_data = get_weather(city, api_key)
                if weather_data:
                    st.write(f"**Temperature:** {weather_data['main']['temp']} °C")
                    st.write(f"**Humidity:** {weather_data['main']['humidity']}%")
                    st.write(f"**Weather:** {weather_data['weather'][0]['description'].capitalize()}")
                    st.write(f"**Wind Speed:** {weather_data['wind']['speed']} m/s")
                else:
                    st.error("Could not fetch weather data. Please check the city name or try again later.")
            
                forecast = get_weather_forecast(city, api_key)
                if 'forecast' in locals() and isinstance(forecast, pd.DataFrame):
                    st.subheader(f"Next 5 days Hourly Forecast")
                    st.write(forecast)   
    with tab2:
        st.header("Crop Recommendation")

        districts = ["Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana", "Chandrapur",
            "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur",
            "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad",
            "Palghar", "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara", "Sindhudurg",
            "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"]

        district = st.selectbox("Select a district:", districts, key = 10)
        westmhdistricts = ["Kolhapur", "Solapur", "Satara", "Sangli", "Pune"]

        if district in westmhdistricts:
            with open("dumped models/westmh_crops.pkl", "rb") as file:
                model = pickle.load(file)

            soil_colors = ['Black', 'Dark Brown', 'Light Brown', 'Medium Brown', 'Red', 'Reddish Brown']
            
            # Input Fields
            soil_color = st.selectbox("Soil Color", options=soil_colors)
            nitrogen = st.number_input("Nitrogen (mg/kg)", min_value=0, max_value=500, step=5)
            phosphorus = st.number_input("Phosphorus (mg/kg)", min_value=0, max_value=500, step=5)
            potassium = st.number_input("Potassium (mg/kg)", min_value=0, max_value=500, step=5)
            soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=1.0)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=1.0)
            
            if st.button("Recommend Crop"):
                input_data = pd.DataFrame({
                "Nitrogen": [nitrogen],
                "Phosphorus": [phosphorus],
                "Potassium": [potassium],
                "pH": [soil_ph],
                "Rainfall": [rainfall],
                "Temperature": [temperature]
             })
            
                # One-hot encode the district and soil color
                for dist in westmhdistricts:
                    input_data[f"District_Name_{dist}"] = [1 if district == dist else 0]
    
                for color in soil_colors:
                    input_data[f"Soil_color_{color}"] = [1 if soil_color == color else 0]
                correct_order = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature',
       'District_Name_Kolhapur', 'District_Name_Pune', 'District_Name_Sangli',
       'District_Name_Satara', 'District_Name_Solapur', 'Soil_color_Black',
       'Soil_color_Dark Brown', 'Soil_color_Light Brown',
       'Soil_color_Medium Brown', 'Soil_color_Red', 'Soil_color_Reddish Brown']
                input_data = input_data.reindex(columns=correct_order, fill_value=0) 
                # Predict crop
                prediction = model.predict(input_data)[0]

                # Display the prediction
                st.success(f"The recommended crop is: **{prediction}**")

        else:
            with open("dumped models/crops.pkl", "rb") as file:
                model = pickle.load(file)
                nitrogen = st.number_input("Nitrogen (mg/kg)", min_value=0, max_value=500, step=5)
                phosphorus = st.number_input("Phosphorus (mg/kg)", min_value=0, max_value=500, step=5)
                potassium = st.number_input("Potassium (mg/kg)", min_value=0, max_value=500, step=5)
                soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=1.0)
                humidity = st.number_input("Humidity %", min_value=0.0, max_value=100.0, step=1.0)
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
                temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=1.0)
                
                if st.button("Recommend Crop"):
                    input_data = pd.DataFrame({
                    "N": [nitrogen],
                    "P": [phosphorus],
                    "K": [potassium],
                    "temperature": [temperature],
                    "humidity":[humidity],
                    "ph": [soil_ph],
                    "rainfall": [rainfall]
                    })

                    # Predict crop
                    prediction = model.predict(input_data)[0]

                    st.success(f"The recommended crop is: **{prediction}**")
    
    with tab3:
        st.header("Fertilizer Recommendation")
        districts = ["Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana", "Chandrapur",
            "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur",
            "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad",
            "Palghar", "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara", "Sindhudurg",
            "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"]

        district = st.selectbox("Select a district:", districts, key = 9)
        westmhdistricts = ["Kolhapur", "Solapur", "Satara", "Sangli", "Pune"]
        if district not in westmhdistricts:
            with open("dumped models/fertilizer_dt.pkl", "rb") as file:
                model = pickle.load(file)

            soil = ['Black', 'Clayey', 'Loamy', 'Sandy', 'Red']
            crop = ['Barley','Cotton','Ground Nuts','Maize','Millets','Oil seeds','Paddy','Pulses','Sugarcane','Tobacco','Wheat','coffee','kidneybeans','orange','pomegranate','rice','watermelon']

            #Feature Values
            Temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=1.0, key = 4)
            Humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0, key = 5)
            Moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, step=1.0, key = 6)
            Nitrogen = st.number_input("Nitrogen (mg/kg)", min_value=0, max_value=500, step=5, key = 1)
            Phosphorus = st.number_input("Phosphorus (mg/kg)", min_value=0, max_value=500, step=5, key = 2)
            Potassium = st.number_input("Potassium (mg/kg)", min_value=0, max_value=500, step=5, key = 3)
            Soil_Type = st.selectbox("Soil Type", options=soil)
            Crop_Type = st.selectbox("Crop Type", options=crop, key = 7)
            if st.button("Recommend Fertilizer",key = 8):
                input_data = pd.DataFrame({
                "Temparature": [Temperature],
                "Humidity": [Humidity],
                "Moisture": [Moisture]
                })
                # One-hot encode the district and soil color
                for s in soil:
                    input_data[f"Soil_Type_{s}"] = [1 if Soil_Type == s else 0]
    
                for ct in crop:
                    input_data[f"Crop_Type_{ct}"] = [1 if Crop_Type == ct else 0]

                input_data["Nitrogen"]= [Nitrogen]

                input_data["Potassium"]=[Potassium]

                input_data["Phosphorous"]=[Phosphorus]
                correct_order = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium',
       'Phosphorous', 'Soil_Type_Black', 'Soil_Type_Clayey', 'Soil_Type_Loamy',
       'Soil_Type_Red', 'Soil_Type_Sandy', 'Crop_Type_Barley',
       'Crop_Type_Cotton', 'Crop_Type_Ground Nuts', 'Crop_Type_Maize',
       'Crop_Type_Millets', 'Crop_Type_Oil seeds', 'Crop_Type_Paddy',
       'Crop_Type_Pulses', 'Crop_Type_Sugarcane', 'Crop_Type_Tobacco',
       'Crop_Type_Wheat', 'Crop_Type_coffee', 'Crop_Type_kidneybeans',
       'Crop_Type_orange', 'Crop_Type_pomegranate', 'Crop_Type_rice',
       'Crop_Type_watermelon']  
                input_data = input_data.reindex(columns=correct_order, fill_value=0)             
                # Predict crop
                prediction = model.predict(input_data)[0]
        
                st.success(f"The recommended fertilizer is: **{prediction}**")
        else:
            with open("dumped models/westmh_fertilizer.pkl", "rb") as file:
                model = pickle.load(file)
            westmhdistricts = ["Kolhapur", "Solapur", "Satara", "Sangli", "Pune"]
            soil_colors = ['Black', 'Dark Brown', 'Light Brown', 'Medium Brown', 'Red', 'Reddish Brown']
            crops = ['Cotton','Ginger', 'Gram', 'Grapes', 'Groundnut', 'Jowar', 'Maize', 'Masoor', 'Moong', 'Rice', 'Soybean', 'Sugarcane', 'Tur', 'Turmeric', 'Urad', 'Wheat']
            # Input Fields
            soil_color = st.selectbox("Soil Color", options=soil_colors)
            nitrogen = st.number_input("Nitrogen (mg/kg)", min_value=0, max_value=500, step=5, key = 12)
            phosphorus = st.number_input("Phosphorus (mg/kg)", min_value=0, max_value=500, step=5,key = 13)
            potassium = st.number_input("Potassium (mg/kg)", min_value=0, max_value=500, step=5,key = 14)
            soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=1.0,key = 15)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0,key = 16)
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=1.0,key = 17)
            Crop_Type = st.selectbox("Crop Type", options=crops, key = 11)
            
            if st.button("Recommend Crop", key = 18):
                input_data = pd.DataFrame({
                "Nitrogen": [nitrogen],
                "Phosphorus": [phosphorus],
                "Potassium": [potassium],
                "pH": [soil_ph],
                "Rainfall": [rainfall],
                "Temperature": [temperature]
             })
            
       
                for dist in westmhdistricts:
                    input_data[f"District_Name_{dist}"] = [1 if district == dist else 0]
    
                for color in soil_colors:
                    input_data[f"Soil_color_{color}"] = [1 if soil_color == color else 0]

                for crop in crops:
                    input_data[f"Crop_{crop}"] = [1 if Crop_Type == crop else 0]
                correct_order = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature',
       'District_Name_Kolhapur', 'District_Name_Pune', 'District_Name_Sangli',
       'District_Name_Satara', 'District_Name_Solapur', 'Soil_color_Black',
       'Soil_color_Dark Brown', 'Soil_color_Light Brown',
       'Soil_color_Medium Brown', 'Soil_color_Red', 'Soil_color_Reddish Brown',
       'Crop_Cotton', 'Crop_Ginger', 'Crop_Gram', 'Crop_Grapes',
       'Crop_Groundnut', 'Crop_Jowar', 'Crop_Maize', 'Crop_Masoor',
       'Crop_Moong', 'Crop_Rice', 'Crop_Soybean', 'Crop_Sugarcane', 'Crop_Tur',
       'Crop_Turmeric', 'Crop_Urad', 'Crop_Wheat']
                input_data = input_data.reindex(columns=correct_order, fill_value=0) 

                # Predict crop
                prediction = model.predict(input_data)[0]

                # Display the prediction
                st.success(f"The recommended fertilizer is: **{prediction}**")

    with tab4:
        def give_solution(crop, disease, language = "English"):
            prompt = f"""Your name is Sheti Sahayak, which is Marathi for farming friend. For this particular use case, I am asking you to take on the role of a farming expert who shall assist a marathi farmer who's {crop} crops are struggling with {disease}. Frame your answer to this query as if you are speaking to that farmer regarding this issue. Frame your answer in {language} only and consider only the information given in this prompt, assume that no other information is there. Give general guidelines for the solution. DOn't mention whether your advice is general or specific, jist proceed as if ordinary."""  
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        
        st.header("Are my crops healthy?")
        potato_model = load_model("dumped models/Potato_leaf_prediction_model.h5")
        rice_model = load_model("dumped models/Rice_leaf_prediction_model.h5")
        wheat_model = load_model("dumped models/Wheat_leaf_prediction_model.keras")
        corn_model = load_model("dumped models/Corn_leaf_prediction_model.h5")
        crop_options = { "Corn": ['Common Rust of Corn', 'Gray Leaf Spot of Corn', 'Healthy', 'Northern Leaf Blight of Corn'], "Potato": ['Early Blight of Potato', 'Healthy', 'Late Blight of Potato'], "Rice": ['Rice Brown Spot', 'Healthy', 'Rice Leaf Blast', 'Rice Neck Blast'], "Wheat": ['Wheat Brown Rust', 'Healthy', 'Wheat Yellow Rust']}
        crop_options_mr = { "मका": ['Common Rust of Corn', 'Gray Leaf Spot of Corn', 'Healthy', 'Northern Leaf Blight of Corn'], "बटाटा": ['Early Blight of Potato', 'Healthy', 'Late Blight of Potato'], "तांदूळ": ['Rice Brown Spot', 'Healthy', 'Rice Leaf Blast', 'Rice Neck Blast'], "गहू": ['Wheat Brown Rust', 'Healthy', 'Wheat Yellow Rust']}  
        mapping = {"मका":"Corn", "बटाटा": "Potato", "तांदूळ": "Rice", "गहू":"Wheat"}
        language__options = ["English", "मराठी"]
        selected_language_name = st.selectbox("Select Language for Response", language__options)
        if selected_language_name == "English":
            selected_crop_name = st.selectbox("Select Crop", crop_options.keys())
            class_labels = crop_options[selected_crop_name]
        else:
            selected_crop_name = st.selectbox("Select Crop", crop_options_mr.keys())
            english_crop_name = mapping[selected_crop_name]
            class_labels = crop_options_mr[selected_crop_name]
            selected_crop_name = mapping[selected_crop_name]
           
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
        # Save the uploaded image
            image_path = "uploaded_image.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        if st.button("Predict"):
            try:
            # Preprocess the image
                preprocessed_image = preprocess_image(image_path)

                # Select and use the appropriate model
                if selected_crop_name == "Potato":
                    model = potato_model
                elif selected_crop_name == "Rice":
                    model = rice_model
                elif selected_crop_name == "Wheat":
                    model = wheat_model
                else:
                    model = corn_model
                prediction = predict_disease(model, preprocessed_image, class_labels)
                st.write(f"Predicted: {prediction}")
                if prediction == 'Healthy' and selected_language_name == "English":
                    st.success("Your Crops are Healthy, good work!")
                elif prediction == 'Healthy' and selected_language_name == "मराठी":
                    st.success("आपल्या पिकांची चांगली वाढ झाली आहे, चांगले काम केले!") 
                else:
                    try:
                        st.success(give_solution(selected_crop_name, prediction, selected_language_name))
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
	
    with tab5:
        def give_schemes(ptype, level, language):
            if ptype == "government":
                supp = f"The schemes should be brought by {level} government for Maharashtrian farmers. "
            else:
                supp = ""

            prompt = f"""
            Please provide me with the schemes brought about by {ptype} players for farmers that you know about in {language} only.{supp} 
            Please structure your output in a list format, with name of the offering entity(private player/central government/state government), 
            name of scheme, a brief description of that scheme. Just share the data that you have. No need to clarify whether you have realtime information or not and no need to provide any disclaimer. Your output should be in a table format only. No need to provide any note or disclaimer or P.S. or anything apart from what was asked.
            """
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        
        st.header("Schemes for Farmers in Maharashtra")
        language__options = ["English", "मराठी"]
        selected_language_name = st.selectbox("Select Language for Response", language__options, key =563)
        level = st.selectbox("Select Government Level", ["Maharashtra State", "Central"])
        
        if st.button("Get Schemes"):
            if level == "government" and level is None:
                st.warning("Please select a government level.")
            else:
                try:
                    schemes_list = give_schemes("government", level, selected_language_name)
                    st.success(schemes_list) 
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    with tab6:
        st.header("Hi! I am Sheti Sahayak, How Can I help you?")
        def chat(query):
            prompt = f"""
           Your name is Sheti Sahayak, which is Marathi for "farming friend. For this particular use case, I am asking you to take on the role of a farming expert who shall assist a            marathi farmer with their questions. They may pose the questions in english or marathi (transliterated in english). If the question is being posed in english, please answer in english. Else, answer in the marathi script without any translation. You are to guide them in their farming related questions only, anything and everything else is out of your scope, which you will clearly mention in your answer. Following is the query. {query}
            """
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        query = st.text_input("Enter your prompt:")
        if st.button("Talk to me!"):
            st.success(chat(query))
    
if __name__ == "__main__":
    main()
