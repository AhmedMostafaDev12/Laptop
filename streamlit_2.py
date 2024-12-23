import pickle 
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import streamlit as st

loaded_model = pickle.load(open('laptop_price.sav','rb'))

preprocessor = loaded_model['preproccesor']
model = loaded_model['model']

def prediction(new):
    transformerd = preprocessor.transform(new)
    predict = model.predict(transformerd)

    return int(np.exp(predict))

example_df = pd.DataFrame([['Apple', 'Ultrabook', 'Intel', 'Core i5', 2.3, 8, 128.0, 'SSD', 0.0, 
                            'Intel', 'Iris Plus Graphics 640', 'macOS', 1.37, '2560x1600', 0.0]],
                          columns=['Company', 'TypeName', 'Cpu_Company', 'Cpu_Type', 'Cpu_Freq(GHz)', 
                                   'RAM(GB)', 'First_memory_volume(GB)', 'First_memory_type', 'Second_memory_volume(GB)', 
                                   'Gpu_company', 'Gpu_Type', 'OpSys', 'Weight(kg)', 'Resolution', 'Touchscreen'])



def main():
    st.title("Laptop Feature Prediction")
    st.write("Provide the details below to predict the target variable.")

    
    Company = st.text_input("Company (e.g., Apple, Dell)", "Apple")
    TypeName = st.text_input("Type Name (e.g., Ultrabook, Notebook)", "Ultrabook")
    Cpu_Company = st.text_input("CPU Company (e.g., Intel, AMD)", "Intel")
    Cpu_Type = st.text_input("CPU Type (e.g., Core i5, Core i7)", "Core i5")
    Cpu_Freq = st.number_input("CPU Frequency (GHz)", min_value=0.0, value=2.3)
    RAM = st.number_input("RAM (GB)", min_value=0, value=8)
    First_memory_volume = st.number_input("First Memory Volume (GB)", min_value=0.0, value=128.0)
    First_memory_type = st.text_input("First Memory Type (e.g., SSD, HDD)", "SSD")
    Second_memory_volume = st.number_input("Second Memory Volume (GB)", min_value=0.0, value=0.0)
    Gpu_company = st.text_input("GPU Company (e.g., Intel, NVIDIA)", "Intel")
    Gpu_Type = st.text_input("GPU Type (e.g., Iris Plus Graphics)", "Iris Plus Graphics 640")
    OpSys = st.text_input("Operating System (e.g., macOS, Windows)", "macOS")
    Weight = st.number_input("Weight (kg)", min_value=0.0, value=1.37)
    Resolution = st.text_input("Resolution (e.g., 2560x1600)", "2560x1600")
    Touchscreen = st.number_input("Touchscreen (0 for No, 1 for Yes)", min_value=0, max_value=1, value=0)

    
    if st.button("Predict"):
        
        input_data = pd.DataFrame({
            'Company': [Company],
            'TypeName': [TypeName],
            'Cpu_Company': [Cpu_Company],
            'Cpu_Type': [Cpu_Type],
            'Cpu_Freq(GHz)': [Cpu_Freq],
            'RAM(GB)': [RAM],
            'First_memory_volume(GB)': [First_memory_volume],
            'First_memory_type': [First_memory_type],
            'Second_memory_volume(GB)': [Second_memory_volume],
            'Gpu_company': [Gpu_company],
            'Gpu_Type': [Gpu_Type],
            'OpSys': [OpSys],
            'Weight(kg)': [Weight],
            'Resolution': [Resolution],
            'Touchscreen': [Touchscreen],
        })

        # Get prediction
        prediction_result = prediction(input_data)

        # Display the prediction result
        st.success(f"Prediction Result: {prediction_result}")

# Run the app
if __name__ == "__main__":
    main()