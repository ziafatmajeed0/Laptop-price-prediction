import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Import the model
pipe = pickle.load(open('model_pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title('Laptop Price Predictor')

#Brand
Company = st.selectbox('Brand',df['Company'].unique())

#Type of laptop
Type = st.selectbox('Type',df['TypeName'].unique())

#Ram
ram = st.selectbox('RAM (in GB)',[2,4,6,8,12,16,24,32,64])

#Weight
weight = st.number_input('Weight of the laptop')

#TouchScreen
touchscreen = st.selectbox('TouchScreen',['YES','No'])

#IPS
ips = st.selectbox('IPS',['YES','NO'])

#screen size
screensize = st.number_input('Screen Size')

#Resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2800x1800','2560x1600','2560x1440','2304x1440'])

#Cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

hdd = st.selectbox('HDD (in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD (in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU', df['Gpu brand'].unique())

os = st.selectbox('Operating System',df['os'].unique())



if st.button('Predict Price'):
    
    #query
    if touchscreen=='YES':
        touchscreen = 1
    else:
        touchscreen = 0
        
    if ips == 'YES':
        ips = 1
    else:
        ips = 0
    ppi = None
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screensize

    query = pd.DataFrame(np.array([[Company, Type, ram, weight, touchscreen, ips, ppi, cpu, ssd, hdd, gpu, os]]), 
                     columns=['Company', 'TypeName', 'Ram', 'Weight', 'TouchScreen', 'ips', 'ppi', 
                              'Cpu brand', 'SSD', 'HDD', 'Gpu brand', 'os'])  # Ensure column names match training data

    st.title("The predicted price for these configurations is: " + str(int(np.exp(pipe.predict(query)[0])))) # Extract single prediction value
 

