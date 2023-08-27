from model import df,pipe
import numpy as np
import streamlit as st
hide_button = """
<style>
button {
  visibility:hidden;
}

</style>

""" 

st.title("Laptop Price Predictor")

#Manufacturer
brand = st.selectbox('Manufacturer',df['Company'].unique())

#category
type = st.selectbox('Category',df['TypeName'].unique())

#RAM
ram = st.selectbox('RAM',[2,4,6,8,12,16,32,64,128])

# Weight
weight = st.number_input('Weight')

# Touch Screen
touch_screen = st.selectbox('Touch Screen ',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['Yes','No'])

# CPU Brand
cpu = st.selectbox('CPU Brand',df['CPU Brand'].unique())

#screen size
size = st.number_input('Screen Size')

#resolution
resolution = st.selectbox('Resolution',['1920x1080','1366x766','1600x900','3840x2160','3200x1800','2880x1800',
                                        '2560x1600','2560x1440','2560x1400','2304x1440'])

# HDD
hdd = st.selectbox('HDD',[0,128,256,512,1024,2048])

# SDD
sdd = st.selectbox('SDD',[0,128,256,512,1024])

# GPU Brand
gpu = st.selectbox('GPU Brand',df['GPU Brand'].unique())

# Operating System
os = st.selectbox('Operating System',df['OS'].unique())

if st.button('Predict Price'):
    
    if touch_screen == 'Yes':
        touch_screen = 1
    else:
        touch_screen = 0
    
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])

    ppi = ((x_res)**2 + (y_res)**2 )**0.5/size
    
    qurey = np.array([brand,type,ram,weight,touch_screen,ips,ppi,cpu,hdd,sdd,gpu,os])

    qurey = qurey.reshape(1,12)

    st.title("The predicted price of this configuration is: " + str(int(np.exp(pipe.predict(qurey)[0]))))
