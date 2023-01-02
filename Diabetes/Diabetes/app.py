import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Index Page"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(Glucose,Bp,Insulin,BMI):
    
   
    prediction=classifier.predict([[Glucose,Bp,Insulin,BMI]])
    print(prediction)
    return prediction



def main():
    st.title("Diabetes Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Diabetes Predictor </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Glucose = st.text_input("Glucose","Type Here")
    Bp = st.text_input("Bp","Type Here")
    Insulin = st.text_input("Insulin","Type Here")
    BMI = st.text_input("BMI","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(Glucose,Bp,Insulin,BMI)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    
