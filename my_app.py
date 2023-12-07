import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle


st.sidebar.title('Churn Prediction')

#dataframe
df = pd.read_csv('HR_Dataset.csv', nrows=(100))
st.markdown("You may check the dataframe below.")
st.write(df.head())

#take user input
satisfaction_level = st.sidebar.number_input("satisfaction_level:",min_value=df.satisfaction_level.min(), max_value=df.satisfaction_level.max())
last_evaluation = st.sidebar.number_input("last_evaluation:",min_value=df.last_evaluation.min(), max_value=df.last_evaluation.max())
number_project = st.sidebar.number_input("number_project:",min_value=df.number_project.min(), max_value=df.number_project.max()) 
average_montly_hours = st.sidebar.number_input("average_montly_hours:",min_value=df.average_montly_hours.min(), max_value=df.average_montly_hours.max()) 
time_spend_company =  st.sidebar.number_input("time_spend_company:",min_value=df.time_spend_company.min(), max_value=df.time_spend_company.max()) 
Work_accident=  st.sidebar.number_input("Work_accident:",min_value=df.Work_accident.min(), max_value=df.Work_accident.max()) 
promotion_last_5years = st.sidebar.number_input("promotion_last_5years:",min_value=df.promotion_last_5years.min(), max_value=df.promotion_last_5years.max()) 
#Departments = st.sidebar.number_input("Departments:",min_value=df.Departments .min(), max_value=df.Departments .max()) 
#Salary = st.sidebar.number_input("Salary:",min_value=df.Salary.min(), max_value=df.Salary.max()) 


# To load machine learning model
final_model=pickle.load(open("xgb.pkl", "rb"))
final_model_encoder=pickle.load(open("encoder", "rb"))


# Create a dataframe using feature inputs
my_dict = {
    "satisfaction_level": satisfaction_level,
    "last_evaluation": last_evaluation,
    "number_project": number_project,
    "average_montly_hours": average_montly_hours,
    "time_spend_company": time_spend_company,
    "Work_accident": Work_accident,
    "promotion_last_5years": promotion_last_5years,
    #"Departments": Departments,
    #"Salary": Salary
}

df = pd.DataFrame.from_dict([my_dict])

st.header(" The features of employee is below:")
st.table(df)

df2 = final_model_encoder.transform(df)

st.subheader("click predict")

if st.button("Predict"):
    prediction = final_model.predict(df2)
