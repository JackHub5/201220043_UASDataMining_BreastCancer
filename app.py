import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model
log_regression = pickle.load(open('log_regression.pkl','rb'))

#load dataset
data = pd.read_csv('Breast Cancer.csv')
data = data[['diagnosis','radius_mean','area_mean', 'radius_se', 'area_se', 'smoothness_mean','smoothness_se']]
data['diagnosis'].replace({'M':1, 'B':0}, inplace = True)

st.title('Aplikasi Kanker Payudara')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Kanker Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
st.sidebar.header('Metode/Model')
st.sidebar.subheader('Logistic Regression')
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini merupakan dataset Breast Cancer Wisconsin (Diagnostic)</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
X = data.drop('diagnosis',axis=1)
y = data['diagnosis']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    radius_mean = st.sidebar.slider('radius_mean: ', 6.981000, 28.11000, 15.0)
    area_mean = st.sidebar.slider('area_mean: ', 143.50, 2501.0, 1000.0)
    radius_se = st.sidebar.slider('radius_se: ', 0.111500, 2.87300, 1.5)
    area_se = st.sidebar.slider('area_se: ', 6.802000, 542.20, 200.0)
    smoothness_mean = st.sidebar.slider('smoothness_mean: ', 0.052630, 0.16340, 0.1)
    smoothness_se = st.sidebar.slider('smoothness_se: ', 0.001713, 0.03113, 0.002000)
    
    user_report_data = {
        'radius_mean':radius_mean,
        'area_mean':area_mean,
        'radius_se':radius_se,
        'area_se':area_se,
        'smoothness_mean':smoothness_mean,
        'smoothness_se':smoothness_se,
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = log_regression.predict(user_data)
log_regres_score = accuracy_score(y_test,log_regression.predict(X_test))

#output
st.subheader('Hasilnya yaitu : ')
output=''
if user_result[0]==0:
    output='Kamu terkena kanker payudara jinak'
else:
    output ='Kamu terkena kanker payudara ganas'
st.title(output)
st.subheader('Model yang digunakan : Logistic Regression')
st.subheader('Accuracy : ')
st.write(str(log_regres_score*100)+'%')