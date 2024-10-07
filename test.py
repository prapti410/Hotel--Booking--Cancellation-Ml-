import streamlit as st
import numpy as np
import pandas as pd
import xgboost
import pickle 

with open ('final_model_xgb.pkl','rb') as file:
    model = pickle.load(file)

def prediction(input_data):
    input_data = np.aaray(input_data, dtype='object')
    
    pred = model.predict_proba(input_data)[:,1][0]
    
    if pred>0.5:
        return f' This booking is more likely to get canceled : Chances = {round(pred*100,2)}%'
    else:
        return f' This booking is less likely to get canceled : Chances = {round(pred*100,2)}%'
    
def main():
    st.title('INN Hotels')    

    lt = st.text_input('Enter lead time. ')
    mkt = (lambda x: 1 if x == 'Online' else 0)(st.selectbox('Enter the type of booking', ['Online', 'Offline']))
    mkt = market[market_res]
    spcl = st.selectbox('How many special requests have been made?',[0,1,2,3,4,5])
    price = st.text_input('Enter the price of the room.')
    adults = st.selectbox('How many adults per room?', [1,2,3,4])
    wknd = st.text_input('How many weekend nights?')
    prk = (lambda x : 1 if x=='Yes' else 0)(st.selectbox('Does booking includes parking facility.',['Yes','No']))
    wk = st.text_input('How many weekday nights?')
    arr_d = st.slider('What will be the day of arrival', min_value=1, max_value=31, step=1)
    arr_m = st.slider('What will be the month of arrival', min_value=1, max_value=12, step=1)
    weekd_lambda = (lambda x : 0 if x=='Mon' else 1 if x=='Tues' else 2 if x=='Wed' else 3 if x=='Thrus' else 4 if x=='Fri' else 5 if x=='Sat' else 6)
    arr_wd = weekd_lambda(st.selectbox('What is the weekday of arrival?'['Mon', 'Tues', 'Wed', 'Thrus', 'Fri', 'Sat', 'Sun']))

    input_list = [[lt, mkt, spcl, price, adults, wknd, prk, wk, arr_d, arr_m, arr_wd]]

    if st.button('Predict'):
        response = prediction(input_list)
        st.success(response)

if __name__ == '__main__':
    main()        