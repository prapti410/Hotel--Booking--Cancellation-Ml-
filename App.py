import streamlit as st 
import numpy as np 
import pandas as pd  
import pickle 
from xgboost import XGBClassifier

with open('Final_model_xgb.pkl','rb')as file:
    model = pickle.load(file)

#input_data = [[lt,mst,spcl,price,adul,wkend,park,wk,ar_d,ar_m,ar_w]]
def prediction(input_data):
    input_data=np.array(input_data,dtype='object')

    pred=model.predict_proba(input_data)[:,1][0]
    
    if pred>0.5:
        return f'This Booking is more likely to get cancelled: Chances={round(pred*100,2)}%'
    else:
        return f'This Booking is less likely to get cancelled: Chances={round(pred*100,2)}%'

def main():
    st.title('INN Hotels')
    st.image('hotelimage.jpg', use_column_width=True)
    lt=st.text_input('Enter Lead time')
    mkt=(lambda x : 1 if x =='Online' else 0)(st.selectbox('Enter the type of booking',['Online','Offline']))
    spcl=st.selectbox('How many special requests have been made?',[0,1,2,3,4,5])
    price=st.text_input('Enter the price of the room.')
    adults= st.selectbox('How many Adults per room?',[1,2,3,4])
    wknd=st.text_input('How many weekend nights?')
    prk=(lambda x : 1 if x=='Yes' else 0)(st.selectbox('Does booking includes parking facility.',['Yes','No']))
    wk=st.text_input('How many weekday nights')
    arr_d=st.slider('What will be the day of arrival.',min_value=1,max_value=31,step=1)
    arr_mon=st.slider('What will be the month of arrival.',min_value=1,max_value=12,step=1)
    week_lambda= (lambda x : 0 if x=='Mon' else 1 if x=='Tue' else 2 if x=='Wed' else 3 
                  if x=='Thus' else 4 if x=='Fri' else 5 if x=='Sat' else 6 )
    arr_wd= week_lambda(st.selectbox('What is the Weekday of arrival?.',['Mon','Tue','Wed','Thus','Fri','Sat','Sun']))

    input_data = [[lt,mkt,spcl,price,adults,wknd,prk,wk,arr_d,arr_mon,arr_wd]]


    if st.button('Predict'):
        responce=prediction(input_data)
        st.success(responce)


if __name__ =='__main__':
    main()
