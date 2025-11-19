import pickle
import streamlit as st

# Load the trained model
model4 = pickle.load(open("insurancedata.pkl", "rb"))

def mydeploy():
    st.title("Insurance Purchase Prediction")
    
    # Input from user
    age = st.number_input("Enter Age:", min_value=0)
    
    # Prediction trigger
    if st.button("Predict Insurance Purchase"):
        output = model4.predict([[age]])
        st.write("Prediction (0 = No, 1 = Yes):", int(output[0]))

mydeploy()

