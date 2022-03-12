import pickle
import numpy as np
import streamlit as st

# loading the saved file
loaded_model = pickle.load(open("trained_model.sav", "rb"))


def bankChurn(input_data):
    input_data_as_array = np.asarray(input_data)
    input_data_reshaped=input_data_as_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return "Customer will not churn."

    else:
        return "Customer will churn."


def main():
    st.title("Diabetes Prediction Web App")
    Gender = st.text_input("Gender")
    Age = st.text_input("Age")
    Balance = st.text_input("Balance")
    CreditScore = st.text_input("Credit")
    CustomerId = st.text_input("Customer_Id")

    Dialog = ''

    if st.button(" Prediction Result"):
        Dialog = bankChurn([Gender, Age, Balance, CreditScore, CustomerId])

    st.success(Dialog)


if __name__ == '__main__':
    main()
