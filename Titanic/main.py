import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("Titanic.sav", "rb"))


def titan(input_data):
    input_data_as_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return "The Passenger did not make it."

    else:
        return "The Passenger survived."


def main():
    st.title("Titanic Survival Prediction App")
    Pclass = st.text_input("P class")
    Sex = st.text_input("Sex")
    Age = st.text_input("Age")

    Dialog = ''

    if st.button("Predict the Survival"):
        Dialog = titan([Pclass, Sex, Age])

    st.success(Dialog)


if __name__ == '__main__':
    main()
