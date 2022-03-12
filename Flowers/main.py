import pickle

import numpy as np
import streamlit as st

loaded_model = pickle.load(open("Flowers.sav", "rb"))


def iris(input_data):
    input_data_as_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_array.reshape(1, -1)
    predictions = loaded_model.predict(input_data_reshaped)
    print(predictions)

    if predictions[0] == 0:
        return "The Species is Setosa."

    if predictions[0] == 1:
        return "the Species is Versicolor."

    if predictions[0] == 2:
        return "The species is Virginica."


def main():
    st.title("Flowers Species Prediction Web Application")
    sepal_length = st.text_input("Sepal Length")
    sepal_Width = st.text_input("Sepal Width")
    petal_length = st.text_input("Petal Length")
    petal_width = st.text_input("Petal Width")

    Dialog = ''

    if st.button("Predict Species"):
        Dialog = iris([sepal_length, sepal_Width, petal_length, petal_width])

    st.success(Dialog)

if __name__=='__main__':
    main()