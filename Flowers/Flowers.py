import numpy as np
import pickle

loaded_model = pickle.load(open("Flowers.sav", "rb"))

input_data = [[10.0, 9.0, 7.0, 6.6]]

input_data_as_array = np.asarray(input_data)

input_reshaped = input_data_as_array.reshape(1, -1)

predictions = loaded_model.predict(input_reshaped)

print(predictions)

if predictions[0] == 0:
    print("The Species is Setosa.")

if predictions[0] == 1:
    print("the Species is Versicolor.")

if predictions[0] == 2:
    print("The species is Virginica.")