import numpy as np
import pickle

loaded_model = pickle.load(open("Titanic.sav", "rb"))

input_data = [[3, 0, 22]]
input_data_as_array = np.asarray(input_data)
input_data_reshaped = input_data_as_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("The passenger did not make it.")

else:
    print("The passenger survived")
