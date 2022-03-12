import numpy as np
import pickle

loaded_model = pickle.load(open("trained_model.sav", "rb"))

input_data = [[1, 41, 83807.86, 608, 15647311]]

input_data_as_array = np.asarray(input_data)
input_data_reshaped = input_data_as_array.reshape(1, -1)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("Customer will not churn.")

else:
    print("Customer will churn.")
