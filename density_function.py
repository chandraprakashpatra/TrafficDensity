#density function 
import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


continous_density_list=[5, 5, 5, 5, 5, 5, 5, 5, 6, 7, 6, 8, 8, 9, 6, 5, 5, 6, 5, 6, 6, 6, 6, 6, 6, 5, 5, 6, 5, 6, 6, 6, 6, 6, 5, 6, 6, 6, 5, 6, 6, 5, 4, 6, 5, 6, 6, 6, 6, 8, 6, 6, 7, 6, 7, 5, 5, 5, 6, 7, 6, 7, 8, 8, 6, 6, 6, 5, 6, 6, 5, 7, 6, 7, 7, 7, 7, 8, 10, 11, 8, 9, 7, 7, 7, 7, 7, 6, 6, 7, 6, 6, 6, 7, 8, 8, 8, 7, 5, 5, 5, 5, 6, 5, 7, 6, 6, 4, 5, 5, 4, 4, 5, 6, 5, 4, 5, 4, 3, 5, 4, 4, 5, 5, 5, 5, 6, 7, 8, 8, 6, 5, 4, 5, 6, 5, 5, 4, 4, 4, 5, 5, 4, 4, 5, 4, 5, 4, 7, 9, 7, 7, 6, 6, 7, 7, 8, 8, 8, 8, 7, 9, 8, 8, 9, 8, 9, 9, 10, 9, 8, 9, 8, 9, 10, 12, 12, 11, 8, 7, 7, 9, 9, 8, 8, 8, 8, 7, 6, 7, 7, 5, 6, 6, 8, 7, 7, 6, 8, 6, 6, 6, 6, 4, 4, 6, 4, 5, 6, 8, 8, 7, 7, 7, 8, 7, 7, 9, 8, 8, 7, 7, 7, 7, 6, 7, 7, 8, 8, 8, 7, 6, 7, 6, 8, 6, 7, 6, 6, 6, 5, 5, 5, 6, 6, 9, 6, 6, 5, 4, 6, 5, 5, 4, 7, 8, 12, 11, 9, 13, 8, 5, 9, 6, 5, 6, 6, 6, 7, 8, 8, 6, 8, 9, 8, 8, 8, 8, 9, 8, 6, 9, 11, 9, 10, 9, 6, 6, 6, 10, 12, 10, 9, 10, 10, 10, 6, 8, 12, 12, 12, 10, 13]


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')



def predict_clear_time(density_data, threshold=1):
    x = np.arange(len(density_data)).reshape(-1, 1)
    y = np.array(density_data).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    future_time = np.arange(len(density_data), len(density_data) + 10).reshape(-1, 1)
    predictions = model.predict(future_time)
    clear_time_index = np.where(predictions <= threshold)[0]
    if len(clear_time_index) > 0:
        return future_time[clear_time_index[0]][0]
    else:
        return None

# Compute the moving average
ma_density = moving_average(list(continous_density_list))

# Predict when the traffic will clear
clear_time = predict_clear_time(ma_density)
print(f"Predicted time until traffic clears: {clear_time} frames")

# Plot the results
plt.plot(ma_density, label='Moving Average Traffic Density')
plt.axhline(y=1, color='r', linestyle='--', label='Clear Traffic Threshold')
if clear_time:
    plt.axvline(x=clear_time, color='g', linestyle='--', label='Predicted Clear Time')
plt.legend()
plt.show()

#density_calculator(continous_density_list)