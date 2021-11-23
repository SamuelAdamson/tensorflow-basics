# Training data : Celsius, Farenheit
# Author: Samuel Adamson
# Last Edit: 11/22/2021

import numpy as np

# Training Arrays
celsius = np.array([-40, -30, -20, -10, 0, 5, 8, 10, 15, 22, 28], dtype=float)
farenheit = np.array([-40,  -22, -4, 14, 32, 41, 46.4, 50, 59, 71.6, 82.4], dtype=float)

# Display Training Values
def display_values():
    for i,c in enumerate(celsius):
        # { Temp } Celsius = { Temp F } Farenheit
        print("{} Celsius = {} Farenheit".format(c, farenheit[i]))