import numpy as np

loka = np.array([[1, 99, 13.2], [999, 0, 18.3], [0.112, 1, 0]], dtype = float)
string = ""
for i in range (3):
    for j in range (3):
        string += (str(loka[i][j]))
        string += ("\n")

file = open("parametros.txt", "w")
file.writelines(string)