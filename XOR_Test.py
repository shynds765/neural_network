from Neural_Network import Neural_Network
import numpy as np

nn = Neural_Network([2,4,2],.25)
x = np.array([[[1],[0]],[[0],[1]],[[0],[0]],[[1],[1]]])
y = np.array([[[1],[0]],[[1],[0]],[[0],[1]],[[0],[1]]])

error = nn.train(x,y,20000)

testData = np.array([[[0],[5]],[[100],[16]],[[1],[0]],[[1],[0]],[[1],[2]],[[3],[4]],[[10],[10]]])
for i in testData:
    output = nn.predict(i)
    if output[0][0] > output[1][0]:
        print("True")
    else:
        print("False")