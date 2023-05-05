import numpy as np
from matplotlib import pyplot as plt

def show_loss(data_file,interval=100):
    training_loss = np.genfromtxt(data_file,delimiter=",")
    average_loss = []
    for i in range(interval,len(training_loss),interval):
        #print(sum(training_loss[i-interval:i,1])/interval)
        average_loss.append([i, sum(training_loss[i-interval:i,1])/interval] )
    average_loss = np.array(average_loss)
    plt.plot(average_loss[:, 0], average_loss[:, 1])
    plt.show()
