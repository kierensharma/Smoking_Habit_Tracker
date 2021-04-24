from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


# def moving_plot(path):


index1 = count()
index2 = count()

def animate(i):
    data = pd.read_csv('noon2noon/labelled_interpolated/2021-02-01.csv')
    time = list(data['Timestamp'])
    hr = list(data['Heart rate'])   
    # label = data.loc[data['sleep time']==int(1)]
    x = pd.to_datetime(time[:next(index1)])
    y = hr[:next(index2)]
    plt.plot_date(x, y, 'r-', lw=1, xdate=True)
    plt.axes(xlim= (pd.to_datetime(time[0]), pd.to_datetime(time[-1])))
    plt.tight_layout()
    ax = plt.gca()
    ax.set_facecolor('black')

data = pd.read_csv('noon2noon/labelled_interpolated/2021-02-01.csv')
time = list(data['Timestamp'])
ani = FuncAnimation(plt.gcf(), animate, interval=100)
fig,ax = plt.subplots()
fig.autofmt_xdate()


#ax.set_xlim([pd.to_datetime(time[0]), pd.to_datetime(time[-1])])
plt.show()


left = pd.to_datetime(time[0])
right = pd.to_datetime(time[-1])
plt.gca().set_xbound(left, right)



#plt.style.use('fivethirtyeight')

# def moving_plot(path):
# data = pd.read_csv('noon2noon/labelled_interpolated/2021-02-01.csv')
# time = list(data['Timestamp'])
# hr = list(data['Heart rate'])   
# label = data.loc[data['sleep time']==int(1)]

# fig = plt.figure()
# ax = plt.axes(ylim = (0,200))
# line, = ax.plot([], [], lw=2)

# def init():
#     line.set_data([], [])
#     return line, 

# counter = 0
# def animate(i):
#     x = time[counter]
#     y = hr[counter]
#     print(x,y)
#     counter += 1
#     line.set_data(x,y)
#     return line, 

# animation.FuncAnimation(fig, animate, init_func=init, interval=1000)
# plt.show()
