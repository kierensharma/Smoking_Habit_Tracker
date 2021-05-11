from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


pvals = pd.read_csv('pvals03.csv')
pvals = pvals.T
pvals = pvals.reset_index()
pvals = pvals.rename(columns = {'index':'pvals'})

pvals = list(pvals['pvals'].astype(float))
print(pvals)
data = pd.read_csv('noon2noon/labelled_interpolated_2min/2021-02-03.csv')
time = list(data['Timestamp'])
hr = list(data['Heart rate'])   
    
# label = data.loc[data['sleep time']==int(1)]


# pvals.loc[pvals.index == 1].pvals.item()
index1 = count()
index2 = count()
index3 = count()
index4 = count()
fig, (ax1, ax2) = plt.subplots(2)
ax1.set_xlim([pd.to_datetime(time[0]), pd.to_datetime(time[-1])])
ax2.set_xlim([pd.to_datetime(time[0]), pd.to_datetime(time[-1])])
ax1.set_xlabel('Time')
ax2.set_xlabel('Time')
ax1.set_ylabel('Heart Rate')
ax2.set_ylabel('p-value')
plt.style.use('dark_background')
def animate_hr(i):
    data = pd.read_csv('noon2noon/labelled_interpolated_2min/2021-02-03.csv')
    time = list(data['Timestamp'])
    hr = list(data['Heart rate'])   
    pvals = pd.read_csv('pvals03.csv')
    pvals = pvals.T
    pvals = pvals.reset_index()
    pvals = pvals.rename(columns = {'index':'pvals'})
    pvals = list(pvals['pvals'].astype(float))

   
    # label = data.loc[data['sleep time']==int(1)]
    x = pd.to_datetime(time[:next(index1)])
    x2 = pd.to_datetime(time[:next(index3)])
    y = hr[:next(index2)]
    p = pvals[:next(index4)]
    
    
    ax1.set_ylim([min(hr)-5, max(hr)+5])
    ax1.plot_date(x, y, 'r-', lw=1, xdate=True)
    #pvals
    
    ax2.set_ylim([0,1])
    ax2.plot_date(x2, p, 'y-', lw=1, xdate=True)
    plt.tight_layout()
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    #plt.style.use('dark_background')
# def animate_pval(i)


# data = pd.read_csv('noon2noon/labelled_interpolated_2min/2021-02-01.csv')
# time = list(data['Timestamp'])
ani = FuncAnimation(plt.gcf(), animate_hr, interval=100)
plt.show()

ax.set_xlim([pd.to_datetime(time[0]), pd.to_datetime(time[-1])])


    

# left = pd.to_datetime(time[0])
# right = pd.to_datetime(time[-1])
# plt.gca().set_xbound(left, right)



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
