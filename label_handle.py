import numpy as np 
import pandas as pd
import glob
import os


def main(label_path):
    for file in glob.glob(label_path):
        label_data = pd.read_csv(file)
        label_data['Sleep time'] = label_data['Day'].str.cat(label_data['Sleep time'],sep=" ")
        label_data.drop('Day', inplace = True, axis = 1)
        hr_files = sorted(glob.glob('noon2noon/interpolated/*'))
        for day in hr_files:
            print('DATE: ' + day[23:33])
            hr_data = pd.read_csv(day)
            hr_data['sleep time'] = int(0)
            for x in label_data['Sleep time']:
                print(pd.to_datetime(x).date)
                # == hr_data['Timestamp']

            # if hr_data['Timestamp'].Date == label_data['Sleep time'].Date:
            #     print(true)
            # time = label_data.loc[label_data['Sleep time'] == day[23:33], 'Sleep time'].to_frame()
            # for a in time.values:
            #     for b in a:
            #         print(b)
                    #hr_data.loc[hr_data['Timestamp'] > b, 'sleep time'] = int(1)
            #print(hr_data)
            #hr_data.to_csv('noon2noon_data/noon2noon_' + day[25:35] + '.csv')
main('/Users/emiliolanzalaco/Smoking_Habit_Tracker/sleep_labels/feb_sleep_labels.csv')

"""dt = pd.to_datetime("2016-11-13 22:01:25.450")
print (dt)
2016-11-13 22:01:25.450000

print (df.index.get_loc(dt, method='nearest'))
2

idx = df.index[df.index.get_loc(dt, method='nearest')]
print (idx)
2016-11-13 22:00:28.417561344"""

#df.loc[df[theme].isnull(), theme] = int(0)
#, ['sleep time']] = int(1)
