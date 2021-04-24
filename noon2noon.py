import numpy as np 
import pandas as pd
import glob 

def main(path):
    all_data = pd.DataFrame()
    
    #looping through each file in path and appending to dataframe
    for file in glob.glob(path + '*'):
        data = pd.read_csv(file)
        all_data = pd.concat([all_data, data])
    all_data = all_data.sort_values('Timestamp')
    all_data = all_data.drop(columns = ['Unnamed: 0'])
    
    #getting rid of first 0h to 12h 
    all_data.index = pd.to_datetime(all_data['Timestamp'])
    all_data = all_data.drop(columns = ['Timestamp'])
    first_time = str(all_data.index[0])
    end_time = first_time[:11] + '11:59:59'
    first_12h = all_data.query("@first_time <= index <= @end_time")
    all_data = all_data.iloc[len(first_12h):] #this is cool
    all_data['sleep time'] = int(0)
    
    all_data = all_data.reindex(all_data.resample('300s').asfreq().index, method='nearest',
                        tolerance=pd.Timedelta('300s')).interpolate('time')
    #appending sleep label
    label_data = pd.read_csv('sleep_labels/feb_sleep_labels.csv')
    label_data['Sleep time'] = label_data['Day'].str.cat(label_data['Sleep time'],sep=" ")
    label_data.drop('Day', inplace = True, axis = 1)
    for x in label_data['Sleep time']:
        if pd.isnull(x) is False:
            time = pd.to_datetime(x)
            index = all_data.index.get_loc(time, method='nearest')
            all_data['sleep time'].iloc[index] = int(1)
            
    
    #looping for all data
    while len(all_data) > 0:
        noon = str(all_data.index[0])
        next_day = str(int(noon[8:10]) + 1).zfill(2)
        noon_next_day = noon[:8] + next_day + ' 11:59:59'
        noon2noon = all_data.query("@first_time <= index <= @noon_next_day")
        all_data = all_data.iloc[len(noon2noon):]
        noon2noon.to_csv('/Users/emiliolanzalaco/Smoking_Habit_Tracker/noon2noon/labelled_interpolated_5min/' + noon[:10] + '.csv')

main('/Users/emiliolanzalaco/Documents/Smoking_Habit_Tracker/HR_CSV_Data/')

