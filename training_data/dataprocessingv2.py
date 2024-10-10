import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split
from datetime import datetime 

def process_data(data, lags):
    df = pd.read_csv(data)

    # Get unique scats
    unique_scats = df['SCATS Number'].unique()

    # Test with first scat number: 0970
    scat = unique_scats[0]
    scat_data = df[df['SCATS Number'] == scat]

    # Get the traffic flow columns
    traffic_flow = scat_data.filter(regex=r'^V\d+$').columns

    # Reshape the data from wide to long
    scat_data_long = pd.melt(
        scat_data,
        id_vars=['Date'],
        value_vars=traffic_flow,
        var_name='Time Period',
        value_name='Lane 1 Flow (Veh/15 Minutes)'
    )

    # Take Time Period interval and change to numerical value return as str
    scat_data_long['Time'] = ((scat_data_long['Time Period'].str.replace("V", "").astype(int) + 1)).astype(str)

     # Create a mapping of V00, V01, ..., V95 to the actual time intervals
    time_mapping = {f'V{i:02d}': f'{i//4:02d}:{i%4*15:02d}' for i in range(96)}

    # Create new 15 Minutes column and combine data and time
    scat_data_long['15 Minutes'] = scat_data_long['Date'] + " " + scat_data_long['Time']

    # restructure df to only have 5 Minutes and Lane 1 Flow (Veh/5 Minutes)
    scat_data_long = scat_data_long[['15 Minutes', 'Lane 1 Flow (Veh/15 Minutes)']]

    scat_data_long.to_csv('scat_data_970.csv', index=False)
    

if __name__ == '__main__':
    data = 'scats_data.csv'
    lags = 5
    process_data(data, lags)
