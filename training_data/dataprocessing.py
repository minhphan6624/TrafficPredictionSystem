import pandas as pd
import os

df = None

def fix_data(long):
    global df

    if long == 0:
        print("Skip 0")
        return

    scat_data = df[df['NB_LONGITUDE'] == long]

    # Get all unique dates
    unique_dates = scat_data['Date'].unique()

    # Get the SCATS number
    scats_number = scat_data['SCATS Number'].unique()[0]

    # Get location of SCATS
    location = scat_data['Location'].unique()[0]

    # Replace    
    location = location.replace('HIGH STREET_RD', 'HIGH_STREET_RD')
    location = location.replace('STUDLEY PARK_RD', 'STUDLEY_PARK_RD')
    location = location.replace('MONT ALBERT_RD', 'MONT_ALBERT_RD')

    location_split = location.split(' ')

    # New dataframe to store processed data
    processed_data = pd.DataFrame()

    # loop through all unique dates
    for date in unique_dates:
        # Get all traffic data for the date
        date_data = scat_data[scat_data['Date'] == date]

        # Loop through all time intervals and add to processed data
        for i in range(0, 96):
            # pad the interval number with 0's
            i_padded = str(i).zfill(2)
            column_fix = f"V{i_padded}"

            #print(column_fix)

            interval_data = date_data[column_fix]
            
            # print(int(interval_data))

            # Turn I into a time format like 00:00 multiple by 15 minutes
            # 00:00 = 0

            i_time = i * 15
            i_time_formatted = f"{i_time // 60}:{i_time % 60}"

            # Pad the time with 0's
            if len(i_time_formatted.split(':')[0]) == 1:
                i_time_formatted = f"0{i_time_formatted}"
                
            if len(i_time_formatted.split(':')[1]) == 1:
                i_time_formatted = f"{i_time_formatted}0"

            mins = f"{date} {i_time_formatted}"

            if len(interval_data) == 0:
                #processed_data = processed_data.append({'Date': date, 'Time Interval': i, 'Total': 0}, ignore_index=True)
                print("Found no data.")
            else:
                processed_data = processed_data._append({'15 Minutes': mins, 'Lane 1 Flow (Veh/15 Minutes)': int(interval_data.iloc[0])}, ignore_index=True)

    # Save processed data to csv
    processed_data.to_csv(f'traffic_flows/{scats_number}_{location_split[1]}_trafficflow.csv', index=False)

def process_data(data, lags):
    global df

    df = pd.read_csv(data)

    # Get unique SCATS numbers
    unique_df = df['NB_LONGITUDE'].unique()
    unique_scats = df['SCATS Number'].unique()

    print(f"Unique Df Numbers: {unique_df}, Count: {len(unique_df)}")

    done = 0

    for number in unique_df:
        print(f"Finished {done} out of {len(unique_df) - 1}")
        fix_data(number)

        done += 1

    return unique_scats

def merge_datasets():
    scat_site = '970'

    unique_scats = df['SCATS Number'].unique()

    file_directions = ['E', 'N', 'S', 'W']  
    base_path = 'traffic_flows'  

    #Initialize the dataframes
    dataframes = []

    for direction in file_directions:
        file_name = f"{scat_site}_{direction}_trafficflow.csv"
        file_path = f"{base_path}/{file_name}"

        #Read the file and add the direction column
        data = pd.read_csv(file_path)
        data["directions"] = direction

        #Append the dataframe to the list
        dataframes.append(data)

    #Concatenate the dataframes
    final_data = pd.concat(dataframes)

    final_data.to_csv(f"new_traffic_flows/{scat_site}_trafficflow.csv", index=False)  

    print(f"Merged files for SCAT site {scat_site}")
    
def merge_all_datasets():
    base_path = 'traffic_flows'  

    # List all files in the directory
    all_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]

    # Dictionary to hold DataFrames for each SCAT site
    scat_site_dfs = {}

    # Loop through each file
    for file_name in all_files:
        # Construct the full file path
        file_path = os.path.join(base_path, file_name)
        
        # Get the SCAT site and direction from the file name
        scat_site = file_name.split('_')[0]
        direction = file_name.split('_')[1]
        
        # Read the file and add the direction column
        data = pd.read_csv(file_path)
        data['direction'] = direction
        
        # If this SCAT site is not already in the dictionary, initialize a list
        if scat_site not in scat_site_dfs:
            scat_site_dfs[scat_site] = []
        
        # Append the DataFrame to the SCAT site's list
        scat_site_dfs[scat_site].append(data)

    for scat_site, df_list in scat_site_dfs.items():
        
        # Concatenate all the df for the current SCAT site
        merged_df = pd.concat(df_list)
        
        # Save the merged DataFrame to a new CSV file
        output_file = f"new_traffic_flows/{scat_site}_trafficflow.csv"
        merged_df.to_csv(output_file, index=False)
        
        print(f"Merged file saved as {output_file}")


if __name__ == '__main__':
    data = 'scats_data.csv'
    lags = 5
    # process_data(data, lags)
    merge_all_datasets()
