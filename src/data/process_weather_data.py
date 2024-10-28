import pandas as pd
import numpy as np

def process_weather_data(input_file_path, output_file_path):
    # Read the csv file
    df = pd.read_csv(input_file_path, sep=';', encoding='utf-8', skiprows=2, header=0)

    # Check the name for datetime column
    date_col = next((col for col in df.columns if 'date' in col.lower()), None)
    time_col = next((col for col in df.columns if 'time' in col.lower()), None)

    if date_col and time_col:
        df[date_col] = df[date_col].str.strip()
        # Delete empty rows
        df = df.dropna(subset=[date_col, time_col])
        # Convert datetime column
        df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
        df[time_col] = df[time_col].replace('24:00', '00:00')
        df.loc[df[time_col] == '00:00', date_col] += pd.Timedelta(days=1)
        df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col], errors='coerce')
    elif date_col:
        df[date_col] = df[date_col].str.strip()
        df = df.dropna(subset=[date_col])
        df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

    # Delete lines with missing datetime
    df = df.dropna(subset=['datetime'])

    # Set datetime as index
    df.set_index('datetime', inplace=True)

    # Delete original time and date column
    if date_col:
        df = df.drop(date_col, axis=1)
    if time_col:
        df = df.drop(time_col, axis=1)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate daily averages
    daily_avg = df.resample('D').mean()

    # Save processed data as CSV file
    daily_avg.to_csv(output_file_path)

    print(f"Processed data saved to {output_file_path}")

    return daily_avg

def main():
    input_file_path = 'data/raw/humid_data.csv'
    output_file_path = 'data/processed/processed_humid_data.csv'
    processed_humid_data = process_weather_data(input_file_path, output_file_path)

    input_file_path = 'data/raw/temp_data.csv'
    output_file_path = 'data/processed/processed_temp_data.csv'
    processed_temp_data = process_weather_data(input_file_path, output_file_path)

    input_file_path = 'data/raw/wind_data.csv'
    output_file_path = 'data/processed/processed_wind_data.csv'
    processed_wind_data = process_weather_data(input_file_path, output_file_path)

    print("All data has been processed.")


if __name__ == "__main__":
    main()
