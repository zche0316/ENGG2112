import pandas as pd
import csv
from datetime import datetime, timedelta

def process_fire_data(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)
    
    # Create DataFrame
    columns = ['fire_id', 'fire_name', 'ignition_date', 'extinguish_date', 'ignition_cause', 'area_ha', 'unused1', 'unused2']
    df = pd.DataFrame(data[1:], columns=columns) 

    df['ignition_date'] = pd.to_datetime(df['ignition_date'].str.split().str[0], format='%m/%d/%Y', errors='coerce').dt.date
    df['extinguish_date'] = pd.to_datetime(df['extinguish_date'].str.split().str[0], format='%m/%d/%Y', errors='coerce').dt.date

    df['area_ha'] = pd.to_numeric(df['area_ha'], errors='coerce')

    # Data cleaning by removing rows containing NaN
    df = df.dropna(subset=['ignition_date', 'extinguish_date', 'area_ha'])

    date_range = pd.date_range(start=df['ignition_date'].min(), end=df['extinguish_date'].max()).date

    result = pd.DataFrame(index=date_range, columns=['fire_count', 'total_area'])
    result = result.fillna(0.0)

    for _, row in df.iterrows():
        date_range = pd.date_range(start=row['ignition_date'], end=row['extinguish_date']).date
        days_affected = (row['extinguish_date'] - row['ignition_date']).days + 1
        daily_area = float(row['area_ha']) / days_affected
        result.loc[date_range, 'fire_count'] += 1
        result.loc[date_range, 'total_area'] += daily_area

    result = result.reset_index()
    result.columns = ['date', 'fire_count', 'total_area']

    return result

def main():
    input_file_path = 'data/raw/fire_data.csv'
    output_file_path = 'data/processed/processed_fire_data.csv'
    
    processed_data = process_fire_data(input_file_path)
    
    # Save processed data as CSV file
    processed_data.to_csv(output_file_path, index=False)
    print(f"Processed data has been saved to {output_file_path}")

if __name__ == "__main__":
    main()
