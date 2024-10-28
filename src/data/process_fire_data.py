import pandas as pd
import csv
from datetime import datetime, timedelta

def process_fire_data(file_path):
    # Read csv file
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)
    
    # Create DataFrame
    columns = ['fire_id', 'fire_name', 'ignition_date', 'extinguish_date', 'ignition_cause', 'area_ha', 'unused1', 'unused2']
    df = pd.DataFrame(data[1:], columns=columns)  # Skip the first row (header)

    # Convert date format and keep only the date part
    df['ignition_date'] = pd.to_datetime(df['ignition_date'].str.split().str[0], format='%m/%d/%Y', errors='coerce').dt.date
    df['extinguish_date'] = pd.to_datetime(df['extinguish_date'].str.split().str[0], format='%m/%d/%Y', errors='coerce').dt.date

    # Convert area_ha to float
    df['area_ha'] = pd.to_numeric(df['area_ha'], errors='coerce')

    # Data cleaning by removing rows containing NaN
    df = df.dropna(subset=['ignition_date', 'extinguish_date', 'area_ha'])

    # Create date range
    date_range = pd.date_range(start=df['ignition_date'].min(), end=df['extinguish_date'].max()).date

    # Initialize result DataFrame
    result = pd.DataFrame(index=date_range, columns=['fire_count', 'total_area'])
    result = result.fillna(0.0)

    # Calculate daily fire count and affected area
    for _, row in df.iterrows():
        date_range = pd.date_range(start=row['ignition_date'], end=row['extinguish_date']).date
        days_affected = (row['extinguish_date'] - row['ignition_date']).days + 1
        daily_area = float(row['area_ha']) / days_affected
        result.loc[date_range, 'fire_count'] += 1
        result.loc[date_range, 'total_area'] += daily_area

    # Reset index, make date a column
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

    print("\nProcessed data sample:")
    print(processed_data.head(10))

    print("\nData type information:")
    print(processed_data.dtypes)
    
    print("\nStatistical information:")
    print(processed_data.describe())

    print("\nDate range:")
    print("Start date:", processed_data['date'].min())
    print("End date:", processed_data['date'].max())

    print("\nMaximum values:")
    print("Maximum fire count:", processed_data['fire_count'].max())
    print("Maximum affected area:", processed_data['total_area'].max())

    # Find the date with the most fires
    max_fire_date = processed_data.loc[processed_data['fire_count'].idxmax()]
    print("\nDate with the most fires:")
    print(f"Date: {max_fire_date['date']}")
    print(f"Fire count: {max_fire_date['fire_count']}")
    print(f"Affected area: {max_fire_date['total_area']:.2f}")

    # Output data for the date with the most fires and its surrounding 5 days
    max_fire_index = processed_data['fire_count'].idxmax()
    print("\nData for the date with the most fires and its surrounding 5 days:")
    print(processed_data.iloc[max(0, max_fire_index-5):min(len(processed_data), max_fire_index+6)])

if __name__ == "__main__":
    main()
