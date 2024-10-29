import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_PATH = None

def split_dataset(merged_data, feature_columns, target_column, test_size=0.2, val_size=0.2, random_state=42):

    train_val_data, test_data = train_test_split(
        merged_data, 
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    val_ratio = val_size / (1 - test_size)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Data set splited successfully:")
    print(f"Training set size: {len(train_data)} ({len(train_data)/len(merged_data)*100:.2f}%)")
    print(f"Validating set size: {len(val_data)} ({len(val_data)/len(merged_data)*100:.2f}%)")
    print(f"Testing set size: {len(test_data)} ({len(test_data)/len(merged_data)*100:.2f}%)")
    
    # Prepare features(X) and target variable(y)
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    
    X_val = val_data[feature_columns]
    y_val = val_data[target_column]
    
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    
    if target_column == 'fire_binary':
        print("\nTraining set classification:")
        print(y_train.value_counts(normalize=True) * 100)
        print("\nValidating set classification:")
        print(y_val.value_counts(normalize=True) * 100)
        print("\nTesting set classification:")
        print(y_test.value_counts(normalize=True) * 100)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def preprocess_features(X_train, X_val, X_test, strategy='mean'):
    """
    Process missing values in the feature data
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    # Process with missing values
    imputer = SimpleImputer(strategy=strategy)
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_imputed),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imputed),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("\nData pre-processing results:")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Validating set shape: {X_val_scaled.shape}")
    print(f"Testing set shape: {X_test_scaled.shape}")
    
    print("\nCheck missing-values in the pre-processed data:")
    print("Missing value of training set:", X_train_scaled.isnull().sum().sum())
    print("Missing value of validating set:", X_val_scaled.isnull().sum().sum())
    print("Missing value of testing set:", X_test_scaled.isnull().sum().sum())
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def prepare_weather_fire_data(data_type='binary', weather_types=None):
    global DATA_PATH

    if DATA_PATH is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        DATA_PATH = os.path.join(project_root, 'data', 'processed')
    
    try:
        data_files = {
            'humid': ('processed_humid_data.csv', 'HUMID'),
            'wind': ('processed_wind_data.csv', 'WSP'),
            'temp': ('processed_temp_data.csv', 'TEMP')
        }
        
        fire_data_path = os.path.join(DATA_PATH, 'processed_fire_data.csv')
        if not os.path.exists(fire_data_path):
            raise FileNotFoundError(f"Fire data file not exist: {fire_data_path}")
        
        if weather_types is None:
            weather_types = list(data_files.keys())
            
        for w_type in weather_types:
            filename, _ = data_files[w_type]
            filepath = os.path.join(DATA_PATH, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Weather data file not exist: {filepath}")
            
        weather_dfs = {}
        for w_type in weather_types:
            filename, _ = data_files[w_type]
            filepath = os.path.join(DATA_PATH, filename)
            weather_dfs[w_type] = pd.read_csv(filepath)
                
        fire_data = pd.read_csv(fire_data_path)
        
        for df in weather_dfs.values():
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        fire_data['date'] = pd.to_datetime(fire_data['date'])
        fire_data.set_index('date', inplace=True)
        
        for df in weather_dfs.values():
            for col in df.columns:
                if col != 'datetime':
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        weather_data = pd.concat(list(weather_dfs.values()), axis=1)
        
        if data_type == 'binary':
            fire_data['fire_binary'] = (fire_data['fire_count'] > 0).astype(int)
            target_col = 'fire_binary'
        else:
            target_col = 'fire_count'
            
        merged_data = pd.merge(
            weather_data,
            fire_data[[target_col]],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        merged_data = merged_data.fillna(merged_data.mean())
        
        feature_cols = {
            'humid': [col for col in merged_data.columns if 'HUMID' in col],
            'wind': [col for col in merged_data.columns if 'WSP' in col],
            'temp': [col for col in merged_data.columns if 'TEMP' in col]
        }
        
        selected_cols = [feature_cols[w_type] for w_type in weather_types]
        
        print("\nData shape after merging:")
        print(f"Data shape: {merged_data.shape}")
        print(f"Data type selected: {weather_types}")
        
        return merged_data, *selected_cols
        
    except Exception as e:
        print(f"Exception while processing data: {str(e)}")
        raise