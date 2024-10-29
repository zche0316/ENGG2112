# 添加必要的导入
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 在文件开头添加
DATA_PATH = None  # 将在运行时设置

def split_dataset(merged_data, feature_columns, target_column, test_size=0.2, val_size=0.2, random_state=42):
    """
    将数据集分割为训练集、验证集和测试集
    
    参数:
        merged_data: 合并后的数据集
        feature_columns: 特征列名列表
        target_column: 目标变量列名
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
    """
    # 首先分出测试集
    train_val_data, test_data = train_test_split(
        merged_data, 
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    # 剩余数据中分出验证集
    val_ratio = val_size / (1 - test_size)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True
    )
    
    print(f"\n数据集划分情况:")
    print(f"训练集大小: {len(train_data)} ({len(train_data)/len(merged_data)*100:.2f}%)")
    print(f"验证集大小: {len(val_data)} ({len(val_data)/len(merged_data)*100:.2f}%)")
    print(f"测试集大小: {len(test_data)} ({len(test_data)/len(merged_data)*100:.2f}%)")
    
    # 准备特征(X)和目标变量(y)
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    
    X_val = val_data[feature_columns]
    y_val = val_data[target_column]
    
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    
    # 打印类别分布
    if target_column == 'fire_binary':
        print("\n训练集类别分布:")
        print(y_train.value_counts(normalize=True) * 100)
        print("\n验证集类别分布:")
        print(y_val.value_counts(normalize=True) * 100)
        print("\n测试集类别分布:")
        print(y_test.value_counts(normalize=True) * 100)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def preprocess_features(X_train, X_val, X_test, strategy='mean'):
    """
    处理特征数据中的缺失值和异常值
    
    参数:
        X_train, X_val, X_test: 训练集、验证集和测试集的特征数据
        strategy: 填充策略，可选 'mean'（平均值）, 'median'（中位数）
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    # 1. 处理缺失值
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
    
    # 2. 标准化特征
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
    
    # 打印处理结果
    print("\n数据预处理结果:")
    print(f"训练集形状: {X_train_scaled.shape}")
    print(f"验证集形状: {X_val_scaled.shape}")
    print(f"测试集形状: {X_test_scaled.shape}")
    
    # 检查是否还有缺失值
    print("\n处理后的缺失值检查:")
    print("训练集缺失值:", X_train_scaled.isnull().sum().sum())
    print("验证集缺失值:", X_val_scaled.isnull().sum().sum())
    print("测试集缺失值:", X_test_scaled.isnull().sum().sum())
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def prepare_weather_fire_data(data_type='binary', weather_types=None):
    """
    准备气象和火灾数据
    
    Args:
        data_type: 'binary' 或 'count' - 决定返回二元分类还是计数数据
        weather_types: 需要处理的天气类型列表，如 ['humid', 'temp']。默认处理所有类型
    """
    global DATA_PATH

    if DATA_PATH is None:
        # 如果没有设置DATA_PATH，则使用默认路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        DATA_PATH = os.path.join(project_root, 'data', 'processed')
    
    try:
        # 读取数据
        data_files = {
            'humid': ('processed_humid_data.csv', 'HUMID'),
            'wind': ('processed_wind_data.csv', 'WSP'),
            'temp': ('processed_temp_data.csv', 'TEMP')
        }
        
        # 检查火灾数据文件
        fire_data_path = os.path.join(DATA_PATH, 'processed_fire_data.csv')
        if not os.path.exists(fire_data_path):
            raise FileNotFoundError(f"火灾数据文件不存在: {fire_data_path}")
        
        if weather_types is None:
            weather_types = list(data_files.keys())
            
        # 检查所需的天气数据文件是否存在
        for w_type in weather_types:
            filename, _ = data_files[w_type]
            filepath = os.path.join(DATA_PATH, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"天气数据文件不存在: {filepath}")
            
        # 读取选定的天气数据
        weather_dfs = {}
        for w_type in weather_types:
            filename, _ = data_files[w_type]
            filepath = os.path.join(DATA_PATH, filename)
            weather_dfs[w_type] = pd.read_csv(filepath)
                
        # 读取火灾数据
        fire_data = pd.read_csv(fire_data_path)
        
        # 处理时间列
        for df in weather_dfs.values():
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        fire_data['date'] = pd.to_datetime(fire_data['date'])
        fire_data.set_index('date', inplace=True)
        
        # 对每个特征进行标准化
        for df in weather_dfs.values():
            for col in df.columns:
                if col != 'datetime':
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        # 合并选定的气象数据
        weather_data = pd.concat(list(weather_dfs.values()), axis=1)
        
        # 准备目标变量
        if data_type == 'binary':
            fire_data['fire_binary'] = (fire_data['fire_count'] > 0).astype(int)
            target_col = 'fire_binary'
        else:
            target_col = 'fire_count'
            
        # 合并气象和火灾数据
        merged_data = pd.merge(
            weather_data,
            fire_data[[target_col]],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # 处理缺失值
        merged_data = merged_data.fillna(merged_data.mean())
        
        # 获取各类型特征的列名
        feature_cols = {
            'humid': [col for col in merged_data.columns if 'HUMID' in col],
            'wind': [col for col in merged_data.columns if 'WSP' in col],
            'temp': [col for col in merged_data.columns if 'TEMP' in col]
        }
        
        # 只返回选定天气类型的列名
        selected_cols = [feature_cols[w_type] for w_type in weather_types]
        
        print("\n合并后的数据信息:")
        print(f"数据形状: {merged_data.shape}")
        print(f"选择的天气类型: {weather_types}")
        
        return merged_data, *selected_cols
        
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        raise
