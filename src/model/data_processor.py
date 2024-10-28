# 添加必要的导入
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 在文件开头添加
DATA_PATH = None  # 将在运行时设置


def prepare_humid_fire_data():
    """
    准备湿度和火灾数据
    """
    global DATA_PATH  # 使用全局变量
    
    if DATA_PATH is None:
        # 如果没有设置DATA_PATH，则使用默认路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        DATA_PATH = os.path.join(project_root, 'data', 'processed')
    
    # 构建数据文件的完整路径
    humid_data_path = os.path.join(DATA_PATH, 'processed_humid_data.csv')
    fire_data_path = os.path.join(DATA_PATH, 'processed_fire_data.csv')
    
    # 检查文件是否存在
    if not os.path.exists(humid_data_path):
        raise FileNotFoundError(f"湿度数据文件不存在: {humid_data_path}")
    if not os.path.exists(fire_data_path):
        raise FileNotFoundError(f"火灾数据文件不存在: {fire_data_path}")
    
    try:
        # 使用完整路径读取数据
        humid_data = pd.read_csv(humid_data_path)
        fire_data = pd.read_csv(fire_data_path)
        
        # 创建二元分类标签
        fire_data['fire_binary'] = (fire_data['fire_count'] > 0).astype(int)
        
        # 规范化湿度数据的列名
        humid_stations = [
            'LIVERPOOL',
            'WYONG',
            'NEWCASTLE',
            'MAYFIELD',
            'TAMWORTH',
            'WOLLONGONG',
            'ALBURY',
            'PARRAMATTA NORTH',
            'MUSWELLBROOK',
            'SINGLETON',
            'MAISON DIEU',
            'CAMBERWELL',
            'SINGLETON NW',
            'MOUNT THORLEY',
            'BULGA',
            'MUSWELLBROOK NW',
            'WYBONG',
            'ABERDEEN',
            'SINGLETON SOUTH',
            'JERRYS PLAINS',
            'WARKWORTH',
            'MERRIWA',
            'CAMDEN',
            'MACQUARIE PARK',
            'ARMIDALE',
            'ORANGE',
            'GOULBURN',
            'COFFS HARBOUR',
            'MORISSET'
        ]
        
        # 重命名列
        humid_columns = {col: station for col, station in 
                        zip(humid_data.columns[1:], humid_stations)}
        humid_data = humid_data.rename(columns=humid_columns)
        
        # 确保日期格式一致
        humid_data['datetime'] = pd.to_datetime(humid_data['datetime'])
        fire_data['date'] = pd.to_datetime(fire_data['date'])
        
        # 合并数据
        merged_data = pd.merge(
            humid_data,
            fire_data[['date', 'fire_binary']],
            left_on='datetime',
            right_on='date',
            how='inner'
        )        

        print("\n合并后的数据列名:")
        print(merged_data.columns.tolist())
        
        print("\n合并后的数据预览:")
        print(merged_data[['datetime', 'fire_binary', 'LIVERPOOL', 'WYONG', 'NEWCASTLE']].head())
        
        print("\n合并后的数据形状:", merged_data.shape)
        
        return merged_data, humid_stations
        
    except Exception as e:
        raise

def split_dataset(merged_data, humid_stations, test_size=0.2, val_size=0.2, random_state=42):
    """
    将数据集分割为训练集、验证集和测试集
    
    参数:
        merged_data: 合并后的数据集
        humid_stations: 湿度站点列表
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
    feature_columns = humid_stations
    target_column = 'fire_binary'
    
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    
    X_val = val_data[feature_columns]
    y_val = val_data[target_column]
    
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    
    # 打印类别分布
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

def prepare_humid_fire_count_data():
    """
    准备湿度和火灾数量数据
    """
    global DATA_PATH
    
    if DATA_PATH is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        DATA_PATH = os.path.join(project_root, 'data', 'processed')
    
    humid_data_path = os.path.join(DATA_PATH, 'processed_humid_data.csv')
    fire_data_path = os.path.join(DATA_PATH, 'processed_fire_data.csv')
    
    if not os.path.exists(humid_data_path):
        raise FileNotFoundError(f"湿度数据文件不存在: {humid_data_path}")
    if not os.path.exists(fire_data_path):
        raise FileNotFoundError(f"火灾数据文件不存在: {fire_data_path}")
    
    try:
        humid_data = pd.read_csv(humid_data_path)
        fire_data = pd.read_csv(fire_data_path)
        
        # 规范化湿度数据的列名
        humid_stations = [
            'LIVERPOOL', 'WYONG', 'NEWCASTLE', 'MAYFIELD', 'TAMWORTH',
            'WOLLONGONG', 'ALBURY', 'PARRAMATTA NORTH', 'MUSWELLBROOK',
            'SINGLETON', 'MAISON DIEU', 'CAMBERWELL', 'SINGLETON NW',
            'MOUNT THORLEY', 'BULGA', 'MUSWELLBROOK NW', 'WYBONG',
            'ABERDEEN', 'SINGLETON SOUTH', 'JERRYS PLAINS', 'WARKWORTH',
            'MERRIWA', 'CAMDEN', 'MACQUARIE PARK', 'ARMIDALE', 'ORANGE',
            'GOULBURN', 'COFFS HARBOUR', 'MORISSET'
        ]
        
        # 重命名列
        humid_columns = {col: station for col, station in 
                        zip(humid_data.columns[1:], humid_stations)}
        humid_data = humid_data.rename(columns=humid_columns)
        
        # 确保日期格式一致
        humid_data['datetime'] = pd.to_datetime(humid_data['datetime'])
        fire_data['date'] = pd.to_datetime(fire_data['date'])
        
        # 合并数据 - 使用fire_count而不是fire_binary
        merged_data = pd.merge(
            humid_data,
            fire_data[['date', 'fire_count']],  # 这里改为fire_count
            left_on='datetime',
            right_on='date',
            how='inner'
        )
        
        print("\n合并后的数据信息:")
        print(f"数据形状: {merged_data.shape}")
        print("\n所有列名:")
        print(merged_data.columns.tolist())
        print("\n数据预览:")
        print(merged_data[['datetime', 'fire_count', 'LIVERPOOL', 'WYONG', 'NEWCASTLE']].head())
        
        return merged_data, humid_stations
        
    except Exception as e:
        raise
