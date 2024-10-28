# 添加必要的导入
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def prepare_humid_fire_data():
    # 1. 读取数据
    humid_data = pd.read_csv('data/processed/processed_humid_data.csv')
    fire_data = pd.read_csv('data/processed/processed_fire_data.csv')
    
    # 2. 规范化湿度数据的列名
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
    
    # 3. 重命名列
    humid_columns = {col: station for col, station in 
                    zip(humid_data.columns[1:], humid_stations)}
    humid_data = humid_data.rename(columns=humid_columns)
    
    # 4. 打印前几个站点的数据样本
    print("湿度数据前3个站点的前5行:")
    print(humid_data[['datetime', 'LIVERPOOL', 'WYONG', 'NEWCASTLE']].head())
    
    # 5. 确保日期格式一致
    humid_data['datetime'] = pd.to_datetime(humid_data['datetime'])
    fire_data['date'] = pd.to_datetime(fire_data['date'])
    
    # 6. 合并数据
    merged_data = pd.merge(
        humid_data,
        fire_data,
        left_on='datetime',
        right_on='date',
        how='inner'
    )
    
    # 7. 检查数据完整性
    print("\n每个站点的数据完整性:")
    for station in humid_stations:
        missing = merged_data[station].isna().sum()
        total = len(merged_data)
        print(f"{station}: {total-missing}/{total} 条数据 "
              f"({((total-missing)/total*100):.2f}% 完整)")
    
    return merged_data, humid_stations

# 运行数据准备
merged_data, stations = prepare_humid_fire_data()

# 显示合并后的数据概览
print("\n合并后的数据概览:")
print(merged_data.head())
print("\n数据形状:", merged_data.shape)
