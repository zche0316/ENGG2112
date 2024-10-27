import pandas as pd
import csv
from datetime import datetime, timedelta

def process_fire_data(file_path):
    # 使用csv模块读取文件
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=';')
        data = list(reader)
    
    # 创建DataFrame
    columns = ['fire_id', 'fire_name', 'ignition_date', 'extinguish_date', 'ignition_cause', 'area_ha', 'unused1', 'unused2']
    df = pd.DataFrame(data[1:], columns=columns)  # 跳过第一行（标题行）

    # 转换日期格式并只保留日期部分
    df['ignition_date'] = pd.to_datetime(df['ignition_date'].str.split().str[0], format='%m/%d/%Y', errors='coerce').dt.date
    df['extinguish_date'] = pd.to_datetime(df['extinguish_date'].str.split().str[0], format='%m/%d/%Y', errors='coerce').dt.date

    # 将area_ha转换为浮点数
    df['area_ha'] = pd.to_numeric(df['area_ha'], errors='coerce')

    # 删除包含NaN的行
    df = df.dropna(subset=['ignition_date', 'extinguish_date', 'area_ha'])

    # 创建日期范围
    date_range = pd.date_range(start=df['ignition_date'].min(), end=df['extinguish_date'].max()).date

    # 初始化结果DataFrame
    result = pd.DataFrame(index=date_range, columns=['fire_count', 'total_area'])
    result = result.fillna(0.0)

    # 计算每天的火灾数量和受灾面积
    for _, row in df.iterrows():
        date_range = pd.date_range(start=row['ignition_date'], end=row['extinguish_date']).date
        days_affected = (row['extinguish_date'] - row['ignition_date']).days + 1
        daily_area = float(row['area_ha']) / days_affected
        result.loc[date_range, 'fire_count'] += 1
        result.loc[date_range, 'total_area'] += daily_area

    # 重置索引,将日期变为列
    result = result.reset_index()
    result.columns = ['date', 'fire_count', 'total_area']

    return result

if __name__ == "__main__":
    input_file_path = 'data/fire_data.csv'
    output_file_path = 'data/processed_fire_data.csv'
    
    processed_data = process_fire_data(input_file_path)
    
    # 将处理后的数据保存为CSV文件
    processed_data.to_csv(output_file_path, index=False)
    print(f"处理后的数据已保存到 {output_file_path}")

    print("\n处理后的数据示例:")
    print(processed_data.head(10))

    print("\n数据类型信息:")
    print(processed_data.dtypes)
    
    print("\n统计信息:")
    print(processed_data.describe())

    print("\n日期范围:")
    print("开始日期:", processed_data['date'].min())
    print("结束日期:", processed_data['date'].max())

    print("\n最大值:")
    print("最大火灾数量:", processed_data['fire_count'].max())
    print("最大受灾面积:", processed_data['total_area'].max())

    # 找出火灾最多的日期
    max_fire_date = processed_data.loc[processed_data['fire_count'].idxmax()]
    print("\n火灾最多的日期:")
    print(f"日期: {max_fire_date['date']}")
    print(f"火灾数量: {max_fire_date['fire_count']}")
    print(f"受灾面积: {max_fire_date['total_area']:.2f}")

    # 输出火灾最多的日期前后5天的数据
    max_fire_index = processed_data['fire_count'].idxmax()
    print("\n火灾最多日期前后5天的数据:")
    print(processed_data.iloc[max(0, max_fire_index-5):min(len(processed_data), max_fire_index+6)])
