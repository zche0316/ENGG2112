import pandas as pd

def process_weather_data(input_file_path, output_file_path):
    # 读取CSV文件，跳过前两行，使用第三行作为列名
    df = pd.read_csv(input_file_path, sep=';', encoding='utf-8', skiprows=2, header=0)

    # 处理24:00的情况
    df['Time'] = df['Time'].replace('24:00', '00:00')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.loc[df['Time'] == '00:00', 'Date'] += pd.Timedelta(days=1)

    # 合并日期和时间列
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'], format='%Y-%m-%d %H:%M')

    # 设置日期时间为索引
    df.set_index('datetime', inplace=True)

    # 删除原始的日期和时间列
    df.drop(['Date', 'Time'], axis=1, inplace=True)

    # 计算每天的平均值
    daily_avg = df.resample('D').mean()

    # 修改列名为 "1d average"
    daily_avg.columns = [f"{col} 1d average" for col in daily_avg.columns]

    # 保存处理后的数据
    daily_avg.to_csv(output_file_path)

    print(f"处理后的数据已保存到 {output_file_path}")

    return daily_avg

# 处理湿度数据
input_file_path = 'data/humid_data.csv'
output_file_path = 'data/processed_humid_data.csv'
processed_humid_data = process_weather_data(input_file_path, output_file_path)

print("\n处理后的数据示例:")
print(processed_humid_data.head())

print("\n数据类型信息:")
print(processed_humid_data.dtypes)

print("\n统计信息:")
print(processed_humid_data.describe())

print("\n日期范围:")
print("开始日期:", processed_humid_data.index.min())
print("结束日期:", processed_humid_data.index.max())

# 处理温度数据
input_file_path = 'data/temp_data.csv'
output_file_path = 'data/processed_temp_data.csv'
processed_temp_data = process_weather_data(input_file_path, output_file_path)

# 处理风速数据
input_file_path = 'data/wind_data.csv'
output_file_path = 'data/processed_wind_data.csv'
processed_wind_data = process_weather_data(input_file_path, output_file_path)
