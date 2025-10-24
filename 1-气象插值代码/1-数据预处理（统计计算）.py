"""
Copyright (c) 2025 张德海
MIT Licensed - 详见项目根目录 LICENSE 文件

项目: Meteorology-Assisted-Crop-Yield-Prediction
仓库: https://github.com/ZDH4869/Meteorology-Assisted-Crop-Yield-Prediction.git
联系: zhangdehai1412@163.com
"""
import os
import pandas as pd
from tqdm import tqdm
import chardet

# ======================== 可调参数区域 ========================

# 输入文件夹：存放多个 CSV 气象站点数据文件的目录
INPUT_FOLDER = r"1-气象插值代码/原始气象数据/气象数据"

# 输出文件夹：保存统计结果
OUTPUT_FOLDER = r"1-气象插值代码/测试数据/气象数据"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 气象站点数据 CSV 文件路径 【由于训练代码设置原因，气象站点数据坐标‘字段名字’必须为lon/lat，实际对应x/y [xy使用投影坐标系 lat/y坐标为7位数的]】
STATION_META_FILE = r"1-气象插值代码/原始气象数据/站点数据.csv"  # 包含字段：scode、name、lon、lat等

# 需要统计的字段及其统计方式
STAT_TARGETS = {
    'Tsun': ['sum', 'max', 'min', 'mean'],  # 日照时数 [累计值、最大值、最小值、平均值]
    'TAVE': ['max', 'min', 'mean'],  # 平均气温
    'Tmax': ['max', 'min', 'mean'],  # 最高气温
    'Tmin': ['max', 'min', 'mean'],  # 最低气温
    'Rain': ['sum', 'max',  'mean'],  # 降水量 'min'恒为0无训练价值
    'GTAVE': ['max', 'min', 'mean'],  # 地温平均值
    'GTmax': ['max', 'min', 'mean'],  # 地温最高值
    'GTmin': ['max', 'min', 'mean'],  # 地温最低值
    'Sevp': ['sum', 'max', 'min', 'mean'],  # 土壤蒸发量
}

# 指定时间段的统计范围
START_YEAR_MONTH = "2005-01"  # 开始年月
END_YEAR_MONTH = "2015-12"  # 结束年月


# ======================== 辅助函数区域 ========================

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))  # 检测前 10000 字节
    return result['encoding']


# 读取 CSV 文件
def read_csv_file(file_path):
    """读取 CSV 文件，自动检测编码"""
    encoding = detect_encoding(file_path)
    return pd.read_csv(file_path, encoding=encoding)


# 按指定时间段筛选数据
def filter_data_by_date_range(df, start_year_month, end_year_month):
    """按指定时间段筛选数据"""
    if not {"YYYY", "MM"}.issubset(df.columns):
        raise ValueError("数据中缺少必要的年份 YYYY 或月份 MM 字段")

    # 将 YYYY 和 MM 列转换为字符串格式
    df['YYYY'] = df['YYYY'].astype(str)
    df['MM'] = df['MM'].astype(str)

    # 创建年月组合列，格式为 YYYY-MM
    df['YEAR_MONTH'] = df.apply(lambda row: f"{row['YYYY']}-{row['MM'].zfill(2)}", axis=1)

    # 筛选指定时间段的数据
    mask = (df['YEAR_MONTH'] >= start_year_month) & (df['YEAR_MONTH'] <= end_year_month)
    return df[mask]


# 单个文件统计处理
def process_file(file_path, station_meta_df):
    df = read_csv_file(file_path)

    # 校验必要列
    if not {"YYYY", "MM", "scode"}.issubset(df.columns):
        raise ValueError("缺少必要字段：YYYY、MM、scode")

    # 筛选指定时间段的数据
    df_filtered = filter_data_by_date_range(df, START_YEAR_MONTH, END_YEAR_MONTH)

    # 如果筛选后数据为空，则跳过此文件
    if df_filtered.empty:
        print(f"文件 {os.path.basename(file_path)} 中没有符合指定时间段的数据")
        return None

    # 提取当前文件中所有的唯一 scode
    scode_vals = df_filtered['scode'].unique()
    if len(scode_vals) != 1:
        raise ValueError(f"文件 {os.path.basename(file_path)} 中包含多个或无效的 scode")
    scode_val = scode_vals[0]

    # 获取该站点的元数据
    station_row = station_meta_df[station_meta_df['scode'] == scode_val]
    if station_row.empty:
        raise ValueError(f"scode {scode_val} 不在站点信息表中")
    station_info = station_row.iloc[0].to_dict()

    # 按年-月分组并计算统计值
    df_grouped = df_filtered.groupby(["YYYY", "MM"])
    result_frames = []

    for field, methods in STAT_TARGETS.items():
        if field not in df_filtered.columns:
            continue
        temp_df = pd.DataFrame()
        for method in methods:
            if method == "max":
                stat_df = df_grouped[field].max().rename(f"{field}_max")
            elif method == "min":
                stat_df = df_grouped[field].min().rename(f"{field}_min")
            elif method == "mean":
                stat_df = df_grouped[field].mean().rename(f"{field}_mean")
            elif method == "sum":
                stat_df = df_grouped[field].sum().rename(f"{field}_sum")
            else:
                raise ValueError(f"不支持的统计方式: {method}")
            temp_df = pd.concat([temp_df, stat_df], axis=1)
        result_frames.append(temp_df)

    # 合并所有字段统计结果并添加站点元信息
    result_df = pd.concat(result_frames, axis=1).reset_index()

    # 将 scode 列插入到最前面
    result_df.insert(0, 'scode', scode_val)

    # 添加其他站点元信息
    for key, value in station_info.items():
        if key != 'scode':  # 避免重复添加 scode 列
            result_df[key] = value

    # 重命名 level_0 列为 YYYY，level_1 列为 MM
    result_df.rename(columns={'level_0': 'YYYY', 'level_1': 'MM'}, inplace=True)

    return result_df


# ======================== 主流程 ========================

def main():
    # 加载站点信息 CSV
    station_meta_df = read_csv_file(STATION_META_FILE)  # 使用修正后的读取方法
    if "scode" not in station_meta_df.columns:
        raise ValueError("站点信息表中缺少 scode 字段")

    csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]
    print(f"共找到 {len(csv_files)} 个 CSV 文件，开始处理...")

    all_results = []

    for file in tqdm(csv_files, desc="处理 CSV 文件"):
        file_path = os.path.join(INPUT_FOLDER, file)
        try:
            result_df = process_file(file_path, station_meta_df)
            if result_df is not None:
                out_file = os.path.join(OUTPUT_FOLDER, file.replace(".csv",
                                                                    f"_{START_YEAR_MONTH}_to_{END_YEAR_MONTH}_monthly_stats.csv"))
                result_df.to_csv(out_file, index=False, encoding="utf-8-sig")
                all_results.append(result_df)
        except Exception as e:
            print(f"处理文件 {file} 时出错：{e}")

    # 保存总汇总文件
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_out = os.path.join(OUTPUT_FOLDER, f"所有站点_{START_YEAR_MONTH}_to_{END_YEAR_MONTH}_汇总统计.csv")
        final_df.to_csv(final_out, index=False, encoding="utf-8-sig")
        print(f"所有站点汇总统计已保存至：{final_out}")
    else:
        print("没有可用数据生成总汇总文件。")


if __name__ == "__main__":
    main()