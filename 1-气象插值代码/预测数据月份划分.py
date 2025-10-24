"""
Copyright (c) 2025 张德海
MIT Licensed - 详见项目根目录 LICENSE 文件

项目: Meteorology-Assisted-Crop-Yield-Prediction
仓库: https://github.com/ZDH4869/Meteorology-Assisted-Crop-Yield-Prediction.git
联系: zhangdehai1412@163.com
"""
# -*- coding: utf-8 -*-
"""
CSV文件筛选工具
根据指定列名和值筛选CSV文件中的数据，并保存到指定文件夹

功能：
1. 读取指定文件夹中的所有CSV文件
2. 根据列名和值筛选数据
3. 删除包含空值的行
4. 显示处理进度
5. 保存筛选后的数据到指定文件夹

作者：AI Assistant
日期：2024
"""

import os
import pandas as pd
import glob
from tqdm import tqdm
import argparse
import sys
from pathlib import Path
import logging
import gc

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVFilter:
    """CSV文件筛选器"""
    
    def __init__(self, input_folder, output_folder, column_name, filter_value, output_filename=None):
        """
        初始化CSV筛选器
        
        Args:
            input_folder (str): 输入文件夹路径
            output_folder (str): 输出文件夹路径
            column_name (str): 要筛选的列名
            filter_value: 要筛选的值
            output_filename (str): 输出文件名（可选）
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.column_name = column_name
        self.filter_value = filter_value
        self.output_filename = output_filename or f"filtered_{column_name}_{filter_value}.csv"
        
        # 创建输出文件夹
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # 存储所有筛选后的数据
        self.filtered_data = []
        
    def find_csv_files(self):
        """查找输入文件夹中的所有CSV文件"""
        csv_pattern = str(self.input_folder / "*.csv")
        csv_files = glob.glob(csv_pattern)
        return csv_files
    
    def get_file_encoding(self, csv_file):
        """检测文件编码"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin-1']
        
        for encoding in encodings:
            try:
                pd.read_csv(csv_file, encoding=encoding, nrows=5)
                return encoding
            except:
                continue
        return 'utf-8'
    
    def process_single_csv(self, csv_file):
        """
        处理单个CSV文件（分块处理）
        
        Args:
            csv_file (str): CSV文件路径
            
        Returns:
            pd.DataFrame: 筛选后的数据，如果没有匹配数据则返回None
        """
        try:
            # 检测文件编码
            encoding = self.get_file_encoding(csv_file)
            logger.info(f"文件 {os.path.basename(csv_file)} 使用编码: {encoding}")
            
            # 分块处理
            file_filtered_data = []
            chunk_count = 0
            
            for chunk in pd.read_csv(csv_file, encoding=encoding, chunksize=CHUNK_SIZE):
                chunk_count += 1
                logger.info(f"处理第 {chunk_count} 块，行数: {len(chunk)}")
                
                # 检查列是否存在
                if self.column_name not in chunk.columns:
                    logger.warning(f"文件 {csv_file} 中不存在列 '{self.column_name}'")
                    logger.warning(f"可用列: {list(chunk.columns)}")
                    break
                
                # 筛选数据
                filtered_chunk = chunk[chunk[self.column_name] == self.filter_value]
                
                # 删除包含空值的行
                filtered_chunk = filtered_chunk.dropna()
                
                if len(filtered_chunk) > 0:
                    # 添加源文件信息
                    filtered_chunk['source_file'] = os.path.basename(csv_file)
                    file_filtered_data.append(filtered_chunk)
                    logger.info(f"第 {chunk_count} 块筛选出 {len(filtered_chunk)} 行数据")
                
                # 强制垃圾回收
                del chunk, filtered_chunk
                gc.collect()
            
            # 合并该文件的所有筛选数据
            if file_filtered_data:
                file_combined = pd.concat(file_filtered_data, ignore_index=True)
                logger.info(f"文件 {os.path.basename(csv_file)} 总共筛选出 {len(file_combined)} 行数据")
                return file_combined
            else:
                logger.info(f"文件 {os.path.basename(csv_file)} 没有符合条件的数据")
                return None
                
        except Exception as e:
            logger.error(f"处理文件 {csv_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_all_csv_files(self):
        """处理所有CSV文件"""
        # 查找所有CSV文件
        csv_files = self.find_csv_files()
        
        if not csv_files:
            logger.warning(f"在文件夹 {self.input_folder} 中没有找到CSV文件")
            return
        
        logger.info(f"找到 {len(csv_files)} 个CSV文件")
        logger.info(f"开始筛选列 '{self.column_name}' = '{self.filter_value}' 的数据")
        
        # 使用进度条处理所有文件
        for csv_file in tqdm(csv_files, desc="处理CSV文件", unit="文件"):
            filtered_df = self.process_single_csv(csv_file)
            if filtered_df is not None:
                self.filtered_data.append(filtered_df)
        
        # 合并所有筛选后的数据
        if self.filtered_data:
            self.combined_data = pd.concat(self.filtered_data, ignore_index=True)
            logger.info(f"总共筛选出 {len(self.combined_data)} 行数据")
        else:
            logger.warning("没有找到符合条件的数据")
            self.combined_data = pd.DataFrame()
    
    def save_filtered_data(self):
        """保存筛选后的数据"""
        if self.combined_data.empty:
            logger.warning("没有数据需要保存")
            return
        
        # 构建输出文件路径
        output_file = self.output_folder / self.output_filename
        
        try:
            # 保存数据
            self.combined_data.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"筛选后的数据已保存到: {output_file}")
            logger.info(f"保存了 {len(self.combined_data)} 行数据")
            
            # 显示数据统计信息
            self.show_data_statistics()
            
        except Exception as e:
            logger.error(f"保存文件时出错: {str(e)}")
    
    def show_data_statistics(self):
        """显示数据统计信息"""
        if self.combined_data.empty:
            return
        
        print("\n" + "="*50)
        print("数据统计信息")
        print("="*50)
        print(f"总行数: {len(self.combined_data)}")
        print(f"总列数: {len(self.combined_data.columns)}")
        print(f"筛选列 '{self.column_name}' 的唯一值数量: {self.combined_data[self.column_name].nunique()}")
        print(f"源文件数量: {self.combined_data['source_file'].nunique()}")
        
        # 显示源文件统计
        print("\n各源文件数据量:")
        source_counts = self.combined_data['source_file'].value_counts()
        for file, count in source_counts.items():
            print(f"  {file}: {count} 行")
        
        print("="*50)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CSV文件筛选工具')
    parser.add_argument('--input', '-i', required=True, help='输入文件夹路径')
    parser.add_argument('--output', '-o', required=True, help='输出文件夹路径')
    parser.add_argument('--column', '-c', required=True, help='要筛选的列名')
    parser.add_argument('--value', '-v', required=True, help='要筛选的值')
    parser.add_argument('--filename', '-f', help='输出文件名（可选）')
    
    args = parser.parse_args()
    
    # 检查输入文件夹是否存在
    if not os.path.exists(args.input):
        logger.error(f"输入文件夹不存在: {args.input}")
        sys.exit(1)
    
    # 创建CSV筛选器
    csv_filter = CSVFilter(
        input_folder=args.input,
        output_folder=args.output,
        column_name=args.column,
        filter_value=args.value,
        output_filename=args.filename
    )
    
    # 处理所有CSV文件
    csv_filter.process_all_csv_files()
    
    # 保存筛选后的数据
    csv_filter.save_filtered_data()

def interactive_mode():
    """交互模式"""
    print("CSV文件筛选工具 - 交互模式")
    print("="*40)
    
    # 获取用户输入
    input_folder = input("请输入输入文件夹路径: ").strip()
    if not input_folder:
        print("输入文件夹路径不能为空")
        return
    
    if not os.path.exists(input_folder):
        print(f"输入文件夹不存在: {input_folder}")
        return
    
    output_folder = input("请输入输出文件夹路径: ").strip()
    if not output_folder:
        print("输出文件夹路径不能为空")
        return
    
    column_name = input("请输入要筛选的列名: ").strip()
    if not column_name:
        print("列名不能为空")
        return
    
    filter_value = input("请输入要筛选的值: ").strip()
    if not filter_value:
        print("筛选值不能为空")
        return
    
    output_filename = input("请输入输出文件名（可选，直接回车使用默认名称）: ").strip()
    if not output_filename:
        output_filename = None
    
    # 创建CSV筛选器
    csv_filter = CSVFilter(
        input_folder=input_folder,
        output_folder=output_folder,
        column_name=column_name,
        filter_value=filter_value,
        output_filename=output_filename
    )
    
    # 处理所有CSV文件
    csv_filter.process_all_csv_files()
    
    # 保存筛选后的数据
    csv_filter.save_filtered_data()

# =============================================================================
# 配置参数区域 - 请在这里修改您的参数
# =============================================================================

# 输入文件夹路径（包含要筛选的CSV文件）
INPUT_FOLDER = r"1-气象插值代码/used_output/csv"

# 输出文件夹路径（筛选结果保存位置）
OUTPUT_FOLDER = r"1-气象插值代码/used_output/csv/mm"

# 要筛选的列名
COLUMN_NAME = "MM"

# 要筛选的值
FILTER_VALUE = 6

# 输出文件名（可选，如果为None则使用默认名称）
OUTPUT_FILENAME = "test_weather_2014_06.csv"

# 分块大小（每次处理的行数，可根据内存情况调整）
CHUNK_SIZE = 5000000

# =============================================================================
# 主程序执行区域
# =============================================================================

def run_filter():
    """运行筛选程序"""
    print("CSV文件筛选工具")
    print("="*50)
    print(f"输入文件夹: {INPUT_FOLDER}")
    print(f"输出文件夹: {OUTPUT_FOLDER}")
    print(f"筛选列名: {COLUMN_NAME}")
    print(f"筛选值: {FILTER_VALUE}")
    print(f"输出文件名: {OUTPUT_FILENAME}")
    print("="*50)
    
    # 检查输入文件夹是否存在
    if not os.path.exists(INPUT_FOLDER):
        logger.error(f"输入文件夹不存在: {INPUT_FOLDER}")
        return
    
    # 创建CSV筛选器
    csv_filter = CSVFilter(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        column_name=COLUMN_NAME,
        filter_value=FILTER_VALUE,
        output_filename=OUTPUT_FILENAME
    )
    
    # 处理所有CSV文件
    csv_filter.process_all_csv_files()
    
    # 保存筛选后的数据
    csv_filter.save_filtered_data()
    
    print("\n筛选完成！")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 如果没有命令行参数，使用配置的参数运行
        run_filter()
    else:
        # 使用命令行参数模式
        main()
