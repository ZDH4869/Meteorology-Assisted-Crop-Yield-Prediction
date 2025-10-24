"""
Copyright (c) 2025 张德海
MIT Licensed - 详见项目根目录 LICENSE 文件

项目: Meteorology-Assisted-Crop-Yield-Prediction
仓库: https://github.com/ZDH4869/Meteorology-Assisted-Crop-Yield-Prediction.git
联系: zhangdehai1412@163.com
"""
import os
import pandas as pd
import glob


def excel_to_csv(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中所有的 Excel 文件
    excel_files = glob.glob(os.path.join(input_folder, "*.xlsx")) + glob.glob(os.path.join(input_folder, "*.xls"))

    for excel_file in excel_files:
        try:
            # 读取 Excel 文件
            df = pd.read_excel(excel_file, engine='xlrd')  # 指定使用 xlrd 作为读取引擎

            # 构建输出的 CSV 文件路径
            file_name = os.path.basename(excel_file)
            file_name_without_extension = os.path.splitext(file_name)[0]
            csv_file_path = os.path.join(output_folder, f"{file_name_without_extension}.csv")

            # 将 DataFrame 写入 CSV 文件
            df.to_csv(csv_file_path, sep=',', lineterminator='\n', encoding='utf-8', index=False)

            # 将 DataFrame 写入 CSV 文件
            # 可修改部分：
            # sep=';' - 修改 CSV 分隔符（默认为逗号 ','
            # lineterminator='\n' - 修改换行符（默认为 '\n'，Windows 系统可能需要 '\r\n'）
            # encoding='utf-8' - 修改编码（默认为 'utf-8'）
            # index=False - 不写入行索引

            print(f"Converted {excel_file} to {csv_file_path}")

        except Exception as e:
            print(f"Error converting {excel_file}: {str(e)}")

if __name__ == "__main__":
    # 指定输入和输出文件夹路径
    input_folder = r"1-气象插值代码/原始气象数据/气象数据"  # Excel 文件所在的文件夹路径
    output_folder = r"1-气象插值代码/原始气象数据/气象数据"  # 转换后的 CSV 文件保存的文件夹路径

    excel_to_csv(input_folder, output_folder)
    print("excel转csv，全部成功！")