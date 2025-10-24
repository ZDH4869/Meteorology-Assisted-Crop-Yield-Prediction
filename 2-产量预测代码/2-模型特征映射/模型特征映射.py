"""
Copyright (c) 2025 张德海
MIT Licensed - 详见项目根目录 LICENSE 文件

项目: Meteorology-Assisted-Crop-Yield-Prediction
仓库: https://github.com/ZDH4869/Meteorology-Assisted-Crop-Yield-Prediction.git
联系: zhangdehai1412@163.com
"""
# -*- coding: utf-8 -*-
"""
增强版特征名称映射工具
基于训练代码的深入分析，精确映射feature_importance_AA.csv中的feature列名到原始数据列名

特征转换过程分析：
1. Weather data: 原始列名 -> 小写转换 -> 添加'right_'前缀 -> 合并后重命名
2. Soil data: 原始列名 -> 小写转换 -> 添加'right_'前缀 -> 合并后重命名
3. 坐标列: x, y -> x_product, y_product (产品数据坐标)
4. XGBoost叶子特征: 从原始特征提取 -> 添加'feature_'前缀 + 数字编号
5. PCA特征: 降维后 -> 添加'pca_'前缀 + 数字编号
6. 最终特征顺序: 坐标 + 气象 + 土壤 + XGBoost叶子 + PCA
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class EnhancedFeatureMappingTool:
    """增强版特征名称映射工具类"""
    
    def __init__(self, feature_importance_file: str, original_feature_names_file: str, output_folder: Optional[str] = None):
        """
        初始化映射工具
        
        Args:
            feature_importance_file: feature_importance_AA.csv文件路径
            original_feature_names_file: original_feature_names.txt文件路径
            output_folder: 输出文件夹路径，如果为None则使用默认路径
        """
        self.feature_importance_file = feature_importance_file
        self.original_feature_names_file = original_feature_names_file
        self.output_folder = output_folder
        
        # 基于训练代码分析的原始数据列名定义
        self.weather_columns_original = [
            'Lon', 'Lat', 'altitude', 'YYYY', 'MM', 'Tsun_mean', 'TAVE_mean',
            'Tmax_mean', 'Tmin_mean', 'Rain_mean', 'GTAVE_mean', 'GTmax_mean', 'GTmin_mean', 'Sevp_mean'
        ]
        
        self.soil_columns_original = [
            'x', 'y', 'TZ', 'CEC', 'PH', 'SOC', 'SOCD', 'TK', 'TND', 'TPD'
        ]
        
        # 产品数据列名
        self.product_columns_original = [
            'x', 'y', 'SUIT', 'YYYY', 'per_mu', 'per_qu'
        ]
        
        # 加载特征重要性数据
        self.feature_importance_df = None
        self.original_feature_names = []
        self.load_data()
        
        # 创建特征映射字典
        self.feature_mapping = self.create_detailed_mapping()
        
    def load_data(self):
        """加载特征重要性数据和原始特征名称"""
        try:
            # 加载特征重要性数据
            self.feature_importance_df = pd.read_csv(self.feature_importance_file, encoding='utf-8')
            print(f"✅ 成功加载特征重要性数据: {len(self.feature_importance_df)} 个特征")
            
            # 加载原始特征名称
            with open(self.original_feature_names_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 解析原始特征名称
            lines = content.strip().split('\n')
            for line in lines:
                if line.strip() and not line.startswith('=') and not line.startswith('原始'):
                    # 提取特征名称（去掉序号）
                    if '. ' in line:
                        feature_name = line.split('. ', 1)[1].strip()
                        self.original_feature_names.append(feature_name)
            
            print(f"✅ 成功加载原始特征名称: {len(self.original_feature_names)} 个特征")
            
        except Exception as e:
            print(f"❌ 加载数据失败: {str(e)}")
            raise
    
    def create_detailed_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        创建详细的特征映射字典
        
        Returns:
            Dict[str, Dict[str, str]]: 映射字典，包含原始列名、数据源、特征类型等信息
        """
        mapping = {}
        
        # 基于训练代码分析的特征顺序和转换过程
        # 特征顺序：坐标列(2) + 气象特征(动态) + 土壤特征(动态) + XGBoost叶子特征(50) + PCA特征(1)
        
        for i, feature_name in enumerate(self.original_feature_names):
            feature_info = {
                'original_column_name': '',
                'data_source': 'unknown',
                'feature_type': 'unknown',
                'description': '',
                'index': i
            }
            
            if feature_name.isdigit():
                # 数字索引特征，根据位置映射
                idx = int(feature_name)
                feature_info.update(self._map_numeric_feature_detailed(idx))
            elif feature_name.startswith('feature_'):
                # XGBoost叶子特征
                feature_info.update({
                    'original_column_name': f"XGBoost_leaf_{feature_name}",
                    'data_source': 'xgboost',
                    'feature_type': 'xgboost_leaf',
                    'description': f"XGBoost模型叶子节点特征，编号{feature_name.split('_')[1]}"
                })
            elif feature_name.startswith('pca_'):
                # PCA特征
                feature_info.update({
                    'original_column_name': f"PCA_{feature_name}",
                    'data_source': 'pca',
                    'feature_type': 'pca_component',
                    'description': f"PCA降维后的主成分，编号{feature_name.split('_')[1]}"
                })
            else:
                feature_info.update({
                    'original_column_name': f"unknown_{feature_name}",
                    'data_source': 'unknown',
                    'feature_type': 'unknown',
                    'description': f"未知特征类型: {feature_name}"
                })
            
            mapping[feature_name] = feature_info
        
        return mapping
    
    def _map_numeric_feature_detailed(self, idx: int) -> Dict[str, str]:
        """
        将数字索引映射到详细的原始列名信息
        
        Args:
            idx: 特征索引
            
        Returns:
            Dict[str, str]: 特征详细信息
        """
        # 基于训练代码分析的特征顺序
        # 0-1: 坐标列
        if idx == 0:
            return {
                'original_column_name': 'x_product',
                'data_source': 'coordinate',
                'feature_type': 'coordinate',
                'description': '产品数据x坐标（经度）'
            }
        elif idx == 1:
            return {
                'original_column_name': 'y_product',
                'data_source': 'coordinate',
                'feature_type': 'coordinate',
                'description': '产品数据y坐标（纬度）'
            }
        
        # 2-30: 气象特征 (动态计算特征数量)
        elif 2 <= idx <= (1 + len(self.weather_columns_original)):
            weather_idx = idx - 2
            if weather_idx < len(self.weather_columns_original):
                original_name = self.weather_columns_original[weather_idx]
                # 转换为小写并添加right_前缀（基于训练代码）
                mapped_name = f"right_{original_name.lower()}"
                
                # 根据列名确定气象特征类型
                if 'tsun' in original_name.lower():
                    feature_type = 'sunshine'
                elif 'tave' in original_name.lower():
                    feature_type = 'temperature_avg'
                elif 'tmax' in original_name.lower():
                    feature_type = 'temperature_max'
                elif 'tmin' in original_name.lower():
                    feature_type = 'temperature_min'
                elif 'rain' in original_name.lower():
                    feature_type = 'precipitation'
                elif 'gtave' in original_name.lower():
                    feature_type = 'ground_temperature_avg'
                elif 'gtmax' in original_name.lower():
                    feature_type = 'ground_temperature_max'
                elif 'gtmin' in original_name.lower():
                    feature_type = 'ground_temperature_min'
                elif 'sevp' in original_name.lower():
                    feature_type = 'evaporation'
                elif 'altitude' in original_name.lower():
                    feature_type = 'elevation'
                elif 'yyyy' in original_name.lower():
                    feature_type = 'year'
                elif 'mm' in original_name.lower():
                    feature_type = 'month'
                elif 'lon' in original_name.lower():
                    feature_type = 'longitude'
                elif 'lat' in original_name.lower():
                    feature_type = 'latitude'
                else:
                    feature_type = 'weather_other'
                
                return {
                    'original_column_name': mapped_name,
                    'data_source': 'weather',
                    'feature_type': feature_type,
                    'description': f"气象数据: {original_name} -> {mapped_name}"
                }

        # 土壤特征 (动态计算特征数量)
        elif (2 + len(self.weather_columns_original)) <= idx <= (1 + len(self.weather_columns_original) + len(self.soil_columns_original)):
            soil_idx = idx - (2 + len(self.weather_columns_original))
            if soil_idx < len(self.soil_columns_original):
                original_name = self.soil_columns_original[soil_idx]
                # 转换为小写并添加right_前缀
                mapped_name = f"right_{original_name.lower()}"

                # 根据列名确定土壤特征类型
                if original_name == 'TZ':
                    feature_type = 'soil_texture'
                elif original_name == 'CEC':
                    feature_type = 'soil_cation_exchange'
                elif original_name == 'PH':
                    feature_type = 'soil_ph'
                elif original_name in ['SOC', 'SOCD']:
                    feature_type = 'soil_organic_carbon'
                elif original_name == 'TK':
                    feature_type = 'soil_potassium'
                elif original_name == 'TND':
                    feature_type = 'soil_nitrogen'
                elif original_name == 'TPD':
                    feature_type = 'soil_phosphorus'
                elif original_name in ['x', 'y']:
                    feature_type = 'coordinate'
                else:
                    feature_type = 'soil_other'

                return {
                    'original_column_name': mapped_name,
                    'data_source': 'soil',
                    'feature_type': feature_type,
                    'description': f"土壤数据: {original_name} -> {mapped_name}"
                }
        
        # 其他特征
        else:
            return {
                'original_column_name': f"unknown_feature_{idx}",
                'data_source': 'unknown',
                'feature_type': 'unknown',
                'description': f"未知特征，索引{idx}"
            }
    
    def generate_mapping_results(self) -> pd.DataFrame:
        """
        生成完整的映射结果DataFrame
        
        Returns:
            pd.DataFrame: 包含所有映射信息的DataFrame
        """
        results = []
        
        for _, row in self.feature_importance_df.iterrows():
            feature_name = row['feature']
            importance = row['importance']
            relative_importance = row['relative_importance']
            
            if feature_name in self.feature_mapping:
                mapping_info = self.feature_mapping[feature_name]
                results.append({
                    'feature': feature_name,
                    'importance': importance,
                    'relative_importance': relative_importance,
                    'original_column_name': mapping_info['original_column_name'],
                    'data_source': mapping_info['data_source'],
                    'feature_type': mapping_info['feature_type'],
                    'description': mapping_info['description'],
                    'index': mapping_info['index']
                })
            else:
                results.append({
                    'feature': feature_name,
                    'importance': importance,
                    'relative_importance': relative_importance,
                    'original_column_name': f"unknown_{feature_name}",
                    'data_source': 'unknown',
                    'feature_type': 'unknown',
                    'description': f"未找到映射信息: {feature_name}",
                    'index': -1
                })
        
        return pd.DataFrame(results)
    
    def generate_detailed_report(self) -> str:
        """
        生成详细的映射报告
        
        Returns:
            str: 映射报告
        """
        mapped_df = self.generate_mapping_results()
        
        report = []
        report.append("=" * 100)
        report.append("增强版特征名称映射报告")
        report.append("基于训练代码深度分析的特征转换过程")
        report.append("=" * 100)
        report.append("")
        
        # 统计信息
        report.append("1. 特征统计信息:")
        report.append(f"   总特征数: {len(mapped_df)}")
        
        # 按数据源统计
        data_source_stats = mapped_df['data_source'].value_counts()
        for source, count in data_source_stats.items():
            report.append(f"   {source.upper()}: {count} 个特征")
        
        report.append("")
        
        # 按特征类型统计
        report.append("2. 按特征类型统计:")
        feature_type_stats = mapped_df['feature_type'].value_counts()
        for ftype, count in feature_type_stats.items():
            report.append(f"   {ftype.upper()}: {count} 个特征")
        
        report.append("")
        
        # 最重要的特征映射
        report.append("3. 前20个最重要的特征映射:")
        report.append("-" * 100)
        report.append(f"{'排名':<4} {'特征名':<15} {'原始列名':<25} {'数据源':<10} {'特征类型':<15} {'重要性':<10}")
        report.append("-" * 100)
        
        for idx, row in mapped_df.head(20).iterrows():
            rank = idx + 1
            feature = row['feature']
            original = row['original_column_name']
            source = row['data_source']
            ftype = row['feature_type']
            importance = f"{row['relative_importance']:.6f}"
            
            report.append(f"{rank:<4} {feature:<15} {original:<25} {source:<10} {ftype:<15} {importance:<10}")
        
        report.append("")
        
        # 按数据源分组的详细特征
        report.append("4. 按数据源分组的详细特征:")
        for source in ['weather', 'soil', 'coordinate', 'xgboost', 'pca', 'unknown']:
            source_features = mapped_df[mapped_df['data_source'] == source]
            if len(source_features) > 0:
                report.append(f"\n{source.upper()} 数据源 ({len(source_features)} 个特征):")
                report.append("-" * 80)
                
                for _, row in source_features.iterrows():
                    report.append(f"  {row['feature']} -> {row['original_column_name']}")
                    report.append(f"    类型: {row['feature_type']}, 重要性: {row['relative_importance']:.6f}")
                    report.append(f"    描述: {row['description']}")
                    report.append("")
        
        # 气象特征详细分析
        weather_features = mapped_df[mapped_df['data_source'] == 'weather']
        if len(weather_features) > 0:
            report.append("5. 气象特征详细分析:")
            report.append("-" * 80)
            
            # 按气象特征类型分组
            weather_type_stats = weather_features['feature_type'].value_counts()
            for wtype, count in weather_type_stats.items():
                report.append(f"  {wtype.upper()}: {count} 个特征")
                type_features = weather_features[weather_features['feature_type'] == wtype]
                for _, row in type_features.iterrows():
                    report.append(f"    {row['feature']} -> {row['original_column_name']} (重要性: {row['relative_importance']:.6f})")
                report.append("")
        
        # 土壤特征详细分析
        soil_features = mapped_df[mapped_df['data_source'] == 'soil']
        if len(soil_features) > 0:
            report.append("6. 土壤特征详细分析:")
            report.append("-" * 80)
            
            # 按土壤特征类型分组
            soil_type_stats = soil_features['feature_type'].value_counts()
            for stype, count in soil_type_stats.items():
                report.append(f"  {stype.upper()}: {count} 个特征")
                type_features = soil_features[soil_features['feature_type'] == stype]
                for _, row in type_features.iterrows():
                    report.append(f"    {row['feature']} -> {row['original_column_name']} (重要性: {row['relative_importance']:.6f})")
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self, output_csv: str = None, output_report: str = None):
        """
        保存映射结果
        
        Args:
            output_csv: CSV输出文件路径，如果为None则自动生成
            output_report: 报告输出文件路径，如果为None则自动生成
        """
        # 生成结果
        mapped_df = self.generate_mapping_results()
        
        # 确定输出文件路径
        if output_csv is None:
            output_csv = os.path.join(self.output_folder or ".", "enhanced_feature_mapping_results.csv")
        if output_report is None:
            output_report = os.path.join(self.output_folder or ".", "enhanced_feature_mapping_report.txt")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存CSV
        mapped_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"✅ 增强版映射结果已保存至: {output_csv}")
        
        # 保存详细报告
        report = self.generate_detailed_report()
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ 增强版详细报告已保存至: {output_report}")
        
        return mapped_df


def main():
    """主函数"""
    # 文件路径
    feature_importance_file = r"2. B774模型脚本\2-适宜度产量预测模型\model_test_06\feature_importanceAA\feature_importance_AA.csv"
    original_feature_names_file = r"2. B774模型脚本\2-适宜度产量预测模型\model_test_06\feature_importanceAA\original_feature_names.txt"
    
    # 参数设置：输出文件夹
    output_folder = r"2. B774模型脚本\2-适宜度产量预测模型\model_test_06\feature_importanceAA"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_folder = os.path.join(output_folder, f"mapping_results_{timestamp}")
    
    # 创建输出文件夹
    os.makedirs(timestamped_folder, exist_ok=True)
    print(f"📁 输出文件夹: {timestamped_folder}")
    
    # 检查文件是否存在
    if not os.path.exists(feature_importance_file):
        print(f"❌ 特征重要性文件不存在: {feature_importance_file}")
        return
    
    if not os.path.exists(original_feature_names_file):
        print(f"❌ 原始特征名称文件不存在: {original_feature_names_file}")
        return
    
    try:
        # 创建增强版映射工具
        print("🔧 创建增强版特征映射工具...")
        mapping_tool = EnhancedFeatureMappingTool(feature_importance_file, original_feature_names_file, timestamped_folder)
        
        # 生成映射结果
        print("📊 生成增强版特征映射结果...")
        mapped_df = mapping_tool.save_results()
        
        # 显示前10个最重要的特征映射
        print("\n📋 前10个最重要的特征映射:")
        print("-" * 100)
        print(f"{'排名':<4} {'特征名':<15} {'原始列名':<25} {'数据源':<10} {'特征类型':<15} {'重要性':<10}")
        print("-" * 100)
        
        for idx, row in mapped_df.head(10).iterrows():
            rank = idx + 1
            feature = row['feature']
            original = row['original_column_name']
            source = row['data_source']
            ftype = row['feature_type']
            importance = f"{row['relative_importance']:.6f}"
            print(f"{rank:<4} {feature:<15} {original:<25} {source:<10} {ftype:<15} {importance:<10}")
        
        # 生成统计摘要
        print("\n📈 特征映射统计摘要:")
        print(f"   总特征数: {len(mapped_df)}")
        
        # 按数据源统计
        data_source_stats = mapped_df['data_source'].value_counts()
        for source, count in data_source_stats.items():
            print(f"   {source.upper()}: {count} 个特征")
        
        # 按特征类型统计
        print("\n📊 按特征类型统计:")
        feature_type_stats = mapped_df['feature_type'].value_counts()
        for ftype, count in feature_type_stats.items():
            print(f"   {ftype.upper()}: {count} 个特征")
        
        print("\n✅ 增强版特征映射完成！")
        print("📁 输出文件夹:")
        print(f"   {timestamped_folder}")
        print("📄 输出文件:")
        print("   - enhanced_feature_mapping_results.csv: 详细映射结果")
        print("   - enhanced_feature_mapping_report.txt: 完整映射报告")
        
    except Exception as e:
        print(f"❌ 特征映射失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
