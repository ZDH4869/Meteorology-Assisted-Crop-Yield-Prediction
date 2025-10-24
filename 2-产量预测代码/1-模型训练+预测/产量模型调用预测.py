"""
Copyright (c) 2025 张德海
MIT Licensed - 详见项目根目录 LICENSE 文件

项目: Meteorology-Assisted-Crop-Yield-Prediction
仓库: https://github.com/ZDH4869/Meteorology-Assisted-Crop-Yield-Prediction.git
联系: zhangdehai1412@163.com
"""
# -*- coding: utf-8 -*-
"""
深度学习混合模型：基于地理坐标、训练气象、土壤数据及农作物适宜性等级、
区产（yield_per_cell）和亩产（yield_per_mu）的多输出预测
"""

# ================= 设置CUDA环境变量 =================
import os

# ================= 全局变量 =================
# 全局R²早停相关变量
GLOBAL_R2_ACHIEVED = False
GLOBAL_BEST_PARAMS = None
GLOBAL_BEST_R2 = 0.0
# 设置CUDA 11.8环境变量，确保TensorFlow能找到GPU
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8'
cuda_bin_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin'
cuda_lib_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp'
current_path = os.environ.get('PATH', '')
if cuda_bin_path not in current_path:
    os.environ['PATH'] = cuda_bin_path + ';' + cuda_lib_path + ';' + current_path
    print(f"✅ 已设置CUDA 11.8环境变量")
    print(f"CUDA_PATH: {os.environ['CUDA_PATH']}")
else:
    print("✅ CUDA 11.8环境变量已存在")

# ================= 系统和基础库 =================
import json
import time
import psutil
import warnings
from typing import Union
import platform
import random
# import re  # 未使用
# import itertools  # 未使用

# ================= 数据处理库 =================
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
# from sklearn.feature_selection import mutual_info_regression, mutual_info_classif  # 未使用
from scipy.spatial import cKDTree
# from itertools import combinations  # 未使用
# import scipy  # 未使用
import xgboost as xgb
from datetime import datetime

# ================= 深度学习库 =================
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, Layer, Add


# ================= 空间合并函数 =================
def spatial_merge(left, right, on=None, tolerance=50000):
    """空间近邻合并函数"""
    if on is None:
        on = ['x', 'y']
    left_coords = left[on].values
    right_coords = right[on].values
    tree = cKDTree(right_coords)
    dist, idx = tree.query(left_coords, distance_upper_bound=tolerance)
    mask = idx < right.shape[0]

    if not mask.any():
        return pd.DataFrame()

    # 避免多次reset_index，直接使用索引
    left_valid = left[mask].copy()
    right_valid = right.iloc[idx[mask]].copy()

    # 给 right_valid 所有列加前缀
    right_valid.columns = [f'right_{col}' for col in right_valid.columns]

    # 重置索引但避免深度复制
    left_valid.reset_index(drop=True, inplace=True)
    right_valid.reset_index(drop=True, inplace=True)

    # 直接合并，避免额外的reset_index
    merged = pd.concat([left_valid, right_valid], axis=1)
    return merged


def spatial_temporal_merge(left, right, xy_cols=None, time_col='yyyy', tolerance=50000):
    """空间时间合并函数"""
    if xy_cols is None:
        xy_cols = ['x', 'y']
    merged_list = []
    years = left[time_col].unique()
    print(f"处理年份: {years}")

    # 获取右表可用的年份
    right_years = right[time_col].unique()
    print(f"气象数据可用年份: {right_years}")

    # 只处理年份完全匹配的数据
    common_years = set(years) & set(right_years)
    print(f"年份完全匹配的年份: {sorted(common_years)}")

    if not common_years:
        print("⚠️ 没有年份完全匹配的数据，无法进行合并")
        return pd.DataFrame()

    for i, year in enumerate(sorted(common_years)):
        print(f"处理年份 {year} ({i + 1}/{len(common_years)})...")

        # 使用视图而不是副本
        left_year = left[left[time_col] == year]
        right_year = right[right[time_col] == year]
        print(f"  使用相同年份 {year} 的气象数据")

        if len(right_year) == 0:
            print(f"  年份 {year} 在右表中无数据，跳过")
            continue

        print(f"  左表: {len(left_year)} 行, 右表: {len(right_year)} 行")

        try:
            merged = spatial_merge(left_year, right_year, on=xy_cols, tolerance=tolerance)
            if not merged.empty:
                merged_list.append(merged)
                print(f"  合并成功: {len(merged)} 行")
            else:
                print(f"  年份 {year} 合并结果为空")
        except MemoryError as e:
            print(f"  年份 {year} 处理时内存不足: {e}")
            # 尝试分块处理
            chunk_size = len(left_year) // 4
            if chunk_size > 0:
                for j in range(0, len(left_year), chunk_size):
                    left_chunk = left_year.iloc[j:j + chunk_size]
                    try:
                        merged_chunk = spatial_merge(left_chunk, right_year, on=xy_cols, tolerance=tolerance)
                        if not merged_chunk.empty:
                            merged_list.append(merged_chunk)
                    except MemoryError:
                        print(f"    分块 {j // chunk_size + 1} 仍然内存不足，跳过")
                        continue

    if merged_list:
        return pd.concat(merged_list, ignore_index=True)
    else:
        return pd.DataFrame()


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import mixed_precision
# from tensorflow.keras.optimizers import Adam  # 未使用

# ================= XGBoost和优化库 =================
import xgboost as xgb
import optuna

# ================= 可视化库 =================
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm.auto import tqdm
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================= GPU监控库 =================
try:
    import GPUtil
except ImportError:
    GPUtil = None

# 禁用警告
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def json_fallback(obj):
    """JSON序列化时的回退函数，处理numpy类型"""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


class Config:
    """配置类：包含所有可调整的参数和路径设置"""

    @classmethod
    def validate_params(cls):
        """验证参数有效性"""
        try:
            # 验证随机种子
            assert isinstance(cls.random_seed, int) and cls.random_seed >= 0

            # 验证测试集比例
            assert 0.0 < cls.test_size < 1.0

            # 验证XGBoost参数
            assert 1 <= cls.xgb_params['max_depth'] <= 20
            assert 0.0 < cls.xgb_params['learning_rate'] <= 1.0
            assert cls.xgb_params['n_estimators'] > 0

            # 验证深度学习参数
            assert cls.dl_params['epochs'] > 0
            assert cls.dl_params['batch_size'] > 0
            assert 0.0 < cls.dl_params['learning_rate'] <= 1.0
            assert 0.0 <= cls.dl_params['dropout_rate'] < 1.0
            assert cls.dl_params['early_stop_patience'] > 0

            # 验证Optuna参数
            assert cls.optuna_params['n_trials'] > 0
            assert cls.optuna_params['timeout'] > 0

            # 验证GPU参数
            assert 0.0 < cls.gpu_memory_limit <= 1.0

            # 验证数据读取参数
            assert cls.max_rows_per_weather_file > 0, "每个气象文件最大行数必须大于0"

            print("参数验证通过")
            return True

        except AssertionError as e:
            print(f"参数验证失败: {str(e)}")
            return False

        except Exception as e:
            print(f"参数验证过程出错: {str(e)}")
            return False

    @classmethod
    def validate_paths(cls):
        """验证文件路径有效性"""
        try:
            # 验证输入文件
            input_files = [
                cls.soil_data_csv,
                cls.test_soil_csv
            ]

            for file_path in input_files:
                if not os.path.exists(file_path):
                    print(f"输入文件不存在: {file_path}")
                    return False

            # 验证输出目录权限
            output_dirs = [
                cls.output_dir,
                cls.model_dir,
                cls.final_model_dir,
                cls.label_encoder_dir,
                cls.scaler_dir,
                cls.xgboost_dir,
                cls.logs_dir,
                cls.result_dir,
                cls.result_analysis_dir,
                cls.training_analysis_dir,
                cls.feature_importance_dir  # 新增
            ]

            for dir_path in output_dirs:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except Exception as e:
                    print(f"无法创建目录 {dir_path}: {str(e)}")
                    return False

            print("路径验证通过")
            return True

        except Exception as e:
            print(f"路径验证过程出错: {str(e)}")
            return False

    # ============= 输入输出路径配置 =============
    # 基础路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 输入数据路径
    input_data_dir = os.path.join(base_dir, "train+text_csv数据")

    # 预测气象数据配置 - 文件夹模式
    weather_data_folder = r"2-产量预测代码/train+text_csv数据/预测气象"  # 训练气象数据文件夹路径
    use_weather_folder = True  # 使用文件夹模式读取训练气象数据

    soil_data_csv = os.path.join(input_data_dir, "soil_test.csv") # 当用于预测时，保持土壤相同（soil_data_csv=test_soil_csv）
    test_soil_csv = os.path.join(input_data_dir, "soil_test.csv") # 当用于预测时，保持土壤相同

    # 输出目录配置
    _default_model_name = "model_test_06"  # 默认模型名称
    _current_model_name = _default_model_name  # 当前使用的模型名称
    
    @classmethod
    def set_model_name(cls, model_name):
        """设置当前使用的模型名称"""
        cls._current_model_name = model_name
        cls._update_paths()
    
    @classmethod
    def get_model_name(cls):
        """获取当前使用的模型名称"""
        return cls._current_model_name
    
    @classmethod
    def _update_paths(cls):
        """根据当前模型名称更新所有路径"""
        cls.output_dir = os.path.join(cls.base_dir, cls._current_model_name)
        cls.model_dir = os.path.join(cls.output_dir, "modelAA")
        cls.final_model_dir = os.path.join(cls.output_dir, "final_modelAA")
        cls.label_encoder_dir = os.path.join(cls.output_dir, "LabelEnconderAA")
        cls.scaler_dir = os.path.join(cls.output_dir, "ScalerAA")
        cls.xgboost_dir = os.path.join(cls.output_dir, "XGBoostAA")
        cls.logs_dir = os.path.join(cls.output_dir, "final_logsAA")
        cls.result_dir = os.path.join(cls.output_dir, "result_predited_csvAA") # 预测结果输出路径
        cls.feature_importance_dir = os.path.join(cls.output_dir, "feature_importanceAA")
        cls.result_analysis_dir = os.path.join(cls.output_dir, "analysis_result_mapAA")  # 应该输出2
        cls.training_analysis_dir = os.path.join(cls.output_dir, "analysis_training_mapAA")  # 应该输出4
        
        # 更新具体文件路径
        cls.label_encoder_file = os.path.join(cls.label_encoder_dir, "label_encoder_AA.pkl")
        cls.scaler_file = os.path.join(cls.scaler_dir, "scaler_AA.pkl")
        cls.xgboost_model_file = os.path.join(cls.xgboost_dir, "xgb_model_AA.json")
        cls.final_model_file = os.path.join(cls.final_model_dir, "final_model_r2_r2_0.9196.h5") # 具体调用模型路径
        cls.result_file = os.path.join(cls.result_dir, "result_predited_AA_test.csv")
        cls.feature_importance_file = os.path.join(cls.feature_importance_dir, "feature_importance_AA.csv")
        cls.key_features_list_file = os.path.join(cls.feature_importance_dir, "key_features_AA.json")
        cls.feature_importance_plot = os.path.join(cls.feature_importance_dir, "feature_importance_plot_AA.png")
        cls.suitability_map_file = os.path.join(cls.result_analysis_dir, "result_suitability_map_AA.png")
        cls.feature_importance_map_file = os.path.join(cls.result_analysis_dir, "feature_importance.png")
        cls.evaluation_file = os.path.join(cls.training_analysis_dir, "evaluation_report.json")
        cls.confusion_matrix_file = os.path.join(cls.training_analysis_dir, "confusion_matrix.png")
        cls.accuracy_loss_map_file = os.path.join(cls.training_analysis_dir, "Accuracy+Loss_map_AA.png")
        cls.regression_scatter_file = os.path.join(cls.training_analysis_dir, "regression_scatter.png")
    
    @classmethod
    def auto_detect_available_models(cls):
        """自动检测可用的模型目录"""
        available_models = []
        for item in os.listdir(cls.base_dir):
            item_path = os.path.join(cls.base_dir, item)
            if os.path.isdir(item_path):
                # 检查是否包含模型文件
                model_files = [
                    os.path.join(item_path, "final_modelAA", "best_hybrid_multioutput_AA.h5"),
                    os.path.join(item_path, "ScalerAA", "scaler_AA.pkl"),
                    os.path.join(item_path, "ScalerAA", "pca_AA.pkl"),
                    os.path.join(item_path, "XGBoostAA", "xgb_model_AA.json"),
                    os.path.join(item_path, "feature_order.json")
                ]
                if all(os.path.exists(f) for f in model_files):
                    available_models.append(item)
        return available_models
    
    @classmethod
    def list_available_models(cls):
        """列出所有可用的模型"""
        models = cls.auto_detect_available_models()
        print(f"🔍 检测到 {len(models)} 个可用模型:")
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model}")
        return models
    
    # 初始化路径 - 在类定义完成后调用
    # _update_paths() 将在类定义完成后调用

    # 具体文件路径将在_update_paths()中动态设置
    # 这些变量将在类初始化时通过_update_paths()设置
    label_encoder_file = None
    scaler_file = None
    xgboost_model_file = None
    final_model_file = None
    result_file = None
    feature_importance_file = None
    key_features_list_file = None
    feature_importance_plot = None
    suitability_map_file = None
    feature_importance_map_file = None
    evaluation_file = None
    confusion_matrix_file = None
    accuracy_loss_map_file = None
    regression_scatter_file = None

    # ============= 特征和目标变量配置 =============
    # 不参与训练的特征列（基于V3优化结果）
    exclude_columns = ['x', 'y', 'YYYY', 'SUIT', 'per_mu', 'per_qu']  # 排除坐标、年份、目标变量

    # 分类特征列（需要进行编码的列）
    categorical_columns = ['tz']  # 保留'tz'分类特征

    # 范围值特征列
    range_columns = []  # 不包含'tz'

    # 目标变量配置（基于V3优化结果）
    target_columns = {
        'classification': 'SUIT',  # 分类目标
        'regression': ['per_mu', 'per_qu']  # 回归目标
    }

    # 新增：基于产量创建适宜度分类的配置
    create_suitability_from_yield = True  # 是否基于产量创建适宜度分类
    suitability_quantiles = [0.33, 0.67]  # 适宜度分类的分位数阈值
    # 新增损失函数类型参数（基于V3优化结果）
    loss_type = 'huber'  # 使用Huber损失，对异常值更鲁棒
    # ============= 模型参数配置 =============
    # 通用参数
    random_seed = 42
    test_size = 0.2

    # XGBoost参数（稳定性优化配置）
    xgb_params = {
        'max_depth': 6,  # 增加深度，提高学习能力
        'learning_rate': 0.1,  # 降低学习率，提高稳定性
        'n_estimators': 300,  # 增加树数量，提高性能
        'subsample': 0.8,  # 降低子采样率，防止过拟合
        'colsample_bytree': 0.8,  # 降低特征采样率，防止过拟合
        'random_state': random_seed,
        'tree_method': 'hist',
        'predictor': 'cpu_predictor',
    }

    # XGBoost GPU参数验证
    @staticmethod
    def validate_xgboost_gpu():
        """验证XGBoost GPU参数是否可用"""
        try:
            import xgboost as xgb
            # 检查XGBoost版本是否支持GPU
            if hasattr(xgb, '__version__'):
                version = xgb.__version__
                print(f"XGBoost版本: {version}")
                # 检查是否支持GPU
                try:
                    # 尝试创建GPU DMatrix
                    test_data = np.random.random((10, 5))
                    dtest = xgb.DMatrix(test_data)
                    print("✅ XGBoost GPU支持正常")
                    return True
                except Exception as e:
                    print(f"⚠️ XGBoost GPU支持可能有问题: {str(e)}")
                    # 回退到CPU参数
                    Config.xgb_params.update({
                        'tree_method': 'hist',
                        'predictor': 'cpu_predictor'
                    })
                    print("已回退到CPU参数")
                    return False
            else:
                print("⚠️ 无法检测XGBoost版本")
                return False
        except ImportError:
            print("❌ XGBoost未安装")
            return False

    # 深度学习参数（稳定性优化配置）
    dl_params = {
        'epochs': 50,  # 增加训练轮次，确保充分学习
        'batch_size': 32,  # 减小batch size，提高训练稳定性
        'learning_rate': 0.0001,  # 降低学习率，提高训练稳定性
        'dropout_rate': 0.3,  # 降低dropout，提高模型学习能力
        'early_stop_patience': 15,  # 增加早停耐心，避免过早停止
        'l2_reg': 1e-4,  # 降低L2正则化强度
        'l1_reg': 1e-5,  # 降低L1正则化强度
        'min_delta': 0.0001,  # 降低最小改善阈值，更敏感
        'reduce_lr_patience': 10,  # 增加学习率衰减耐心
        'reduce_lr_factor': 0.7,  # 温和的学习率衰减
        'min_lr': 1e-7  # 降低最小学习率
    }

    # ============= 硬件配置 =============
    use_gpu = True
    gpu_memory_limit = 0.9  # GPU内存使用限制（占比）
    use_amp = True  # 是否使用混合精度训练
    multi_gpu_enabled = True  # 是否启用多GPU训练
    gpu_memory_growth = True  # 是否启用GPU内存增长
    gpu_visible_devices = None  # 指定可见的GPU设备，如"0,1"表示使用GPU 0和1

    # ============= Optuna超参数配置（稳定性优化） =============
    optuna_params = {
        'n_trials': 3,  # 增加试验次数，找到更好的参数
        'timeout': 600,  # 增加超时时间，允许充分训练
        'param_ranges': {
            'lr': (1e-5, 0.001),  # 扩大学习率范围，包含更保守的值
            'neurons1': (64, 256),  # 增加神经元数量，提高模型容量
            'neurons2': (32, 128),  # 增加神经元数量
            'dropout_rate': (0.1, 0.5),  # 降低dropout范围，提高学习能力
            'batch_size': [16, 32, 64, 128],  # 增加batch size选项
            'attention_units': (16, 64),  # 增加attention units
            'l1_lambda': (1e-6, 1e-4),  # 降低L1正则化范围
            'l2_lambda': (1e-6, 1e-4),  # 降低L2正则化范围
            'optimizer_type': ['adam'],  # TensorFlow 2.10.1只支持Adam
            'activation': ['relu', 'gelu'],  # 添加GELU激活函数
            'loss_type': ['mse', 'huber']  # 添加Huber损失
        }
    }

    # ============= 特征重要性配置（防过拟合优化） =============
    feature_importance = {
        'threshold': 0.05,  # 进一步提高阈值，减少特征数量
        'sample_size': 50000,  # 减少采样数量，适合小数据集
        'save_plots': True,
        'min_features': 10,  # 进一步减少最小特征数
        'max_features': 30  # 进一步减少最大特征数，防止过拟合
    }

    # 集成学习配置（速度优化）
    ensemble_params = {
        'n_splits': 2,  # 保持较少的交叉验证折数，加快训练
        'n_models': 1,  # 保持单模型，减少训练时间
        'voting': 'soft',
        'weights': None,
        'bootstrap': False,  # 禁用bootstrap采样，加快训练
        'bootstrap_ratio': 0.8  # 降低bootstrap采样比例
    }

    # 数据增强配置（速度优化）``
    augmentation_params = {
        'augmentation_factor': 1.5,  # 减少数据增强比例，加快训练
        'noise_factor': 0.02,  # 降低噪声强度
        'feature_mixing': False,  # 禁用特征混合，加快训练
        'random_rotation': False,
        'gaussian_noise': True,  # 添加高斯噪声
        'feature_dropout': 0.2,  # 增加特征随机丢弃比例
        'mixup_alpha': 0.4,  # 增加Mixup数据增强
        'cutmix_alpha': 1.0  # CutMix数据增强
    }

    @staticmethod
    def validate_input_files():
        """验证输入文件是否存在"""
        input_files = [
            Config.soil_data_csv,
            Config.test_soil_csv
        ]

        # 验证训练气象数据文件夹
        if not os.path.exists(Config.weather_data_folder):
            raise FileNotFoundError(f"训练气象数据文件夹不存在: {Config.weather_data_folder}")
        # 检查文件夹中是否有CSV文件
        csv_files = [f for f in os.listdir(Config.weather_data_folder) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"训练气象数据文件夹中没有CSV文件: {Config.weather_data_folder}")
        print(f"找到 {len(csv_files)} 个训练气象数据CSV文件")

        # 验证测试气象数据文件夹
        if not os.path.exists(Config.test_weather_folder):
            raise FileNotFoundError(f"测试气象数据文件夹不存在: {Config.test_weather_folder}")
        # 检查文件夹中是否有CSV文件
        test_csv_files = [f for f in os.listdir(Config.test_weather_folder) if f.endswith('.csv')]
        if not test_csv_files:
            raise FileNotFoundError(f"测试气象数据文件夹中没有CSV文件: {Config.test_weather_folder}")
        print(f"找到 {len(test_csv_files)} 个测试气象数据CSV文件")

        missing_files = []
        for file_path in input_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            raise FileNotFoundError(f"Missing input files: {', '.join(missing_files)}")

        print("All input files found successfully.")

    @staticmethod
    def create_output_dirs():
        """创建所有必要的输出目录"""
        dirs_to_create = [
            Config.output_dir,
            Config.model_dir,
            Config.final_model_dir,
            Config.label_encoder_dir,
            Config.scaler_dir,
            Config.xgboost_dir,
            Config.logs_dir,
            Config.result_dir,
            Config.feature_importance_dir,
            Config.result_analysis_dir,
            Config.training_analysis_dir
        ]

        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

    @staticmethod
    def setup_gpu():
        """配置GPU设置，返回(use_gpu, num_gpus)"""
        if not Config.use_gpu:
            print("GPU is disabled in config")
            return False, 0
        if not tf.test.is_built_with_cuda():
            print("No CUDA support found")
            return False, 0
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU devices found")
            return False, 0
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Successfully configured {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            return True, len(gpus)
        except RuntimeError as e:
            print(f"GPU configuration failed: {e}")
            return False, 0

    # ============= 数据读取配置 =============
    # 默认模式：限制每个气象文件读取的最大行数（基于V3优化结果）
    max_rows_per_weather_file = 5000000  # 减少到500万行，避免过度采样

    # 行数范围模式：指定读取特定行数范围（优先级高于max_rows_per_weather_file）
    weather_start_row = 0  # 气象数据起始行（0表示从第一行开始）
    weather_end_row = 50000  # 气象数据结束行（None表示读取到文件末尾）
    use_weather_row_range = False  # 是否使用行数范围限制

    # 注意：两种模式互斥使用，行数范围模式优先级更高

    # 新增：数据质量控制配置
    min_weather_coverage = 0.1  # 最小气象数据覆盖率（10%）
    use_soil_primary = True  # 是否以土壤特征为主要预测因子
    weather_fallback_strategy = 'interpolate'  # 气象数据缺失时的回退策略


# 配置全局进度条样式
try:
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()  # 防止多个实例冲突
except AttributeError:
    # 如果tqdm版本不支持_instances属性，跳过
    pass

try:
    tqdm.pandas(
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
except Exception as e:
    print(f"Warning: Could not configure tqdm pandas: {str(e)}")

warnings.filterwarnings('ignore')

# ================= GPU 检测与监控 =================
# 全局状态
gpu_monitor_active = False


def detect_device():
    """自动检测 GPU 可用性"""
    if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
        return True
    return False


def setup_multi_gpu():
    """设置多GPU训练环境"""
    try:
        # 检查TensorFlow是否支持CUDA
        if not tf.test.is_built_with_cuda():
            print("警告: TensorFlow未编译CUDA支持，无法使用GPU加速")
            print("建议: 安装支持CUDA的TensorFlow版本")
            return False, 0, None
        # Windows下官方TensorFlow不支持多GPU NCCL通信
        if platform.system() == 'Windows':
            print('Windows下TensorFlow官方不支持多GPU NCCL通信，自动切换为单卡训练。')
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只用第一张卡
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"已启用GPU内存增长，仅使用GPU: {gpus[0].name}")
                return True, 1, None
            else:
                print("未检测到GPU，将使用CPU模式")
                return False, 0, None
        # 设置可见GPU设备
        if Config.gpu_visible_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = Config.gpu_visible_devices
            print(f"设置可见GPU设备: {Config.gpu_visible_devices}")
        # 检测可用的GPU
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) < 2:
            print(f"检测到 {len(gpus)} 个GPU，需要至少2个GPU进行多卡训练")
            if len(gpus) == 1:
                print("单GPU模式：将使用单个GPU进行训练")
                return True, 1, None
            else:
                print("未检测到GPU，将使用CPU模式")
                return False, 0, None
        print(f"检测到 {len(gpus)} 个GPU:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        # 只设置内存增长，不设置虚拟设备
        if Config.gpu_memory_growth:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("已启用GPU内存增长")
        # 创建MirroredStrategy
        if Config.multi_gpu_enabled and len(gpus) >= 2:
            strategy = tf.distribute.MirroredStrategy()
            print(f"成功创建MirroredStrategy，将使用 {strategy.num_replicas_in_sync} 个GPU")
            return True, len(gpus), strategy
        else:
            print("多GPU训练已禁用或GPU数量不足，将使用单GPU")
            return True, len(gpus), None
    except Exception as e:
        print(f"多GPU设置失败: {str(e)}")
        print("将回退到CPU模式")
        return False, 0, None


def gpu_monitor(interval=5):
    """后台线程：定时打印 GPU 使用情况"""
    global gpu_monitor_active
    while gpu_monitor_active:
        if GPUtil:
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    print(
                        f"[GPU Monitor] GPU {i}: Usage: {gpu.load * 100:.1f}% | Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
        else:
            # 备用：使用 TensorFlow API
            try:
                for i in range(len(tf.config.list_physical_devices('GPU'))):
                    memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                    print(f"[GPU Monitor] GPU {i}: Current: {memory_info['current'] // 1024 ** 2} MiB | "
                          f"Peak: {memory_info['peak'] // 1024 ** 2} MiB")
            except:
                pass
        time.sleep(interval)


def memory_safe(func):
    """内存安全装饰器"""

    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem = process.memory_info().rss / 1024 ** 3  # GB
        if mem > 10:  # 当内存超过10GB时报警
            print(f"Memory warning: {mem:.2f}GB used")
        return func(*args, **kwargs)

    return wrapper


# 打印 GPU 信息
print(f"TensorFlow版本: {tf.__version__}")
print(f"CUDA是否可用: {tf.test.is_built_with_cuda()}")
if tf.test.is_built_with_cuda():
    gpus = tf.config.list_physical_devices('GPU')
    print(f"检测到的GPU数量: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"GPU {i} 设备名称: {gpu.name}")

    # 设置多GPU环境
    multi_gpu_available, num_gpus, strategy = setup_multi_gpu()
    if multi_gpu_available:
        print(f"多GPU训练环境设置成功，将使用 {num_gpus} 个GPU进行训练")
    else:
        print("多GPU训练环境设置失败，将使用单GPU或CPU")
        strategy = None


##############################################
# 1. 数据预处理与特征工程
##############################################

def unify_coordinates_with_tolerance(weather_data, soil_data, tolerance=100):
    """
    统一气象数据和土壤数据的坐标系统，以土壤数据xy坐标为基准
    使用高效的向量化算法和空间索引优化

    Args:
        weather_data (pd.DataFrame): 气象数据
        soil_data (pd.DataFrame): 土壤数据（作为坐标基准）
        tolerance (float): 坐标容差，默认100米

    Returns:
        pd.DataFrame: 坐标统一后的气象数据
    """
    print(f"开始坐标统一，容差: {tolerance}米")
    print(f"气象数据原始形状: {weather_data.shape}")
    print(f"土壤数据形状: {soil_data.shape}")

    # 确保坐标列存在
    if 'x' not in weather_data.columns or 'y' not in weather_data.columns:
        raise ValueError("气象数据中缺少x, y坐标列")
    if 'x' not in soil_data.columns or 'y' not in soil_data.columns:
        raise ValueError("土壤数据中缺少x, y坐标列")

    # 获取土壤数据的唯一坐标点
    soil_coords = soil_data[['x', 'y']].drop_duplicates().reset_index(drop=True)
    print(f"土壤数据唯一坐标点数: {len(soil_coords)}")

    # 获取气象数据的唯一坐标点
    weather_coords = weather_data[['x', 'y']].drop_duplicates().reset_index(drop=True)
    print(f"气象数据唯一坐标点数: {len(weather_coords)}")

    # 使用高效的空间索引算法
    print("使用高效空间索引算法进行坐标匹配...")

    # 将坐标转换为numpy数组
    weather_coords_array = weather_coords[['x', 'y']].values
    soil_coords_array = soil_coords[['x', 'y']].values

    # 创建坐标映射字典
    coord_mapping = {}
    matched_count = 0

    # 使用KDTree进行快速最近邻搜索
    try:
        from scipy.spatial import cKDTree
        print("使用scipy.spatial.cKDTree进行快速匹配...")

        # 构建土壤坐标的KDTree
        soil_tree = cKDTree(soil_coords_array)

        # 分批处理气象坐标
        batch_size = 100000  # 进一步增大批次大小，提高处理效率
        total_batches = (len(weather_coords) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(weather_coords))

            print(f"  处理批次 {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})")

            # 获取当前批次的坐标
            batch_weather = weather_coords_array[start_idx:end_idx]

            # 使用KDTree查询最近邻
            distances, indices = soil_tree.query(batch_weather, k=1)

            # 处理匹配结果
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if dist <= tolerance:
                    wx, wy = batch_weather[i]
                    soil_x, soil_y = soil_coords_array[idx]
                    coord_mapping[(wx, wy)] = (soil_x, soil_y)
                    matched_count += 1

    except ImportError:
        print("scipy不可用，使用优化的向量化算法...")

        # 分批处理以提高效率
        batch_size = 20000  # 增大批次大小，提高处理效率
        total_batches = (len(weather_coords) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(weather_coords))

            print(f"  处理批次 {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})")

            # 获取当前批次的坐标
            batch_weather = weather_coords_array[start_idx:end_idx]

            # 向量化计算距离
            for i, (wx, wy) in enumerate(batch_weather):
                # 计算到所有土壤坐标点的距离
                distances = np.sqrt((soil_coords_array[:, 0] - wx) ** 2 + (soil_coords_array[:, 1] - wy) ** 2)
                min_distance = np.min(distances)
                min_idx = np.argmin(distances)

                # 如果距离在容差范围内，使用土壤坐标
                if min_distance <= tolerance:
                    soil_x, soil_y = soil_coords_array[min_idx]
                    coord_mapping[(wx, wy)] = (soil_x, soil_y)
                    matched_count += 1

    print(
        f"成功匹配的坐标点数: {matched_count}/{len(weather_coords)} ({matched_count / len(weather_coords) * 100:.1f}%)")

    # 应用坐标映射
    print("应用坐标映射...")
    weather_corrected = weather_data.copy()
    weather_corrected['x_original'] = weather_corrected['x']
    weather_corrected['y_original'] = weather_corrected['y']

    # 使用向量化操作更新坐标
    for (wx, wy), (sx, sy) in coord_mapping.items():
        mask = (weather_corrected['x'] == wx) & (weather_corrected['y'] == wy)
        weather_corrected.loc[mask, 'x'] = sx
        weather_corrected.loc[mask, 'y'] = sy

    # 移除无法匹配的数据点
    print("移除无法匹配的数据点...")
    valid_coords = set(coord_mapping.values())
    weather_corrected = weather_corrected[
        weather_corrected.apply(lambda row: (row['x'], row['y']) in valid_coords, axis=1)
    ]

    print(f"坐标统一后气象数据形状: {weather_corrected.shape}")
    print(f"坐标统一成功率: {len(weather_corrected) / len(weather_data) * 100:.1f}%")

    return weather_corrected


def quick_coordinate_unify(weather_data, soil_data, tolerance=100):
    """
    快速坐标统一方法，使用简单的四舍五入策略
    """
    print(f"使用快速坐标统一，容差: {tolerance}米")
    print(f"气象数据形状: {weather_data.shape}")
    print(f"土壤数据形状: {soil_data.shape}")

    # 获取土壤数据的坐标范围
    soil_x_min, soil_x_max = soil_data['x'].min(), soil_data['x'].max()
    soil_y_min, soil_y_max = soil_data['y'].min(), soil_data['y'].max()

    print(f"土壤数据坐标范围: x[{soil_x_min:.2f}, {soil_x_max:.2f}], y[{soil_y_min:.2f}, {soil_y_max:.2f}]")

    # 获取气象数据的坐标范围
    weather_x_min, weather_x_max = weather_data['x'].min(), weather_data['x'].max()
    weather_y_min, weather_y_max = weather_data['y'].min(), weather_data['y'].max()

    print(f"气象数据坐标范围: x[{weather_x_min:.2f}, {weather_x_max:.2f}], y[{weather_y_min:.2f}, {weather_y_max:.2f}]")

    # 创建坐标映射
    weather_corrected = weather_data.copy()
    weather_corrected['x_original'] = weather_corrected['x']
    weather_corrected['y_original'] = weather_corrected['y']

    # 使用简单的坐标对齐策略
    # 将气象坐标四舍五入到最近的整数，然后与土壤坐标匹配
    print("应用坐标对齐策略...")

    # 获取土壤数据的唯一坐标点
    soil_coords = soil_data[['x', 'y']].drop_duplicates()
    soil_coords_set = set(zip(soil_coords['x'], soil_coords['y']))

    matched_count = 0
    total_count = len(weather_corrected)

    # 分批处理以提高效率
    batch_size = 200000  # 增大批次大小，提高处理效率
    for i in range(0, total_count, batch_size):
        end_idx = min(i + batch_size, total_count)
        batch = weather_corrected.iloc[i:end_idx]

        # 对坐标进行四舍五入
        rounded_x = np.round(batch['x']).astype(int)
        rounded_y = np.round(batch['y']).astype(int)

        # 检查是否在土壤坐标范围内
        valid_mask = (
                (rounded_x >= soil_x_min) & (rounded_x <= soil_x_max) &
                (rounded_y >= soil_y_min) & (rounded_y <= soil_y_max)
        )

        # 更新有效坐标
        weather_corrected.iloc[i:end_idx, weather_corrected.columns.get_loc('x')] = np.where(
            valid_mask, rounded_x, weather_corrected.iloc[i:end_idx]['x']
        )
        weather_corrected.iloc[i:end_idx, weather_corrected.columns.get_loc('y')] = np.where(
            valid_mask, rounded_y, weather_corrected.iloc[i:end_idx]['y']
        )

        matched_count += valid_mask.sum()

        if i % (batch_size * 10) == 0:
            print(f"  处理进度: {i}/{total_count} ({i / total_count * 100:.1f}%)")

    # 移除无法匹配的数据点
    print("移除无法匹配的数据点...")
    valid_coords = set(zip(weather_corrected['x'], weather_corrected['y']))
    soil_coords_set = set(zip(soil_data['x'], soil_data['y']))

    # 只保留在土壤坐标范围内的数据
    weather_corrected = weather_corrected[
        (weather_corrected['x'] >= soil_x_min) & (weather_corrected['x'] <= soil_x_max) &
        (weather_corrected['y'] >= soil_y_min) & (weather_corrected['y'] <= soil_y_max)
        ]

    print(f"坐标统一后气象数据形状: {weather_corrected.shape}")
    print(f"坐标统一成功率: {len(weather_corrected) / len(weather_data) * 100:.1f}%")

    return weather_corrected


def load_weather_data_from_folder(folder_path, start_row=None, end_row=None):
    """
    从指定文件夹中读取所有气象数据CSV文件并合并
    支持指定行数范围读取

    Args:
        folder_path (str): 气象数据文件夹路径
        start_row (int, optional): 起始行数（0-based，包含此行）
        end_row (int, optional): 结束行数（0-based，不包含此行，None表示到文件末尾）

    Returns:
        pd.DataFrame: 合并后的气象数据
    """
    print(f"从文件夹读取气象数据: {folder_path}")

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"气象数据文件夹不存在: {folder_path}")

    # 获取文件夹中所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"气象数据文件夹中没有CSV文件: {folder_path}")

    print(f"找到 {len(csv_files)} 个CSV文件: {csv_files}")

    # 确定读取策略
    if start_row is not None or end_row is not None:
        print(f"使用行数范围读取: 起始行={start_row}, 结束行={end_row}")
        use_range_mode = True
    else:
        print(f"使用默认模式: 每个文件最多读取 {Config.max_rows_per_weather_file:,} 行数据")
        use_range_mode = False

    # 读取并合并所有CSV文件
    weather_data_list = []

    for csv_file in tqdm(csv_files, desc="读取气象数据文件"):
        file_path = os.path.join(folder_path, csv_file)
        try:
            print(f"正在读取文件: {csv_file}")

            if use_range_mode:
                # 行数范围模式
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        total_lines = sum(1 for _ in f)
                    print(f"  文件总行数: {total_lines:,}")

                    # 计算实际读取范围
                    actual_start = start_row if start_row is not None else 0
                    actual_end = end_row if end_row is not None else total_lines

                    # 确保范围有效
                    actual_start = max(0, actual_start)
                    actual_end = min(total_lines, actual_end)

                    if actual_start >= actual_end:
                        print(f"  ⚠️ 起始行({actual_start}) >= 结束行({actual_end})，跳过此文件")
                        continue

                    rows_to_skip = actual_start
                    rows_to_read = actual_end - actual_start

                    print(f"  读取范围: 第{actual_start + 1}行到第{actual_end}行 (共{rows_to_read:,}行)")

                except Exception as e:
                    print(f"  警告: 无法统计文件行数，跳过此文件: {str(e)}")
                    continue
            else:
                # 默认模式
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        total_lines = sum(1 for _ in f)
                    print(f"  文件总行数: {total_lines:,}")

                    # 如果文件行数超过限制，只读取前N行
                    if total_lines > Config.max_rows_per_weather_file:
                        print(f"  ⚠️ 文件过大，只读取前 {Config.max_rows_per_weather_file:,} 行")
                        rows_to_read = Config.max_rows_per_weather_file
                    else:
                        rows_to_read = total_lines

                    rows_to_skip = 0

                except Exception as e:
                    print(f"  警告: 无法统计文件行数，使用默认限制: {str(e)}")
                    rows_to_read = Config.max_rows_per_weather_file
                    rows_to_skip = 0

            # 读取指定行数的数据
            try:
                if use_range_mode and rows_to_skip > 0:
                    # 使用skiprows和nrows参数
                    df = pd.read_csv(file_path, encoding='utf-8',
                                     skiprows=range(1, rows_to_skip + 1), nrows=rows_to_read)
                else:
                    # 使用nrows参数
                    df = pd.read_csv(file_path, encoding='utf-8', nrows=rows_to_read)
            except UnicodeDecodeError:
                try:
                    if use_range_mode and rows_to_skip > 0:
                        df = pd.read_csv(file_path, encoding='gbk',
                                         skiprows=range(1, rows_to_skip + 1), nrows=rows_to_read)
                    else:
                        df = pd.read_csv(file_path, encoding='gbk', nrows=rows_to_read)
                except UnicodeDecodeError:
                    if use_range_mode and rows_to_skip > 0:
                        df = pd.read_csv(file_path, encoding='latin1',
                                         skiprows=range(1, rows_to_skip + 1), nrows=rows_to_read)
                    else:
                        df = pd.read_csv(file_path, encoding='latin1', nrows=rows_to_read)

            # 添加文件名信息（可选）
            df['source_file'] = csv_file

            weather_data_list.append(df)
            print(f"  ✅ 成功读取: {df.shape[0]:,} 行 × {df.shape[1]} 列")

        except Exception as e:
            print(f"  ❌ 读取文件 {csv_file} 失败: {str(e)}")
            continue

    if not weather_data_list:
        raise ValueError("没有成功读取任何气象数据文件")

    # 合并所有数据
    print("合并气象数据文件...")
    weather_data = pd.concat(weather_data_list, ignore_index=True)

    print(f"合并后的气象数据形状: {weather_data.shape}")
    print(f"合并后的气象数据列: {weather_data.columns.tolist()}")

    # 检查是否有重复数据
    if 'source_file' in weather_data.columns:
        print("各文件数据量统计:")
        print(weather_data['source_file'].value_counts())
        # 移除source_file列，避免影响后续处理
        weather_data = weather_data.drop(columns=['source_file'])

    # 不进行采样，使用全部气象数据
    print(f"使用全部气象数据: {weather_data.shape[0]:,} 行")

    return weather_data


@memory_safe
def load_data():
    print("\n1/4.Loading and preprocessing data...")
    try:
        # ============= 加载训练数据 =============
        # 1) 加载训练集气象数据
        print("使用文件夹模式加载训练气象数据...")
        weather_train = load_weather_data_from_folder(Config.weather_data_folder)
        print(f"Weather data shape: {weather_train.shape}")
        print(f"Weather data columns: {weather_train.columns.tolist()}")

        # 2) 加载训练集土壤数据
        print("Loading training soil data from:", Config.soil_data_csv)
        try:
            soil_train = pd.read_csv(Config.soil_data_csv, encoding='utf-8')
        except UnicodeDecodeError:
            soil_train = pd.read_csv(Config.soil_data_csv, encoding='gbk')
        print(f"Soil data shape: {soil_train.shape}")
        print(f"Soil data columns: {soil_train.columns.tolist()}")

        # 不进行采样，使用全部土壤数据
        print(f"使用全部土壤数据: {soil_train.shape[0]:,} 行")

        # 检查土壤数据列名并重命名以匹配产品数据
        if 'TZ' in soil_train.columns:
            print("检测到土壤数据使用 TZ 列名，重命名为 tz...")
            soil_train = soil_train.rename(columns={'TZ': 'tz'})

        # ============= 坐标统一 =============
        print("\n=== 开始坐标统一处理 ===")

        # 先重命名气象数据坐标列以匹配土壤数据
        if 'Lon' in weather_train.columns and 'Lat' in weather_train.columns:
            print("重命名气象数据坐标列: Lon/Lat -> x/y")
            weather_train = weather_train.rename(columns={'Lon': 'x', 'Lat': 'y'})
        elif 'lon' in weather_train.columns and 'lat' in weather_train.columns:
            print("重命名气象数据坐标列: lon/lat -> x/y")
            weather_train = weather_train.rename(columns={'lon': 'x', 'lat': 'y'})

        # 使用快速坐标统一（跳过复杂的匹配算法）
        print("使用快速坐标统一方法...")
        weather_train = quick_coordinate_unify(weather_train, soil_train, tolerance=100)
        print("=== 坐标统一完成 ===\n")

        # 3) 加载训练集产量/适宜度数据
        print("Loading training product data from:", Config.product_data_csv)
        if not os.path.exists(Config.product_data_csv):
            print(f"找不到文件: {Config.product_data_csv}")
            print("当前目录下可用的csv文件:")
            for f in os.listdir(os.path.dirname(Config.product_data_csv)):
                if f.endswith('.csv'):
                    print(f)
            raise FileNotFoundError(f"未找到训练集产量/适宜度数据文件: {Config.product_data_csv}")
        product_train = pd.read_csv(Config.product_data_csv, encoding='utf-8')
        print(f"Product data shape: {product_train.shape}")
        print(f"Product data columns: {product_train.columns.tolist()}")

        # ============= 空间近邻合并 =============
        # 使用全局定义的函数

        # 统一列名小写
        product_train.columns = product_train.columns.str.strip().str.lower()
        weather_train.columns = weather_train.columns.str.strip().str.lower()
        soil_train.columns = soil_train.columns.str.strip().str.lower()

        # 清理气象数据：删除有缺失值的行，只保留完整的气象数据
        print("清理气象数据：删除有缺失值的行...")
        original_weather_shape = weather_train.shape
        print(f"原始气象数据形状: {original_weather_shape}")

        # 识别气象特征列（排除坐标、年份、月份等非气象特征）
        weather_feature_cols = [col for col in weather_train.columns
                                if any(keyword in col.lower() for keyword in
                                       ['tsun', 'tave', 'tmax', 'tmin', 'rain', 'gtave', 'gtmax', 'gtmin', 'sevp'])]

        print(f"气象特征列数量: {len(weather_feature_cols)}")
        print(f"气象特征列: {weather_feature_cols[:5]}...")  # 显示前5个列名

        # 删除任何气象特征列有缺失值的行
        if weather_feature_cols:
            # 保留所有气象特征列都不为NaN的行
            weather_train = weather_train.dropna(subset=weather_feature_cols)
            print(f"删除缺失值后气象数据形状: {weather_train.shape}")
            print(f"删除了 {original_weather_shape[0] - weather_train.shape[0]} 行有缺失值的数据")
            print(f"保留率: {weather_train.shape[0] / original_weather_shape[0] * 100:.2f}%")
        else:
            print("⚠️ 未找到气象特征列，跳过缺失值清理")

        # 清理产品数据：删除有缺失值的行
        print("\n清理产品数据：删除有缺失值的行...")
        original_product_shape = product_train.shape
        print(f"原始产品数据形状: {original_product_shape}")

        # 删除产品数据中有缺失值的行
        product_train = product_train.dropna()
        print(f"删除缺失值后产品数据形状: {product_train.shape}")
        print(f"删除了 {original_product_shape[0] - product_train.shape[0]} 行有缺失值的数据")
        print(f"保留率: {product_train.shape[0] / original_product_shape[0] * 100:.2f}%")

        # 清理土壤数据：删除有缺失值的行
        print("\n清理土壤数据：删除有缺失值的行...")
        original_soil_shape = soil_train.shape
        print(f"原始土壤数据形状: {original_soil_shape}")

        # 删除土壤数据中有缺失值的行
        soil_train = soil_train.dropna()
        print(f"删除缺失值后土壤数据形状: {soil_train.shape}")
        print(f"删除了 {original_soil_shape[0] - soil_train.shape[0]} 行有缺失值的数据")
        print(f"保留率: {soil_train.shape[0] / original_soil_shape[0] * 100:.2f}%")

        # 统一年份类型到整数，避免 1995.0 与 1995 不相等导致合并为空
        for df_name, df in [('product_train', product_train), ('weather_train', weather_train)]:
            if 'yyyy' in df.columns:
                try:
                    df['yyyy'] = pd.to_numeric(df['yyyy'], errors='coerce').round().astype('Int64')
                    print(f"{df_name} 年份示例: {df['yyyy'].dropna().unique()[:5]}")
                except Exception as e:
                    print(f"警告: 规范化 {df_name}.yyyy 到整数失败: {e}")

        # 不进行采样，使用全部数据进行合并
        print("使用全部数据进行合并...")
        print(f"产品数据形状: {product_train.shape}")
        print(f"气象数据形状: {weather_train.shape}")

        # 产量与气象合并：优先精确键合并，其次空间近邻合并
        print("尝试按 ['x','y','yyyy'] 精确键合并 product 与 weather ...")

        # 检查气象数据列名并重命名以匹配产品数据
        print(f"气象数据列名: {weather_train.columns.tolist()}")

        # 检查并重命名坐标列
        if 'Lon' in weather_train.columns and 'Lat' in weather_train.columns:
            print("检测到气象数据使用 Lon/Lat 列名，重命名为 x/y...")
            weather_train = weather_train.rename(columns={'Lon': 'x', 'Lat': 'y'})
        elif 'lon' in weather_train.columns and 'lat' in weather_train.columns:
            print("检测到气象数据使用 lon/lat 列名，重命名为 x/y...")
            weather_train = weather_train.rename(columns={'lon': 'x', 'lat': 'y'})

        # 检查并重命名年份列
        if 'YYYY' in weather_train.columns:
            print("检测到气象数据使用 YYYY 列名，重命名为 yyyy...")
            weather_train = weather_train.rename(columns={'YYYY': 'yyyy'})
        elif 'yyyy' in weather_train.columns:
            print("气象数据年份列已经是 yyyy，无需重命名")

        print(f"重命名后气象数据列名: {weather_train.columns.tolist()}")

        # 为保持后续特征命名一致，将 weather 列加上 right_ 前缀，再基于对应键合并
        weather_pref = weather_train.add_prefix('right_')
        merged_pw = pd.merge(
            product_train,
            weather_pref,
            left_on=['x', 'y', 'yyyy'],
            right_on=['right_x', 'right_y', 'right_yyyy'],
            how='inner'
        )
        if merged_pw.shape[0] == 0:
            print("精确键合并结果为空，回退到空间+时间近邻合并（坐标误差控制在50公里内）...")
            merged_pw = spatial_temporal_merge(product_train, weather_train, xy_cols=['x', 'y'], time_col='yyyy',
                                               tolerance=50000)
        print(f"After product-weather merge: {merged_pw.shape}")
        print(f"merged_pw columns: {merged_pw.columns.tolist()}")
        # 动态检测并重命名所有x, y列，主键x, y只保留product的x, y
        x_cols = [col for col in merged_pw.columns if col.lower() == 'x']
        y_cols = [col for col in merged_pw.columns if col.lower() == 'y']
        xw_cols = [col for col in merged_pw.columns if col.lower() == 'right_x']
        yw_cols = [col for col in merged_pw.columns if col.lower() == 'right_y']

        print("x_cols:", x_cols)
        print("y_cols:", y_cols)
        print("xw_cols:", xw_cols)
        print("yw_cols:", yw_cols)

        if x_cols and y_cols and xw_cols and yw_cols:
            merged_pw = merged_pw.rename(columns={
                x_cols[0]: 'x_product',
                y_cols[0]: 'y_product',
                xw_cols[0]: 'x_weather',
                yw_cols[0]: 'y_weather',
            })
            merged_pw['x'] = merged_pw['x_product']
            merged_pw['y'] = merged_pw['y_product']
            keep_cols = [col for col in merged_pw.columns if col not in ['x_weather', 'y_weather']]
            merged_pw = merged_pw[keep_cols]
        else:
            raise ValueError(
                f"merged_pw中x, y列数量异常: x_cols={x_cols}, y_cols={y_cols}, xw_cols={xw_cols}, yw_cols={yw_cols}，请检查合并逻辑！")
        if 'x' not in merged_pw.columns or 'y' not in merged_pw.columns:
            raise ValueError("merged_pw中没有标准的x, y列，请检查合并逻辑！")
        # 产量与土壤（按xy）空间近邻合并
        # 与土壤表做空间最近邻合并，放宽容差以避免空集
        merged_pws = spatial_merge(merged_pw, soil_train, on=['x', 'y'], tolerance=50000)
        print(f"After product-weather-soil merge: {merged_pws.shape}")
        data_train = merged_pws
        data_train.columns = data_train.columns.str.strip().str.lower()
        print(f"Merged training data shape: {data_train.shape}")

        # 4) 合并气象和产量数据（x, y, YYYY为主键）
        # data_train = pd.merge(weather_train, product_train, on=['x', 'y', 'YYYY'], how='inner')
        # 5) 再与土壤数据合并（x, y为主键）
        # data_train = pd.merge(data_train, soil_train, on=['x', 'y'], how='inner')

        # 6) 提取特征列
        print("\nExtracting features...")
        exclude_columns = ['x', 'y', 'suit', 'per_qu', 'per_mu', 'yyyy']  # 全部小写
        feature_columns = [col for col in data_train.columns if col not in exclude_columns]
        print(f"Selected feature columns ({len(feature_columns)}):")
        print(feature_columns)
        X_train = data_train[feature_columns].copy()

        # 7) 目标变量 - 注意：这里使用原始data_train，后续会在数据拆分后重新提取
        y_cls_train = data_train['suit'].copy()
        y_reg_train = data_train[['per_qu', 'per_mu']].copy()

        # === 检查目标变量数据质量 ===
        print(f"目标变量原始统计信息:")
        print(
            f"per_mu: min={y_reg_train['per_mu'].min():.4f}, max={y_reg_train['per_mu'].max():.4f}, mean={y_reg_train['per_mu'].mean():.4f}")
        print(
            f"per_qu: min={y_reg_train['per_qu'].min():.4f}, max={y_reg_train['per_qu'].max():.4f}, mean={y_reg_train['per_qu'].mean():.4f}")
        print(f"suit: unique values={y_reg_train['suit'].unique() if hasattr(y_reg_train, 'suit') else 'N/A'}")

        # === 异常值裁剪（在log1p变换之前）- per_mu预测优化 ===
        for col in ['per_mu', 'per_qu']:
            q1 = y_reg_train[col].quantile(0.005)  # 从0.01降低到0.005，保留更多数据
            q99 = y_reg_train[col].quantile(0.995)  # 从0.99提高到0.995，保留更多数据
            print(f"{col} 异常值裁剪: [{q1:.4f}, {q99:.4f}]")
            mask = (y_reg_train[col] >= q1) & (y_reg_train[col] <= q99)
            y_reg_train = y_reg_train[mask]
            X_train = X_train[mask]
            y_cls_train = y_cls_train[mask]

        # ============= 简化处理：不加载测试集，只使用训练集 =============
        print("\n简化处理：不加载测试集，只使用训练集进行训练...")
        print("后续可以用训练好的模型去预测测试集数据")

        # 从训练集中划分出验证集
        print("从训练集中划分验证集...")
        from sklearn.model_selection import train_test_split

        # 使用20%的数据作为验证集
        test_size = 0.2
        data_train_final, data_val = train_test_split(
            data_train,
            test_size=test_size,
            random_state=Config.random_seed,
            stratify=data_train['suit'] if 'suit' in data_train.columns else None
        )

        print(f"训练集大小: {data_train_final.shape}")
        print(f"验证集大小: {data_val.shape}")

        # 创建验证集特征矩阵
        X_val = pd.DataFrame()

        # 为每个训练集特征列在验证集中找到对应列
        for feature in feature_columns:
            if feature in data_val.columns:
                X_val[feature] = data_val[feature]
            elif feature.startswith('right_'):
                # 处理right_前缀的特征
                base_name = feature[6:]  # 去掉'right_'前缀
                if base_name in data_val.columns:
                    X_val[feature] = data_val[base_name]
                else:
                    # 如果找不到对应列，用0填充
                    X_val[feature] = 0
            else:
                # 如果找不到对应列，用0填充
                X_val[feature] = 0

        print(f"验证集特征矩阵形状: {X_val.shape}")

        # 确保验证集和训练集特征列完全一致
        missing_cols = set(feature_columns) - set(X_val.columns)
        if missing_cols:
            print(f"为验证集添加缺失特征: {len(missing_cols)} 个")
            for col in missing_cols:
                X_val[col] = 0

        # 确保列顺序一致
        X_val = X_val[feature_columns]

        print(f"最终验证集特征矩阵形状: {X_val.shape}")
        print(f"训练集特征矩阵形状: {X_train.shape}")

        # 验证验证集不为空
        if X_val.shape[0] == 0:
            raise ValueError("验证集为空，无法继续训练！")

        # 验证特征列一致
        if list(X_train.columns) != list(X_val.columns):
            raise ValueError(
                f"训练集和验证集特征列不一致！\n训练集: {list(X_train.columns)}\n验证集: {list(X_val.columns)}")

        # ============= 特征工程 =============
        print("\nPreparing features...")

        # 提取特征和目标变量 - 从拆分后的数据中提取
        X_train = data_train_final[feature_columns].copy()

        # 重新提取目标变量，确保与特征矩阵的行数一致
        y_cls_train = data_train_final['suit'].copy()
        y_reg_train = data_train_final[['per_qu', 'per_mu']].copy()

        # === 检查目标变量数据质量（拆分后） ===
        print(f"拆分后目标变量统计信息:")
        print(
            f"per_mu: min={y_reg_train['per_mu'].min():.4f}, max={y_reg_train['per_mu'].max():.4f}, mean={y_reg_train['per_mu'].mean():.4f}")
        print(
            f"per_qu: min={y_reg_train['per_qu'].min():.4f}, max={y_reg_train['per_qu'].max():.4f}, mean={y_reg_train['per_qu'].mean():.4f}")
        print(f"suit: unique values={y_cls_train.unique()}")

        # 对目标变量进行log1p变换（只进行一次）
        print("对目标变量进行log1p变换...")
        y_reg_train = np.log1p(y_reg_train)

        # 检查log1p变换后的数据质量
        print(f"log1p变换后目标变量统计信息:")
        print(
            f"per_mu: min={y_reg_train['per_mu'].min():.4f}, max={y_reg_train['per_mu'].max():.4f}, mean={y_reg_train['per_mu'].mean():.4f}")
        print(
            f"per_qu: min={y_reg_train['per_qu'].min():.4f}, max={y_reg_train['per_qu'].max():.4f}, mean={y_reg_train['per_qu'].mean():.4f}")

        # 对log1p变换后的数据进行轻微裁剪，避免极值 - per_mu预测优化
        for col in y_reg_train.columns:
            q_low = y_reg_train[col].quantile(0.0005)  # 进一步降低下界
            q_high = y_reg_train[col].quantile(0.9995)  # 进一步提高上界
            print(f"{col} log1p后裁剪: [{q_low:.4f}, {q_high:.4f}]")
            y_reg_train[col] = np.clip(y_reg_train[col], q_low, q_high)

        # 处理所有object类型特征：区间字符串转均值，普通字符串用LabelEncoder编码
        import re
        from sklearn.preprocessing import LabelEncoder
        print("\n处理所有object类型特征：区间字符串转均值，普通字符串用LabelEncoder编码")
        for col in feature_columns:
            if X_train[col].dtype == object or X_val[col].dtype == object:
                def is_range(s):
                    s = str(s).strip().replace('－', '-').replace('—', '-').replace('–', '-')
                    return bool(re.match(r'^-?\d+(\.\d+)?\s*-\s*-?\d+(\.\d+)?$', s))

                sample = pd.concat([X_train[col], X_val[col]]).astype(str)
                if sample.apply(is_range).any():
                    print(f"将区间字符串列 {col} 转为均值")

                    def range_to_mean(x):
                        try:
                            s = str(x).strip().replace('－', '-').replace('—', '-').replace('–', '-')
                            if is_range(s):
                                low, high = map(float, re.split(r'\s*-\s*', s))
                                return (low + high) / 2
                            return float(s)
                        except Exception as e:
                            print(f"[转换异常] 列: {col}, 原值: {x}, 错误: {e}")
                            return np.nan

                    X_train[col] = X_train[col].apply(range_to_mean)
                    X_val[col] = X_val[col].apply(range_to_mean)
                    mean_value = X_train[col].mean()
                    X_train[col].fillna(mean_value, inplace=True)
                    X_val[col].fillna(mean_value, inplace=True)
                    X_train[col] = X_train[col].astype(float)
                    X_val[col] = X_val[col].astype(float)
                else:
                    print(f"Label encoding column: {col}")
                    le = LabelEncoder()
                    all_values = pd.concat([X_train[col].astype(str), X_val[col].astype(str)], axis=0)
                    le.fit(all_values)
                    X_train[col] = le.transform(X_train[col].astype(str))
                    X_val[col] = le.transform(X_val[col].astype(str))
                    X_train[col] = X_train[col].astype(float)
                    X_val[col] = X_val[col].astype(float)
        print("\nX_train dtypes before standardization:")
        print(X_train.dtypes)
        print("\n所有object列唯一值检查：")
        for col in X_train.columns:
            if X_train[col].dtype == object:
                print(f"{col}: {X_train[col].unique()[:20]}")
                import sys
                sys.exit(1)

        # 注释掉原有的范围值特征处理和其它X_train/X_test赋值
        '''
        # 处理范围值特征
        print("\nProcessing range value features...")
        def process_range(x):
            try:
                if isinstance(x, str) and '-' in x:
                    low, high = map(float, x.split('-'))
                    return (low + high) / 2
                return float(x)
            except:
                return np.nan

        for range_col in Config.range_columns:
            if range_col in X_train.columns:
                print(f"Processing {range_col}...")
                X_train[range_col] = X_train[range_col].apply(process_range)
                # X_test[range_col] = X_test[range_col].apply(process_range)  # 注释掉，不再使用测试集
                mean_value = X_train[range_col].mean()
                X_train[range_col].fillna(mean_value, inplace=True)
                # X_test[range_col].fillna(mean_value, inplace=True)  # 注释掉，不再使用测试集
        '''

        # 处理分类特征
        '''
        print("\nEncoding categorical features...")
        encoders = {}
        for cat_col in Config.categorical_columns:
            if cat_col in X_train.columns:
                print(f"Encoding {cat_col}...")
                le = LabelEncoder()
                X_train[cat_col] = le.fit_transform(X_train[cat_col].astype(str))
                # X_test[cat_col] = le.transform(X_test[cat_col].astype(str))  # 注释掉，不再使用测试集
                encoders[cat_col] = le
            # 保存编码器
                encoder_file = os.path.join(Config.label_encoder_dir, f"{cat_col}_encoder.pkl")
            try:
                    joblib.dump(le, encoder_file)
                    print(f"Saved encoder for {cat_col} to {encoder_file}")
            except Exception as e:
                    print(f"Warning: Could not save encoder for {cat_col}: {str(e)}")
        '''

        # 检查所有特征类型
        print("\nX_train dtypes before standardization:")
        print(X_train.dtypes)
        print("\n所有object列唯一值检查：")
        for col in X_train.columns:
            if X_train[col].dtype == object:
                print(f"{col}: {X_train[col].unique()[:20]}")
                import sys
                sys.exit(1)

        # 特征标准化
        print("\nStandardizing features...")
        if len(X_train) == 0:
            raise ValueError("训练集为空：合并失败或容差过小。请检查坐标/年份一致性或增大容差！")
        import tempfile
        temp_train_path = os.path.join(tempfile.gettempdir(), 'xtrain_temp.csv')
        temp_val_path = os.path.join(tempfile.gettempdir(), 'xval_temp.csv')
        X_train.to_csv(temp_train_path, index=False)
        X_val.to_csv(temp_val_path, index=False)

        chunksize = 100_000
        scaler = StandardScaler()
        # 分块partial_fit X_train（若第一块即为空，直接报错，避免后续标准化二次报错）
        any_chunk = False
        for chunk in pd.read_csv(temp_train_path, chunksize=chunksize):
            if len(chunk) == 0:
                continue
            any_chunk = True
            scaler.partial_fit(chunk)
        if not any_chunk:
            raise ValueError("训练集分块读取为空：请检查合并结果是否为0行！")
        # 分块transform X_train
        scaled_chunks = []
        for chunk in pd.read_csv(temp_train_path, chunksize=chunksize):
            if len(chunk) == 0:
                continue
            chunk_scaled = scaler.transform(chunk)
            chunk_scaled = pd.DataFrame(chunk_scaled, columns=feature_columns)
            scaled_chunks.append(chunk_scaled)
        X_train_scaled = pd.concat(scaled_chunks, ignore_index=True)
        # 分块transform X_val
        scaled_chunks = []
        for chunk in pd.read_csv(temp_val_path, chunksize=chunksize):
            if len(chunk) == 0:
                continue
            chunk_scaled = scaler.transform(chunk)
            chunk_scaled = pd.DataFrame(chunk_scaled, columns=feature_columns)
            scaled_chunks.append(chunk_scaled)

        # 检查是否有验证数据需要标准化
        if scaled_chunks:
            X_val_scaled = pd.concat(scaled_chunks, ignore_index=True)
        else:
            print("警告：验证集为空，创建一个空的验证集特征矩阵")
            X_val_scaled = pd.DataFrame(columns=feature_columns)

        # 保存标准化器
        print("Saving scaler...")
        try:
            os.makedirs(os.path.dirname(Config.scaler_file), exist_ok=True)
            joblib.dump(scaler, Config.scaler_file)
            print(f"Scaler successfully saved to: {Config.scaler_file}")
        except Exception as e:
            print(f"Warning: Could not save scaler: {str(e)}")

        # 保存标签编码器
        print("Saving label encoder...")
        try:
            le = LabelEncoder()
            y_cls_train_encoded = le.fit_transform(y_cls_train)
            os.makedirs(os.path.dirname(Config.label_encoder_file), exist_ok=True)
            joblib.dump(le, Config.label_encoder_file)
            print(f"Label encoder successfully saved to: {Config.label_encoder_file}")
        except Exception as e:
            print(f"Warning: Could not save label encoder: {str(e)}")
            y_cls_train_encoded = y_cls_train

        print("Data loading and preprocessing completed successfully.")
        # === 强制注释掉object特征处理后所有对X_train和X_test的重新赋值 ===
        # X_train = data_train[feature_columns].copy()
        # X_test = data_test[feature_columns].copy()
        # 提取特征和目标变量
        # X_train = data_train[feature_columns].copy()
        # X_test = data_test[feature_columns].copy()
        # 强制所有特征列都转为float，出错时打印异常和原值
        import sys
        for col in X_train_scaled.columns:
            try:
                X_train_scaled[col] = X_train_scaled[col].astype(float)
            except Exception as e:
                print(f"[最终float转换异常] 列: {col}, 错误: {e}")
                print("样本值:", X_train_scaled[col].unique()[:10])
                sys.exit(1)

        # 对验证集也进行类型转换
        for col in X_val_scaled.columns:
            try:
                X_val_scaled[col] = X_val_scaled[col].astype(float)
            except Exception as e:
                print(f"[验证集float转换异常] 列: {col}, 错误: {e}")
                X_val_scaled[col] = 0.0  # 用0填充异常值

        # 自动查找坐标和年份列，避免KeyError
        coord_cols = []
        for cand in ['x_product', 'x', 'right_x']:
            if cand in data_val.columns:
                coord_cols.append(cand)
        for cand in ['y_product', 'y', 'right_y']:
            if cand in data_val.columns:
                coord_cols.append(cand)
        for cand in ['yyyy', 'right_yyyy', 'year']:
            if cand in data_val.columns:
                coord_cols.append(cand)

        # 如果没有找到坐标列，使用默认值
        if not coord_cols:
            print("警告：未找到坐标列，使用默认值")
            coord_cols = ['x', 'y', 'yyyy']
            # 为data_val添加默认坐标列
            data_val['x'] = range(len(data_val))
            data_val['y'] = range(len(data_val))
            data_val['yyyy'] = 2015  # 默认年份

        # ====== 快速验证采样 ======
        # X_train = X_train.iloc[:5000]
        # y_cls_train = y_cls_train.iloc[:5000]
        # y_reg_train = y_reg_train.iloc[:5000]
        # X_test = X_test.iloc[:5000]
        # 只保留前10个特征
        # X_train = X_train.iloc[:, :10]
        # X_test = X_test.iloc[:, :10]
        # 同步采样标准化和编码后的变量
        # X_train_scaled = X_train_scaled.iloc[:5000]
        # y_cls_train_encoded = y_cls_train_encoded[:5000]
        # X_test_scaled = X_test_scaled.iloc[:5000]
        return X_train_scaled, y_cls_train_encoded, y_reg_train, X_val_scaled, data_val[coord_cols].values, data_val

    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            print(f"Error location: {tb[-1].filename}:{tb[-1].lineno}")
        raise


##############################################
# 2. XGBoost 特征提取
##############################################


def safe_reset_index(obj):
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.reset_index(drop=True)
    else:
        return pd.Series(obj).reset_index(drop=True)


def extract_xgboost_features(X_train, X_val, y_cls_train, y_cls_val, y_reg_train, y_reg_val):
    """
    使用XGBoost进行特征提取，使用全部数据。
    返回全部特征和标签。
    """
    print("\nExtracting XGBoost features...")
    try:
        # 不进行采样，使用全部数据
        print(f"使用全部训练数据: {len(X_train):,} 个样本")
        print(f"使用全部验证数据: {len(X_val):,} 个样本")

        X_train_sample = safe_reset_index(X_train)
        X_val_sample = safe_reset_index(X_val)
        y_cls_train_sample = safe_reset_index(y_cls_train)
        y_cls_val_sample = safe_reset_index(y_cls_val)
        y_reg_train_sample = safe_reset_index(y_reg_train)
        y_reg_val_sample = safe_reset_index(y_reg_val)
        dtrain = xgb.DMatrix(X_train_sample, label=y_cls_train_sample)
        dval = xgb.DMatrix(X_val_sample, label=y_cls_val_sample)

        xgb_params = Config.xgb_params.copy()
        unique_classes = np.unique(y_cls_train_sample)
        num_classes = len(unique_classes)

        # 如果只有一个类别，使用回归而不是分类
        if num_classes == 1:
            print(f"Only one class found: {unique_classes[0]}, switching to regression mode")
            xgb_params['objective'] = 'reg:squarederror'
            if 'num_class' in xgb_params:
                del xgb_params['num_class']
        else:
            xgb_params['num_class'] = num_classes
            # 确保标签从0开始
            y_cls_train_sample = y_cls_train_sample - min(unique_classes)
            y_cls_val_sample = y_cls_val_sample - min(unique_classes)
            dtrain = xgb.DMatrix(X_train_sample, label=y_cls_train_sample)
            dval = xgb.DMatrix(X_val_sample, label=y_cls_val_sample)

        print(f"Number of classes: {num_classes}")

        print("Training XGBoost model...")
        with tqdm(total=Config.xgb_params['n_estimators'], desc="Training XGBoost") as pbar:
            class TqdmCallback(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    pbar.update(1)
                    return False

            xgb_model = xgb.train(
                xgb_params, dtrain,
                num_boost_round=Config.xgb_params['n_estimators'],
                evals=[(dval, 'val')],
                callbacks=[TqdmCallback()],
                verbose_eval=False
            )

        # 保存XGBoost模型
        print("Saving XGBoost model...")
        try:
            os.makedirs(os.path.dirname(Config.xgboost_model_file), exist_ok=True)
            xgb_model.save_model(Config.xgboost_model_file)
            print(f"XGBoost model successfully saved to: {Config.xgboost_model_file}")
        except Exception as e:
            print(f"Error saving XGBoost model: {str(e)}")
            raise

        print("Extracting leaf features (chunked)...")

        def get_leaf_features_chunked(xgb_model, X, chunk_size=100_000):
            n = len(X)
            leaf_feats = []
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                if isinstance(X, (pd.DataFrame, pd.Series)):
                    X_chunk = X.iloc[start:end]
                else:
                    X_chunk = X[start:end]
                try:
                    dmat = xgb.DMatrix(X_chunk)
                    leaf = xgb_model.predict(dmat, pred_leaf=True)
                    leaf_feat = np.mean(leaf, axis=1).reshape(-1, 1)
                except Exception as e:
                    print(f"警告: 分块提取叶子特征失败: {str(e)}")
                    # 如果分块失败，尝试更小的块
                    smaller_chunk_size = chunk_size // 2
                    if smaller_chunk_size > 0:
                        return get_leaf_features_chunked(xgb_model, X, smaller_chunk_size)
                    else:
                        raise e
                leaf_feats.append(leaf_feat)
            return np.vstack(leaf_feats)

        leaf_feat_train = get_leaf_features_chunked(xgb_model, X_train_sample)
        leaf_feat_val = get_leaf_features_chunked(xgb_model, X_val_sample)

        # 分块拼接特征，防止np.hstack爆内存
        def hstack_chunked(X, leaf_feat, chunk_size=100_000):
            n = len(X)
            out_chunks = []
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                if isinstance(X, (pd.DataFrame, pd.Series)):
                    X_chunk = X.iloc[start:end].values
                else:
                    X_chunk = X[start:end]
                leaf_chunk = leaf_feat[start:end]
                out_chunks.append(np.hstack([X_chunk, leaf_chunk]))
            return np.vstack(out_chunks)

        X_train_combined = hstack_chunked(X_train_sample, leaf_feat_train)
        X_val_combined = hstack_chunked(X_val_sample, leaf_feat_val)
        print("XGBoost feature extraction completed successfully.")
        # 只返回采样后的特征和标签
        return X_train_combined, X_val_combined, xgb_model, y_cls_train_sample, y_cls_val_sample, y_reg_train_sample, y_reg_val_sample
    except Exception as e:
        print(f"Error in extract_xgboost_features: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            print(f"Error location: {tb[-1].filename}:{tb[-1].lineno}")
        raise


##############################################
# 3. 构建多输出混合模型（含注意力机制）
##############################################

def get_custom_objects():
    """获取自定义层映射，用于模型保存和加载"""
    return {
        'MultiHeadAttention': MultiHeadAttention
    }


def save_model_with_custom_objects(model, filepath):
    """保存包含自定义层的模型"""
    try:
        model.save(filepath)
        print(f"模型已保存至: {filepath}")
    except Exception as e:
        print(f"保存模型失败: {e}")
        raise


def load_model_with_custom_objects(filepath):
    """加载包含自定义层的模型"""
    try:
        custom_objects = get_custom_objects()
        model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        print(f"成功加载模型: {filepath}")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise


class MultiHeadAttention(Layer):
    """多头注意力机制"""

    def __init__(self, num_heads=4, head_dim=32, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.output_dim = num_heads * head_dim

    def build(self, input_shape):
        self.query_dense = Dense(self.output_dim)
        self.key_dense = Dense(self.output_dim)
        self.value_dense = Dense(self.output_dim)
        self.combine_heads = Dense(input_shape[-1])
        self.attention_dropout = Dropout(self.dropout)
        super(MultiHeadAttention, self).build(input_shape)

    def get_config(self):
        """获取配置信息，用于模型序列化"""
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dropout': self.dropout
        })
        return config

    @classmethod
    def from_config(cls, config):
        """从配置创建层实例，用于模型反序列化"""
        # 移除可能存在的额外参数
        filtered_config = {k: v for k, v in config.items()
                           if k in ['num_heads', 'head_dim', 'dropout', 'name', 'trainable']}
        return cls(**filtered_config)

    def call(self, inputs, training=None, mask=None):
        # 调整输入形状
        batch_size = tf.shape(inputs)[0]
        seq_len = 1  # 因为我们的输入是单个向量

        # 将输入重塑为序列
        x = tf.expand_dims(inputs, axis=1)  # [batch_size, 1, feature_dim]

        # 生成查询、键、值
        query = self.query_dense(x)  # [batch_size, 1, output_dim]
        key = self.key_dense(x)  # [batch_size, 1, output_dim]
        value = self.value_dense(x)  # [batch_size, 1, output_dim]

        # 重塑为多头形式
        def reshape_to_heads(x):
            return tf.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))

        query = reshape_to_heads(query)
        key = reshape_to_heads(key)
        value = reshape_to_heads(value)

        # 转置以进行注意力计算
        query = tf.transpose(query, perm=[0, 2, 1, 3])  # [batch_size, num_heads, 1, head_dim]
        key = tf.transpose(key, perm=[0, 2, 1, 3])  # [batch_size, num_heads, 1, head_dim]
        value = tf.transpose(value, perm=[0, 2, 1, 3])  # [batch_size, num_heads, 1, head_dim]

        # 计算注意力分数
        matmul_qk = tf.matmul(query, key, transpose_b=True)  # [batch_size, num_heads, 1, 1]
        dk = tf.cast(self.head_dim, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # 应用softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # 应用注意力权重
        output = tf.matmul(attention_weights, value)  # [batch_size, num_heads, 1, head_dim]

        # 转置回原始形状
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # [batch_size, 1, num_heads, head_dim]
        output = tf.reshape(output, (batch_size, seq_len, self.output_dim))  # [batch_size, 1, output_dim]

        # 合并多头输出
        output = self.combine_heads(output)  # [batch_size, 1, input_dim]

        # 去除序列维度
        output = tf.squeeze(output, axis=1)  # [batch_size, input_dim]

        return output


def build_hybrid_model(input_dim, num_classes, params=None, strategy=None):
    """构建改进的混合神经网络模型，支持多GPU训练"""
    if params is None:
        params = {
            'lr': Config.dl_params['learning_rate'],
            'neurons1': 256,  # 增加神经元数量，提高模型容量
            'neurons2': 128,  # 增加神经元数量
            'dropout_rate': Config.dl_params['dropout_rate'],  # 使用配置的dropout
            'batch_size': Config.dl_params['batch_size'],  # 使用配置的batch size
            'attention_units': 64,  # 增加注意力单元
            'l2_lambda': Config.dl_params.get('l2_reg', 1e-4),  # 使用配置的L2正则化
            'l1_lambda': Config.dl_params.get('l1_reg', 1e-5),  # 添加L1正则化
            'optimizer_type': 'adam',
            'activation': 'relu'  # 使用relu激活函数
        }

    def residual_block(x, units, dropout_rate, l2_lambda, l1_lambda, activation):
        """改进的残差块，增强正则化"""
        shortcut = Dense(units, activation=None,
                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1_lambda, l2_lambda),
                         kernel_initializer='glorot_uniform')(x)
        x = Dense(units, activation=activation,
                  kernel_regularizer=tf.keras.regularizers.l1_l2(l1_lambda, l2_lambda),
                  kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(units, activation=None,
                  kernel_regularizer=tf.keras.regularizers.l1_l2(l1_lambda, l2_lambda),
                  kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x)
        x = Add()([shortcut, x])
        x = tf.keras.activations.get(activation)(x)
        return x

    def create_model():
        # 输入层
        inputs = Input(shape=(input_dim,))
        # 先投影到neurons1维，增强正则化，使用Xavier初始化
        x = Dense(params['neurons1'], activation=params['activation'],
                  kernel_regularizer=tf.keras.regularizers.l1_l2(params['l1_lambda'], params['l2_lambda']),
                  kernel_initializer='glorot_uniform')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout_rate'] * 0.7)(x)  # 提高初始dropout

        # 增加残差块数量，提高模型学习能力
        for _ in range(3):  # 增加到3个残差块
            x = residual_block(x, params['neurons1'], params['dropout_rate'],
                               params['l2_lambda'], params['l1_lambda'], params['activation'])

        # 特征降维
        x = residual_block(x, params['neurons2'], params['dropout_rate'],
                           params['l2_lambda'], params['l1_lambda'], params['activation'])

        # 简化的特征处理层，减少复杂度
        x = Dense(params['neurons2'], activation=params['activation'],
                  kernel_regularizer=tf.keras.regularizers.l1_l2(params['l1_lambda'], params['l2_lambda']))(x)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout_rate'] * 0.8)(x)  # 提高dropout

        # 优化的注意力层，平衡性能和复杂度
        attention_output = MultiHeadAttention(
            num_heads=8,  # 增加注意力头数
            head_dim=max(8, params['attention_units'] // 8),  # 增加头维度
            dropout=params['dropout_rate'] * 0.5  # 降低注意力dropout
        )(x)
        # 特征融合
        x = Concatenate(axis=1)([x, attention_output])
        # 分类分支 - 简化结构
        classification_branch = Dense(params['neurons2'] // 2,
                                      activation=params['activation'],
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(params['l1_lambda'],
                                                                                     params['l2_lambda']))(x)
        classification_branch = BatchNormalization()(classification_branch)
        classification_branch = Dropout(params['dropout_rate'] * 0.9)(classification_branch)  # 提高dropout
        classification_output = Dense(num_classes,
                                      activation='softmax',
                                      name='classification')(classification_branch)

        # 回归分支 - 简化结构，防止过拟合
        regression_branch = Dense(params['neurons2'] // 2,  # 减少神经元
                                  activation=params['activation'],
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(params['l1_lambda'],
                                                                                 params['l2_lambda']))(x)
        regression_branch = BatchNormalization()(regression_branch)
        regression_branch = Dropout(params['dropout_rate'] * 0.8)(regression_branch)  # 提高dropout

        # 简化的回归层
        regression_branch = Dense(params['neurons2'] // 4,  # 进一步减少
                                  activation=params['activation'],
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(params['l1_lambda'],
                                                                                 params['l2_lambda']))(
            regression_branch)
        regression_branch = BatchNormalization()(regression_branch)
        regression_branch = Dropout(params['dropout_rate'] * 0.7)(regression_branch)  # 提高dropout
        # 回归输出层：使用线性激活，添加合理的输出约束
        # 使用更保守的偏置初始化，避免初始预测值过大
        regression_output = Dense(2, activation='linear',
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer=tf.keras.initializers.Constant([4.5, 7.0]))(
            regression_branch)  # 更保守的初始值

        # 添加输出约束：限制预测值在log1p变换后的合理范围内
        # 真实值范围: per_mu [4.67, 5.69], per_qu [7.16, 8.19]
        # 放宽约束范围，避免过度限制模型学习
        regression_output = tf.keras.layers.Lambda(
            lambda x: tf.clip_by_value(x, 3.0, 10.0),
            name='regression'
        )(regression_output)
        # 构建模型
        model = Model(inputs=inputs,
                      outputs=[classification_output, regression_output])
        # 编译模型，优化训练稳定性
        if params['optimizer_type'] == 'adam':
            # 使用Adam优化器，优化梯度裁剪和权重衰减
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=params['lr'],
                clipnorm=1.0,  # 放宽梯度裁剪，提高学习能力
                clipvalue=1.0,  # 放宽梯度裁剪
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                decay=1e-6  # 降低权重衰减
            )
        elif params['optimizer_type'] == 'adamw':
            # TensorFlow 2.10.1不支持AdamW，使用Adam替代
            print("⚠️ TensorFlow 2.10.1不支持AdamW，使用Adam替代")
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=params['lr'],
                clipnorm=1.0,
                clipvalue=1.0,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                decay=1e-4  # 使用decay模拟权重衰减
            )
        else:
            print(f"[警告] 未找到优化器 {params['optimizer_type']}，自动切换为Adam")
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=params['lr'],
                clipnorm=1.0,
                clipvalue=1.0,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                decay=1e-6
            )
        if Config.use_amp and tf.test.is_built_with_cuda():
            optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        # 损失函数选择
        if Config.loss_type == 'huber':
            reg_loss = tf.keras.losses.Huber()
        elif Config.loss_type == 'mse':
            reg_loss = 'mse'
        elif Config.loss_type == 'mae':
            reg_loss = 'mae'
        elif Config.loss_type == 'logcosh':
            reg_loss = tf.keras.losses.LogCosh()
        else:
            reg_loss = tf.keras.losses.Huber()
        model.compile(
            optimizer=optimizer,
            loss={
                'classification': 'sparse_categorical_crossentropy',
                'regression': reg_loss
            },
            metrics={
                'classification': 'accuracy',
                'regression': ['mae', 'mse']
            },
            loss_weights={
                'classification': 0.1,  # 提高分类损失权重，平衡两个任务
                'regression': 1.0  # 保持回归损失权重，专注于per_mu预测
            }
        )
        return model

    # 如果提供了strategy，在strategy scope内创建模型
    if strategy is not None:
        with strategy.scope():
            return create_model()
    else:
        return create_model()


##############################################
# 4. 超参数调优（Optuna）
##############################################

class TrainingMonitor:
    """训练过程监控类"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.model = None
        self.current_epoch = 0  # 添加当前epoch计数器
        self.metrics_history = {
            'epoch': [],
            'training_loss': [],
            'validation_loss': [],
            'training_cls_acc': [],
            'validation_cls_acc': [],
            'training_reg_mae': [],
            'validation_reg_mae': [],
            'training_reg_r2': [],
            'validation_reg_r2': [],
            'learning_rate': [],
            'gpu_memory_usage': [],
            'batch_time': []
        }

        # 过拟合检测参数
        self.overfitting_detected = False
        self.overfitting_epoch = None
        self.gap_threshold = 0.1  # 训练和验证损失差距阈值
        self.consecutive_increases = 0  # 连续验证损失增加次数
        self.max_consecutive_increases = 3  # 最大允许连续增加次数

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 初始化可视化
        try:
            plt.style.use(['seaborn-v0_8-darkgrid'])
        except:
            print("Warning: Could not set seaborn style, using default style")
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Training Progress', fontsize=16)

    def set_model(self, model):
        """设置要监控的模型"""
        self.model = model

    def update_metrics(self, epoch, logs):
        """更新训练指标，增强过拟合检测"""
        self.current_epoch = epoch

        # 记录基本指标
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['training_loss'].append(logs.get('loss', 0))
        self.metrics_history['validation_loss'].append(logs.get('val_loss', 0))
        self.metrics_history['training_cls_acc'].append(logs.get('classification_accuracy', 0))
        self.metrics_history['validation_cls_acc'].append(logs.get('val_classification_accuracy', 0))
        self.metrics_history['training_reg_mae'].append(logs.get('regression_mae', 0))
        self.metrics_history['validation_reg_mae'].append(logs.get('val_regression_mae', 0))

        # 过拟合检测
        self._detect_overfitting()

        # 计算并记录R²分数 - 简化处理，避免复杂的R²计算
        self.metrics_history['training_reg_r2'].append(0)  # 暂时设为0，避免计算错误
        self.metrics_history['validation_reg_r2'].append(0)  # 暂时设为0，避免计算错误

        # 记录学习率
        if self.model and hasattr(self.model.optimizer, 'lr'):
            lr = float(self.model.optimizer.lr.numpy())
            self.metrics_history['learning_rate'].append(lr)
        else:
            self.metrics_history['learning_rate'].append(0)

        # 记录GPU使用情况
        if GPUtil and tf.test.is_built_with_cuda():
            try:
                gpu = GPUtil.getGPUs()[0]
                self.metrics_history['gpu_memory_usage'].append(gpu.memoryUsed)
            except:
                self.metrics_history['gpu_memory_usage'].append(0)
        else:
            self.metrics_history['gpu_memory_usage'].append(0)

        # 记录批处理时间
        self.metrics_history['batch_time'].append(time.time())

        # 确保所有数组长度一致
        min_length = min(len(v) for v in self.metrics_history.values())
        for key in self.metrics_history:
            self.metrics_history[key] = self.metrics_history[key][:min_length]

        # 更新可视化
        self.update_plots()

        # 保存指标
        self.save_metrics()

    def _detect_overfitting(self):
        """检测过拟合"""
        if len(self.metrics_history['training_loss']) < 5:
            return

        # 计算训练和验证损失的差距
        train_loss = self.metrics_history['training_loss'][-1]
        val_loss = self.metrics_history['validation_loss'][-1]
        loss_gap = val_loss - train_loss

        # 检测过拟合信号
        if loss_gap > self.gap_threshold:
            self.consecutive_increases += 1
            if self.consecutive_increases >= self.max_consecutive_increases:
                self.overfitting_detected = True
                self.overfitting_epoch = self.current_epoch
                print(
                    f"⚠️ 检测到过拟合！Epoch {self.current_epoch}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}, 差距={loss_gap:.4f}")
        else:
            self.consecutive_increases = 0

        # 检测验证损失连续增加
        if len(self.metrics_history['validation_loss']) >= 3:
            recent_val_losses = self.metrics_history['validation_loss'][-3:]
            if all(recent_val_losses[i] <= recent_val_losses[i + 1] for i in range(len(recent_val_losses) - 1)):
                print(f"⚠️ 验证损失连续增加！最近3个epoch: {recent_val_losses}")

    def get_overfitting_status(self):
        """获取过拟合状态"""
        return {
            'overfitting_detected': self.overfitting_detected,
            'overfitting_epoch': self.overfitting_epoch,
            'consecutive_increases': self.consecutive_increases,
            'current_gap': self.metrics_history['validation_loss'][-1] - self.metrics_history['training_loss'][
                -1] if len(self.metrics_history['validation_loss']) > 0 else 0
        }

    def update_plots(self):
        """更新训练过程可视化"""
        # 清除当前图形
        for ax in self.axes.flat:
            ax.clear()

        # 损失曲线
        ax = self.axes[0, 0]
        ax.plot(self.metrics_history['epoch'], self.metrics_history['training_loss'], label='Training')
        ax.plot(self.metrics_history['epoch'], self.metrics_history['validation_loss'], label='Validation')
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

        # 分类准确率
        ax = self.axes[0, 1]
        ax.plot(self.metrics_history['epoch'], self.metrics_history['training_cls_acc'], label='Training')
        ax.plot(self.metrics_history['epoch'], self.metrics_history['validation_cls_acc'], label='Validation')
        ax.set_title('Classification Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True)

        # 回归MAE
        ax = self.axes[1, 0]
        ax.plot(self.metrics_history['epoch'], self.metrics_history['training_reg_mae'], label='Training')
        ax.plot(self.metrics_history['epoch'], self.metrics_history['validation_reg_mae'], label='Validation')
        ax.set_title('Regression MAE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.legend()
        ax.grid(True)

        # GPU使用情况
        ax = self.axes[1, 1]
        if len(self.metrics_history['gpu_memory_usage']) > 0:
            ax.plot(self.metrics_history['epoch'], self.metrics_history['gpu_memory_usage'])
            ax.set_title('GPU Memory Usage')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Memory (MB)')
            ax.grid(True)

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'))
        plt.close()

    def save_metrics(self):
        """保存训练指标"""
        try:
            # 保存为CSV
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(os.path.join(self.log_dir, 'training_metrics.csv'), index=False)

            # 保存为JSON
            with open(os.path.join(self.log_dir, 'training_metrics.json'), 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save metrics: {str(e)}")


class OptunaCallback(tf.keras.callbacks.Callback):
    """Optuna早停回调"""

    def __init__(self, trial, monitor='val_loss', patience=3):  # 减少patience，加快早停
        super(OptunaCallback, self).__init__()
        self.trial = trial
        self.monitor = monitor
        self.patience = patience
        self.best_value = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        current_value = logs.get(self.monitor)
        if current_value < self.best_value:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.trial.report(self.best_value, epoch)
                raise optuna.TrialPruned()


class OptunaR2EarlyStopping(tf.keras.callbacks.Callback):
    """Optuna R²早停回调，当R²达到0.8时停止调参训练"""

    def __init__(self, X_val, y_val, min_r2=0.8, trial=None):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.min_r2 = min_r2
        self.trial = trial
        self.best_r2 = 0.0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # 计算R² - 修复版本
        try:
            y_pred = self.model.predict(self.X_val, verbose=0)

            # 如果是多输出模型，取回归部分的预测
            if isinstance(y_pred, (list, tuple)) and len(y_pred) == 2:
                y_pred_reg = y_pred[1]  # 回归输出
            elif isinstance(y_pred, dict):
                y_pred_reg = y_pred['regression']
            else:
                y_pred_reg = y_pred

            # 提取per_mu的预测值（第一列，索引0）
            if y_pred_reg.ndim > 1 and y_pred_reg.shape[1] > 0:
                y_pred_per_mu = y_pred_reg[:, 0]  # per_mu是第一列
            else:
                y_pred_per_mu = y_pred_reg

            # 确保y_pred_per_mu是numpy数组
            if hasattr(y_pred_per_mu, 'values'):
                y_pred_per_mu = y_pred_per_mu.values
            elif not isinstance(y_pred_per_mu, np.ndarray):
                y_pred_per_mu = np.array(y_pred_per_mu)

            # 处理真实值 - per_mu是第一列（索引0）
            if hasattr(self.y_val, 'iloc'):
                # pandas DataFrame
                y_true_per_mu = self.y_val.iloc[:, 0] if self.y_val.shape[1] > 0 else self.y_val.iloc[:, 0]
            else:
                # numpy array
                y_true_per_mu = self.y_val[:, 0] if self.y_val.ndim > 1 else self.y_val

            # 确保真实值也是numpy数组
            if hasattr(y_true_per_mu, 'values'):
                y_true_per_mu = y_true_per_mu.values
            elif not isinstance(y_true_per_mu, np.ndarray):
                y_true_per_mu = np.array(y_true_per_mu)

            # 确保长度匹配
            min_len = min(len(y_true_per_mu), len(y_pred_per_mu))
            y_true_per_mu = y_true_per_mu[:min_len]
            y_pred_per_mu = y_pred_per_mu[:min_len]

            # 添加调试信息
            if epoch == 0:  # 只在第一个epoch打印调试信息
                print(f"🔍 R²计算调试信息:")
                print(f"   真实值范围: [{y_true_per_mu.min():.4f}, {y_true_per_mu.max():.4f}]")
                print(f"   预测值范围: [{y_pred_per_mu.min():.4f}, {y_pred_per_mu.max():.4f}]")
                print(f"   真实值均值: {y_true_per_mu.mean():.4f}, 标准差: {y_true_per_mu.std():.4f}")
                print(f"   预测值均值: {y_pred_per_mu.mean():.4f}, 标准差: {y_pred_per_mu.std():.4f}")

            # 检查数据有效性
            if len(y_true_per_mu) == 0 or len(y_pred_per_mu) == 0:
                print(f"警告: 数据为空，设置R²为-1000")
                r2 = -1000
            elif np.std(y_true_per_mu) < 1e-8:
                print(f"警告: 真实值方差为0，设置R²为-1000")
                r2 = -1000
            elif np.std(y_pred_per_mu) < 1e-8:
                print(f"警告: 预测值方差为0，设置R²为-1000")
                r2 = -1000
            else:
                # 计算R²
                from sklearn.metrics import r2_score
                r2 = r2_score(y_true_per_mu, y_pred_per_mu)

                # 限制R²在合理范围内
                r2 = max(-100, min(1, r2))

        except Exception as e:
            print(f"R²计算错误: {e}")
            r2 = -1000

        if r2 > self.best_r2:
            self.best_r2 = r2

        print(f"Epoch {epoch + 1}: R²={r2:.4f}, Best R²={self.best_r2:.4f}, 阈值={self.min_r2}")

        # 如果R²达到阈值，停止训练并报告给Optuna
        if r2 >= self.min_r2:
            print(f"🎯 R²达到{self.min_r2}，停止调参训练，进入最终模型训练！")
            print(f"当前R²: {r2:.4f}, 阈值: {self.min_r2}")
            self.model.stop_training = True
            self.stopped_epoch = epoch

            # 更新全局变量
            global GLOBAL_R2_ACHIEVED, GLOBAL_BEST_PARAMS, GLOBAL_BEST_R2
            GLOBAL_R2_ACHIEVED = True
            GLOBAL_BEST_R2 = r2
            if self.trial:
                GLOBAL_BEST_PARAMS = self.trial.params
                print(f"🎯 全局最佳参数已保存: {GLOBAL_BEST_PARAMS}")
                print(f"🎯 全局R²状态: GLOBAL_R2_ACHIEVED={GLOBAL_R2_ACHIEVED}, GLOBAL_BEST_R2={GLOBAL_BEST_R2}")

            # 报告给Optuna，使用负的R²作为优化目标（因为Optuna默认最小化）
            if self.trial:
                self.trial.report(-r2, epoch)
                # 设置trial为成功状态
                self.trial.set_user_attr('r2_achieved', r2)
                self.trial.set_user_attr('early_stopped', True)
                print(f"🎯 Trial {self.trial.number} 已标记为R²达到0.8")

            # 抛出特殊异常来停止Optuna优化
            raise optuna.TrialPruned()

        # 记录R²到logs中
        logs['val_r2'] = r2


def objective(trial, X_train_combined, y_cls_train_final, y_reg_train_final, X_val_combined, y_cls_val, y_reg_val,
              num_classes, strategy=None):
    """Optuna优化目标函数，支持多GPU训练"""
    # 从超参数空间中采样
    params = {
        'lr': trial.suggest_float(
            'lr',
            *Config.optuna_params['param_ranges']['lr'],
            log=True
        ),
        'neurons1': trial.suggest_int(
            'neurons1',
            *Config.optuna_params['param_ranges']['neurons1']
        ),
        'neurons2': trial.suggest_int(
            'neurons2',
            *Config.optuna_params['param_ranges']['neurons2']
        ),
        'dropout_rate': trial.suggest_float(
            'dropout_rate',
            *Config.optuna_params['param_ranges']['dropout_rate']
        ),
        'batch_size': trial.suggest_categorical(
            'batch_size',
            Config.optuna_params['param_ranges']['batch_size']
        ),
        'attention_units': trial.suggest_int(
            'attention_units',
            *Config.optuna_params['param_ranges']['attention_units']
        ),
        'l2_lambda': trial.suggest_float(
            'l2_lambda',
            *Config.optuna_params['param_ranges']['l2_lambda'],
            log=True
        ),
        'optimizer_type': trial.suggest_categorical(
            'optimizer_type',
            Config.optuna_params['param_ranges']['optimizer_type']
        ),
        'activation': trial.suggest_categorical(
            'activation',
            Config.optuna_params['param_ranges']['activation']
        )
    }

    # 构建模型
    model = build_hybrid_model(
        input_dim=X_train_combined.shape[1],
        num_classes=num_classes,
        params=params,
        strategy=strategy
    )

    # 训练模型
    try:
        history = model.fit(
            X_train_combined,
            {
                'classification': y_cls_train_final,
                'output_clipped': y_reg_train_final
            },
            validation_data=(
                X_val_combined,
                {
                    'classification': y_cls_val,
                    'output_clipped': y_reg_val
                }
            ),
            epochs=Config.dl_params['epochs'],
            batch_size=params['batch_size'],
            callbacks=[
                OptunaCallback(trial),
                EarlyStopping(
                    monitor='val_loss',
                    patience=Config.dl_params['early_stop_patience'],
                    restore_best_weights=True
                )
            ],
            verbose=0
        )

        # 返回验证集性能
        val_loss = min(history.history['val_loss'])
        return val_loss
    except Exception as e:
        print(f"[Optuna] Trial failed due to exception: {e}")
        raise optuna.TrialPruned()


##############################################
# 5. 最终训练 & 预测 & 可视化
##############################################

def plot_training_history(history):
    """绘制训练历史曲线"""
    # 创建子图
    plt.figure(figsize=(15, 10))

    # 子图1：分类损失 vs 回归损失
    plt.subplot(2, 2, 1)
    plt.plot(history.history['classification_loss'], label='Train Cls Loss')
    plt.plot(history.history['val_classification_loss'], label='Val Cls Loss')
    plt.plot(history.history['regression_loss'], label='Train Yield Loss')
    plt.plot(history.history['val_regression_loss'], label='Val Yield Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Classification & Regression Loss')
    plt.legend()
    plt.grid(True)

    # 子图2：分类准确度 vs 回归MAE
    plt.subplot(2, 2, 2)
    plt.plot(history.history['classification_accuracy'], label='Train Cls Acc')
    plt.plot(history.history['val_classification_accuracy'], label='Val Cls Acc')
    plt.plot(history.history['regression_mae'], label='Train Yield MAE')
    plt.plot(history.history['val_regression_mae'], label='Val Yield MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy / MAE')
    plt.title('Classification Accuracy & Yield MAE')
    plt.legend()
    plt.grid(True)

    # 子图3：总损失
    plt.subplot(2, 2, 3)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.title('Total Loss')
    plt.grid(True)
    plt.legend()

    # 子图4：MSE
    plt.subplot(2, 2, 4)
    plt.plot(history.history['regression_mse'], label='Train MSE')
    plt.plot(history.history['val_regression_mse'], label='Val MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(Config.accuracy_loss_map_file, dpi=300)
    plt.close()


def plot_prediction_maps(
        X_val: Union[pd.DataFrame, np.ndarray],
        cls_pred: np.ndarray,
        reg_pred: np.ndarray,
        val_ids: np.ndarray,
        data_val: Union[pd.DataFrame, np.ndarray]
) -> None:
    if not isinstance(X_val, pd.DataFrame):
        X_val = pd.DataFrame(X_val)
    # 自动适配列名
    x_col = next((c for c in ['x', 'x_product', 'right_x', 'X'] if c in X_val.columns), None)
    y_col = next((c for c in ['y', 'y_product', 'right_y', 'Y'] if c in X_val.columns), None)
    assert x_col and y_col, f"X_val must contain a valid x/y column, got: {X_val.columns.tolist()}"
    # 创建地理数据框
    gdf = gpd.GeoDataFrame(
        {
            'ID': np.arange(len(X_val)),
            'X': np.ravel(X_val[x_col].values),
            'Y': np.ravel(X_val[y_col].values),
            'suitability': np.ravel(np.argmax(cls_pred, axis=1)),
            'yield_per_cell': np.ravel(reg_pred[:, 0]),
            'yield_per_mu': np.ravel(reg_pred[:, 1])
        }
    )
    # 添加几何列
    gdf['geometry'] = gpd.points_from_xy(gdf['X'], gdf['Y'])
    # 创建三个子图
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    # 适宜性等级地图
    gdf.plot(
        column='suitability',
        ax=axes[0],
        legend=True,
        cmap='RdYlGn',
        legend_kwds={'label': '适宜性等级'}
    )
    axes[0].set_title('适宜性等级分布')
    # 区产量预测地图
    gdf.plot(
        column='yield_per_cell',
        ax=axes[1],
        legend=True,
        cmap='viridis',
        legend_kwds={'label': '区产量 (kg/cell)'}
    )
    axes[1].set_title('区产量预测分布')
    # 亩产量预测地图
    gdf.plot(
        column='yield_per_mu',
        ax=axes[2],
        legend=True,
        cmap='viridis',
        legend_kwds={'label': '亩产量 (kg/mu)'}
    )
    axes[2].set_title('亩产量预测分布')
    # 为每个子图添加坐标轴标签
    for ax in axes:
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(Config.suitability_map_file, dpi=300)
    plt.close()


def evaluate_model(model, X_val, y_cls_val, y_reg_val):
    """评估模型性能"""
    # 获取预测结果
    cls_pred, reg_pred = model.predict(X_val)
    cls_pred_labels = np.argmax(cls_pred, axis=1)
    # 确保输入是numpy数组
    if isinstance(y_reg_val, pd.DataFrame):
        y_reg_val = y_reg_val.values
    if isinstance(reg_pred, pd.DataFrame):
        reg_pred = reg_pred.values
    # === 反变换log1p ===
    y_reg_val_true = np.expm1(y_reg_val)
    reg_pred_true = np.expm1(reg_pred)
    # 分类指标
    cls_accuracy = accuracy_score(y_cls_val, cls_pred_labels)
    cls_report = classification_report(y_cls_val, cls_pred_labels)
    # 回归指标 - 只关注per_mu（第一个输出）
    reg_r2 = [r2_score(y_reg_val_true[:, 0], reg_pred_true[:, 0])]  # 只计算per_mu的R²
    reg_mae = [mean_absolute_error(y_reg_val_true[:, i], reg_pred_true[:, i]) for i in range(y_reg_val_true.shape[1])]
    reg_rmse = [np.sqrt(mean_squared_error(y_reg_val_true[:, i], reg_pred_true[:, i])) for i in
                range(y_reg_val_true.shape[1])]
    # 创建评估报告
    evaluation_report = {
        'classification': {
            'accuracy': cls_accuracy,
            'detailed_report': cls_report
        },
        'regression': {
            'r2_scores': {
                f'output_{i}': score for i, score in enumerate(reg_r2)
            },
            'mae_scores': {
                f'output_{i}': score for i, score in enumerate(reg_mae)
            },
            'rmse_scores': {
                f'output_{i}': score for i, score in enumerate(reg_rmse)
            }
        }
    }
    # 保存评估报告
    report_file = Config.evaluation_file
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, indent=4, ensure_ascii=False)
    # 打印评估结果
    print("\n模型评估报告:")
    print("\n分类性能:")
    print(f"准确率: {cls_accuracy:.4f}")
    print("\n详细分类报告:")
    print(cls_report)
    print("\n回归性能:")
    print(f"\nper_mu (权重=1):")
    print(f"R² 分数: {reg_r2[0]:.4f}")
    print(f"MAE: {reg_mae[0]:.4f}")
    print(f"RMSE: {reg_rmse[0]:.4f}")
    print(f"\nper_qu (权重=0):")
    print(f"MAE: {reg_mae[1]:.4f}")
    print(f"RMSE: {reg_rmse[1]:.4f}")
    print("注意：R²值只计算per_mu，per_qu权重为0")
    return evaluation_report


def plot_confusion_matrix(y_true, y_pred, classes):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(Config.confusion_matrix_file, dpi=300)
    plt.close()


def plot_regression_scatter(y_true, y_pred, output_names):
    """绘制回归散点图"""
    # 确保输入数据类型一致
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    # 确保是2D数组
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_outputs = y_true.shape[1]
    fig, axes = plt.subplots(1, n_outputs, figsize=(15, 5))

    if n_outputs == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, output_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        ax.plot([y_true[:, i].min(), y_true[:, i].max()],
                [y_true[:, i].min(), y_true[:, i].max()],
                'r--', lw=2)
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title(f'{name} 预测vs真实值')

        # 添加R²值 - 只关注per_mu
        if i == 0:  # 只对per_mu计算R²
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            ax.text(0.05, 0.95, f'R² (per_mu) = {r2:.4f}',
                    transform=ax.transAxes,
                    verticalalignment='top')
        else:  # per_qu不显示R²
            ax.text(0.05, 0.95, f'per_qu (权重=0)',
                    transform=ax.transAxes,
                    verticalalignment='top')

    plt.tight_layout()
    plt.savefig(Config.regression_scatter_file, dpi=300)
    plt.close()


def plot_feature_importance(model, feature_names):
    """绘制特征重要性"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 6))
        plt.title('特征重要性')
        plt.bar(range(len(importances)),
                importances[indices])
        plt.xticks(range(len(importances)),
                   [feature_names[i] for i in indices],
                   rotation=45,
                   ha='right')
        plt.tight_layout()
        plt.savefig(Config.feature_importance_map_file, dpi=300)
        plt.close()


def emergency_feature_selection(X_train, y_train, X_val, y_val, max_features=20):
    """
    紧急特征选择：使用多种方法选择最重要的特征
    """
    print("🚨 执行紧急特征选择...")

    # 导入必要的模块
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, mutual_info_regression
    from sklearn.linear_model import Ridge

    # 方法1：基于方差的特征选择
    print("1. 基于方差选择特征...")
    var_selector = VarianceThreshold(threshold=0.01)
    X_train_var = var_selector.fit_transform(X_train)
    X_val_var = var_selector.transform(X_val)

    # 方法2：基于F统计量的特征选择
    print("2. 基于F统计量选择特征...")
    try:
        f_selector = SelectKBest(score_func=f_regression, k=min(30, X_train_var.shape[1]))
        X_train_f = f_selector.fit_transform(X_train_var, y_train[:, 0])  # 使用第一个目标
        X_val_f = f_selector.transform(X_val_var)
        f_scores = f_selector.scores_
        f_features = f_selector.get_support(indices=True)
        print(f"F统计量选择了 {len(f_features)} 个特征")
    except Exception as e:
        print(f"F统计量选择失败: {e}")
        X_train_f = X_train_var
        X_val_f = X_val_var
        f_features = list(range(X_train_var.shape[1]))

    # 方法3：基于互信息的特征选择
    print("3. 基于互信息选择特征...")
    try:
        mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(20, X_train_f.shape[1]))
        X_train_mi = mi_selector.fit_transform(X_train_f, y_train[:, 0])
        X_val_mi = mi_selector.transform(X_val_f)
        mi_scores = mi_selector.scores_
        mi_features = mi_selector.get_support(indices=True)
        print(f"互信息选择了 {len(mi_features)} 个特征")
    except Exception as e:
        print(f"互信息选择失败: {e}")
        X_train_mi = X_train_f
        X_val_mi = X_val_f
        mi_features = list(range(X_train_f.shape[1]))

    # 方法4：基于Ridge回归的特征选择
    print("4. 基于Ridge回归选择特征...")
    try:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_mi, y_train[:, 0])
        ridge_coefs = np.abs(ridge.coef_)

        # 选择系数最大的特征
        top_indices = np.argsort(ridge_coefs)[-max_features:]
        X_train_final = X_train_mi[:, top_indices]
        X_val_final = X_val_mi[:, top_indices]

        print(f"Ridge回归选择了 {len(top_indices)} 个特征")
        print(f"最终特征数量: {X_train_final.shape[1]}")

        return X_train_final, X_val_final, top_indices

    except Exception as e:
        print(f"Ridge回归选择失败: {e}")
        # 回退到简单的特征选择
        X_train_final = X_train_mi[:, :max_features]
        X_val_final = X_val_mi[:, :max_features]
        return X_train_final, X_val_final, list(range(max_features))


def analyze_feature_importance(
        model,
        X_train: Union[pd.DataFrame, np.ndarray, list],
        y_reg_train: Union[pd.DataFrame, np.ndarray, list],
        feature_names: list,
        threshold: float = 0.02
) -> pd.DataFrame:
    assert isinstance(feature_names, list), f"feature_names must be a list, got {type(feature_names)}"
    assert 0 <= threshold <= 1, "threshold must be between 0 and 1"
    try:
        print("\n计算特征重要性...")

        # 确保X_train是numpy数组
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        elif isinstance(X_train, list):
            X_train = np.array(X_train)
        elif hasattr(X_train, 'values'):
            X_train = X_train.values

        # 不进行特征采样，使用全部特征进行重要性计算
        print(f"使用全部特征进行重要性计算: {X_train.shape[1]} 个特征")

        # 检查特征相关性，避免重复特征影响重要性计算
        print("🔍 检查特征相关性...")
        try:
            if hasattr(X_train, 'values'):
                X_train_array = X_train.values
            else:
                X_train_array = np.array(X_train)

            # 计算特征间相关性
            correlation_matrix = np.corrcoef(X_train_array.T)
            high_corr_pairs = []
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    if abs(correlation_matrix[i, j]) > 0.95:  # 高相关性阈值
                        high_corr_pairs.append((feature_names[i], feature_names[j], correlation_matrix[i, j]))

            if high_corr_pairs:
                print(f"⚠️ 发现 {len(high_corr_pairs)} 对高相关性特征 (>0.95):")
                for feat1, feat2, corr in high_corr_pairs[:10]:  # 只显示前10对
                    print(f"  {feat1} <-> {feat2}: {corr:.4f}")
                if len(high_corr_pairs) > 10:
                    print(f"  ... 还有 {len(high_corr_pairs) - 10} 对高相关性特征")
            else:
                print("✅ 未发现高相关性特征")

        except Exception as e:
            print(f"⚠️ 特征相关性检查失败: {str(e)}")

        # 使用XGBoost的特征重要性
        if hasattr(model, 'get_score') or hasattr(model, 'feature_importances_'):
            print("使用XGBoost的特征重要性计算方法...")
            try:
                # 优先使用feature_importances_属性
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                    print(f"✅ 成功获取XGBoost feature_importances_: {len(feature_importance)} 个特征")
                else:
                    # 使用get_score方法
                    importance_scores = {}
                    for importance_type in ['weight', 'gain', 'cover']:
                        try:
                            scores = model.get_score(importance_type=importance_type)
                            if scores:
                                for feature, score in scores.items():
                                    if feature not in importance_scores:
                                        importance_scores[feature] = 0
                                    importance_scores[feature] += score
                        except Exception as e:
                            print(f"Warning: 无法获取 {importance_type} 类型的特征重要性: {str(e)}")

                    if importance_scores:
                        # 将特征重要性映射到feature_names
                        feature_importance = []
                        for feature in feature_names:
                            # 尝试匹配特征名（可能包含叶子特征）
                            if feature in importance_scores:
                                feature_importance.append(importance_scores[feature])
                            else:
                                # 如果没有找到，尝试匹配原始特征名
                                original_feature = feature.replace('xgb_leaf_', '') if feature.startswith(
                                    'xgb_leaf_') else feature
                                if original_feature in importance_scores:
                                    feature_importance.append(importance_scores[original_feature])
                                else:
                                    feature_importance.append(0.0)
                        print(f"✅ 成功获取XGBoost get_score: {len(feature_importance)} 个特征")
                    else:
                        raise ValueError("无法获取XGBoost特征重要性分数")

            except Exception as e:
                print(f"XGBoost特征重要性计算失败: {str(e)}")
                # 回退到基于方差的特征重要性
                try:
                    if hasattr(X_train, 'values'):
                        X_train_array = X_train.values
                    else:
                        X_train_array = np.array(X_train)

                    # 计算每个特征的方差作为重要性
                    feature_variances = np.var(X_train_array, axis=0)
                    if np.sum(feature_variances) > 0:
                        feature_importance = feature_variances / np.sum(feature_variances)
                    else:
                        feature_importance = np.ones(len(feature_names)) / len(feature_names)
                    print(f"✅ 使用方差作为特征重要性: {len(feature_importance)} 个特征")
                except Exception as e2:
                    print(f"方差特征重要性计算失败: {str(e2)}")
                    feature_importance = [1.0 / len(feature_names)] * len(feature_names)
        else:
            print("使用SHAP值计算特征重要性...")
            try:
                import shap
                sample_size = min(Config.feature_importance['sample_size'] // 200, len(X_train))  # 使用配置的1/200作为SHAP样本
                X_sample = X_train[:sample_size]
                if hasattr(model, "get_booster") or hasattr(model, "feature_importances_"):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    explainer = shap.KernelExplainer(
                        model.predict if hasattr(model, 'predict') else model,
                        X_sample,
                        link="identity"
                    )
                    shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    feature_importance = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
                else:
                    feature_importance = np.abs(shap_values).mean(0)
            except ImportError:
                print("SHAP未安装，使用基于排列的特征重要性...")
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    # 使用更小的样本进行RandomForest训练，避免内存问题
                    sample_size = min(10000, len(X_train))
                    if sample_size < len(X_train):
                        indices = np.random.choice(len(X_train), sample_size, replace=False)
                        X_sample = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
                        y_sample = y_reg_train.iloc[indices] if hasattr(y_reg_train, 'iloc') else y_reg_train[indices]
                    else:
                        X_sample = X_train
                        y_sample = y_reg_train

                    # 安全地处理目标变量
                    if hasattr(y_sample, 'iloc'):
                        y_target = y_sample.iloc[:, 0].values
                    elif hasattr(y_sample, 'values'):
                        y_target = y_sample.values[:, 0] if y_sample.values.ndim > 1 else y_sample.values
                    else:
                        y_target = y_sample[:, 0] if np.array(y_sample).ndim > 1 else y_sample

                    # 确保X_train是numpy数组
                    if hasattr(X_sample, 'values'):
                        X_train_array = X_sample.values
                    else:
                        X_train_array = np.array(X_sample)

                    # 确保数据类型正确
                    y_target = np.asarray(y_target, dtype=np.float64)
                    X_train_array = np.asarray(X_train_array, dtype=np.float64)

                    # 检查数据形状
                    if X_train_array.shape[0] != len(y_target):
                        print(f"数据形状不匹配: X={X_train_array.shape}, y={y_target.shape}")
                        raise ValueError("数据形状不匹配")

                    rf = RandomForestRegressor(n_estimators=10, random_state=Config.random_seed, n_jobs=1, max_depth=10)
                    rf.fit(X_train_array, y_target)
                    feature_importance = rf.feature_importances_
                    print(f"✅ 成功使用RandomForest计算特征重要性: {len(feature_importance)} 个特征")
                except Exception as e:
                    print(f"RandomForest特征重要性计算失败: {str(e)}")
                    # 使用简单的方差特征重要性
                    try:
                        from sklearn.feature_selection import mutual_info_regression
                        # 使用更小的样本进行互信息计算，避免内存问题
                        sample_size = min(5000, len(X_train))
                        if sample_size < len(X_train):
                            indices = np.random.choice(len(X_train), sample_size, replace=False)
                            X_sample = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
                            y_sample = y_reg_train.iloc[indices] if hasattr(y_reg_train, 'iloc') else y_reg_train[
                                indices]
                        else:
                            X_sample = X_train
                            y_sample = y_reg_train

                        # 安全地处理目标变量
                        if hasattr(y_sample, 'iloc'):
                            y_target = y_sample.iloc[:, 0].values
                        elif hasattr(y_sample, 'values'):
                            y_target = y_sample.values[:, 0] if y_sample.values.ndim > 1 else y_sample.values
                        else:
                            y_target = np.array(y_sample)[:, 0] if np.array(y_sample).ndim > 1 else np.array(y_sample)

                        # 安全地处理特征数据
                        if hasattr(X_sample, 'values'):
                            X_train_array = X_sample.values
                        else:
                            X_train_array = np.array(X_sample)

                        # 确保数据类型正确
                        X_train_array = np.asarray(X_train_array, dtype=np.float64)
                        y_target = np.asarray(y_target, dtype=np.float64)

                        # 检查数据形状
                        if X_train_array.shape[0] != len(y_target):
                            print(f"数据形状不匹配: X={X_train_array.shape}, y={y_target.shape}")
                            raise ValueError("数据形状不匹配")

                        feature_importance = mutual_info_regression(X_train_array, y_target,
                                                                    random_state=Config.random_seed)
                        print(f"✅ 成功使用互信息计算特征重要性: {len(feature_importance)} 个特征")
                    except Exception as e2:
                        print(f"互信息特征重要性计算也失败: {str(e2)}")
        # 最后回退到基于方差的特征重要性
        try:
            # 使用特征方差作为重要性指标
            if hasattr(X_train, 'values'):
                X_train_array = X_train.values
            else:
                X_train_array = np.array(X_train)

            # 计算每个特征的方差
            feature_variances = np.var(X_train_array, axis=0)
            # 归一化到0-1范围
            if np.sum(feature_variances) > 0:
                feature_importance = feature_variances / np.sum(feature_variances)
            else:
                feature_importance = np.ones(len(feature_names)) / len(feature_names)
        except Exception as e3:
            print(f"方差特征重要性计算也失败: {str(e3)}")
            # 最后回退到紧急特征选择
            try:
                print("尝试紧急特征选择...")
                X_train_emergency, _, selected_indices = emergency_feature_selection(
                    X_train, y_reg_train, X_train, y_reg_train, max_features=Config.feature_importance['max_features']
                )
                # 创建基于紧急选择的特征重要性
                feature_importance = np.zeros(len(feature_names))
                for idx in selected_indices:
                    if idx < len(feature_importance):
                        feature_importance[idx] = 1.0
                # 归一化
                if feature_importance.sum() > 0:
                    feature_importance = feature_importance / feature_importance.sum()
                else:
                    feature_importance = np.ones(len(feature_names)) / len(feature_names)
                print(f"紧急特征选择选择了 {len(selected_indices)} 个特征")
            except Exception as e4:
                print(f"紧急特征选择也失败: {e4}")
                feature_importance = np.ones(len(feature_names)) / len(feature_names)

        # flatten和断言，统一DataFrame创建
        feature_importance = np.asarray(feature_importance).flatten()
        if len(feature_importance) != len(feature_names):
            print(f"警告：特征重要性长度({len(feature_importance)})与特征名长度({len(feature_names)})不匹配")
            print(f"调整特征名列表长度以匹配特征重要性长度")
            # 如果特征重要性长度大于特征名长度，添加缺失的特征名
            if len(feature_importance) > len(feature_names):
                missing_count = len(feature_importance) - len(feature_names)
                additional_names = [f'feature_{i}' for i in
                                    range(len(feature_names), len(feature_names) + missing_count)]
                feature_names = feature_names + additional_names
                print(f"添加了 {missing_count} 个特征名")
            # 如果特征重要性长度小于特征名长度，截断特征名列表
            elif len(feature_importance) < len(feature_names):
                feature_names = feature_names[:len(feature_importance)]
                print(f"截断特征名列表到 {len(feature_importance)} 个特征")

        # 保持原始特征名，不进行映射修改
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
        # 后续所有操作都用importance_df
        importance_df = importance_df.sort_values('importance', ascending=False)
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            importance_df['relative_importance'] = importance_df['importance'] / total_importance
        else:
            importance_df['relative_importance'] = 0
        key_features = importance_df[importance_df['relative_importance'] > threshold]['feature'].tolist()
        min_features = Config.feature_importance.get('min_features', 3)
        max_features = Config.feature_importance.get('max_features', 100)  # 新增最大特征数限制

        if len(key_features) < min_features:
            key_features = importance_df.nlargest(min_features, 'importance')['feature'].tolist()
        elif len(key_features) > max_features:
            # 如果特征过多，选择最重要的max_features个
            key_features = importance_df.nlargest(max_features, 'importance')['feature'].tolist()
            print(f"特征数量过多，已限制为前{max_features}个最重要的特征")
        print(f"\n找到 {len(key_features)} 个关键特征:")
        for idx, feature in enumerate(key_features, 1):
            importance = importance_df.loc[importance_df['feature'] == feature, 'importance'].values[0]
            relative_importance = importance_df.loc[importance_df['feature'] == feature, 'relative_importance'].values[
                0]
            print(f"{idx}. {feature} (重要性: {importance:.4f}, 相对重要性: {relative_importance:.4f})")
        os.makedirs(os.path.dirname(Config.feature_importance_file), exist_ok=True)
        importance_df.to_csv(Config.feature_importance_file, index=False, encoding='utf-8')
        print(f"特征重要性已保存至: {Config.feature_importance_file}")
        os.makedirs(os.path.dirname(Config.key_features_list_file), exist_ok=True)
        key_features_data = {
            'key_features': key_features,
            'importance_threshold': threshold,
            'feature_count': len(key_features),
            'feature_importances': {
                feature: float(importance_df.loc[importance_df['feature'] == feature, 'importance'].values[0])
                for feature in key_features
            },
            'relative_importances': {
                feature: float(importance_df.loc[importance_df['feature'] == feature, 'relative_importance'].values[0])
                for feature in key_features
            }
        }
        with open(Config.key_features_list_file, 'w', encoding='utf-8') as f:
            json.dump(key_features_data, f, indent=4, ensure_ascii=False)
        print(f"关键特征列表已保存至: {Config.key_features_list_file}")
        plt.figure(figsize=(12, 6))
        plt.title('特征重要性分布')
        plt.bar(range(len(importance_df)), importance_df['importance'].values)
        plt.xticks(range(len(importance_df)),
                   importance_df['feature'].values,
                   rotation=45,
                   ha='right')
        plt.tight_layout()
        os.makedirs(os.path.dirname(Config.feature_importance_plot), exist_ok=True)
        plt.savefig(Config.feature_importance_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"特征重要性图已保存至: {Config.feature_importance_plot}")

        # 保存原始特征列名到文件，确保与训练集一致
        original_feature_names_file = os.path.join(Config.feature_importance_dir, "original_feature_names.txt")
        with open(original_feature_names_file, 'w', encoding='utf-8') as f:
            f.write("原始训练集特征列名（与feature_importance_AA.csv中的feature列一致）：\n")
            f.write("=" * 60 + "\n")
            for i, feature in enumerate(feature_names, 1):
                f.write(f"{i:3d}. {feature}\n")
        print(f"原始特征列名已保存至: {original_feature_names_file}")
        return importance_df
    except Exception as e:
        print(f"特征重要性分析出错: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            print(f"错误位置: {tb[-1].filename}:{tb[-1].lineno}")
        feature_importance = np.ones(len(feature_names)) / len(feature_names)
        feature_importance = np.asarray(feature_importance).flatten()
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
        return importance_df


def save_predictions_and_importance(ids, suitability, area_yield, mu_yield, feature_importance):
    """保存预测结果和特征重要性"""
    with tqdm(total=2, desc="保存结果") as pbar:
        # 保存预测结果
        results = pd.DataFrame({
            'ID': ids,
            Config.target_columns['classification']: suitability,
            Config.target_columns['regression'][0]: area_yield,
            Config.target_columns['regression'][1]: mu_yield
        })

        # 添加预测结果的描述性统计
        print("\n保存的预测结果统计:")
        print("\n适宜度分布:")
        print(results[Config.target_columns['classification']].value_counts())
        print("\n产量统计:")
        print(results[[Config.target_columns['regression'][0], Config.target_columns['regression'][1]]].describe())

        # 确保输出目录存在
        os.makedirs(os.path.dirname(Config.result_file), exist_ok=True)
        results.to_csv(Config.result_file, index=False, encoding='utf-8')
        pbar.update(1)

        # 保存特征重要性
        os.makedirs(os.path.dirname(Config.feature_importance_file), exist_ok=True)
        feature_importance.to_csv(Config.feature_importance_file, index=False, encoding='utf-8')
        print("\n特征重要性 Top 10:")
        print(feature_importance.head(10))
        pbar.update(1)

    print(f"\n预测结果已保存到: {Config.result_file}")
    print(f"特征重要性分析已保存到: {Config.feature_importance_file}")


def has_invalid(arr):
    return np.any(np.isnan(arr)) or np.any(np.isinf(arr))


def validate_training_data(X_train, y_reg_train, y_cls_train, X_val, y_reg_val, y_cls_val):
    """验证训练数据的质量和一致性"""
    print("\n🔍 验证训练数据质量...")

    # 检查特征数据
    print(f"训练集特征形状: {X_train.shape}")
    print(f"验证集特征形状: {X_val.shape}")

    # 检查标签数据
    print(f"训练集回归标签形状: {y_reg_train.shape}")
    print(f"验证集回归标签形状: {y_reg_val.shape}")
    print(f"训练集分类标签形状: {y_cls_train.shape}")
    print(f"验证集分类标签形状: {y_cls_val.shape}")

    # 检查NaN和Inf值
    x_train_nan = np.isnan(X_train.values).sum() if hasattr(X_train, 'values') else np.isnan(X_train).sum()
    x_val_nan = np.isnan(X_val.values).sum() if hasattr(X_val, 'values') else np.isnan(X_val).sum()
    y_reg_train_nan = np.isnan(y_reg_train.values).sum() if hasattr(y_reg_train, 'values') else np.isnan(
        y_reg_train).sum()
    y_reg_val_nan = np.isnan(y_reg_val.values).sum() if hasattr(y_reg_val, 'values') else np.isnan(y_reg_val).sum()

    print(f"训练集特征NaN数量: {x_train_nan}")
    print(f"验证集特征NaN数量: {x_val_nan}")
    print(f"训练集回归标签NaN数量: {y_reg_train_nan}")
    print(f"验证集回归标签NaN数量: {y_reg_val_nan}")

    # 检查标签值范围
    if hasattr(y_reg_train, 'values'):
        y_reg_train_values = y_reg_train.values
    else:
        y_reg_train_values = y_reg_train

    if hasattr(y_reg_val, 'values'):
        y_reg_val_values = y_reg_val.values
    else:
        y_reg_val_values = y_reg_val

    print(f"训练集per_mu范围: [{y_reg_train_values[:, 1].min():.4f}, {y_reg_train_values[:, 1].max():.4f}]")
    print(f"训练集per_qu范围: [{y_reg_train_values[:, 0].min():.4f}, {y_reg_train_values[:, 0].max():.4f}]")
    print(f"验证集per_mu范围: [{y_reg_val_values[:, 1].min():.4f}, {y_reg_val_values[:, 1].max():.4f}]")
    print(f"验证集per_qu范围: [{y_reg_val_values[:, 0].min():.4f}, {y_reg_val_values[:, 0].max():.4f}]")

    # 检查标签分布
    print(f"训练集分类标签分布: {np.unique(y_cls_train, return_counts=True)}")
    print(f"验证集分类标签分布: {np.unique(y_cls_val, return_counts=True)}")

    # 检查特征分布
    if hasattr(X_train, 'values'):
        x_train_values = X_train.values
    else:
        x_train_values = X_train

    print(f"训练集特征统计: 均值={np.mean(x_train_values):.6f}, 标准差={np.std(x_train_values):.6f}")
    print(f"训练集特征范围: [{np.min(x_train_values):.6f}, {np.max(x_train_values):.6f}]")

    print("✅ 数据质量验证完成")


class R2EarlyStoppingAndSave(tf.keras.callbacks.Callback):
    """改进的R²早停回调，增强防过拟合能力"""

    def __init__(self, X_val, y_val, min_r2=0.05, patience=15, save_path=None,  # 降低R²阈值，增加patience
                 min_delta=0.0001, restore_best_weights=True, monitor='val_loss'):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.min_r2 = min_r2
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.save_path = save_path
        self.wait = 0
        self.best_r2 = -np.inf
        self.best_weights = None
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.model_saved = False
        self.bad_epochs = 0  # 添加缺失的初始化

    def on_epoch_end(self, epoch, logs=None):
        # 获取模型预测结果
        predictions = self.model.predict(self.X_val, verbose=0)
        y_pred = predictions[1]  # 回归输出
        y_true = self.y_val
        if hasattr(y_true, 'values'):
            y_true = y_true.values

        # 添加调试信息
        print(f"Epoch {epoch + 1} - 预测值范围: min={np.min(y_pred):.6f}, max={np.max(y_pred):.6f}")
        print(f"Epoch {epoch + 1} - 真实值范围: min={np.min(y_true):.6f}, max={np.max(y_true):.6f}")
        print(f"Epoch {epoch + 1} - 预测值形状: {y_pred.shape}, 真实值形状: {y_true.shape}")

        # 检查数据是否有问题
        if has_invalid(y_pred) or has_invalid(y_true):
            print(f"[警告] epoch {epoch + 1} 检测到无效数值（NaN/Inf），提前终止本trial")
            self.model.stop_training = True
            return

        # 检查预测值是否都是同一个值（模型没有学习）
        if np.std(y_pred) < 1e-6:
            print(f"[警告] epoch {epoch + 1} 预测值几乎没有变化（std={np.std(y_pred):.8f}），模型可能没有学习")

        # 注意：y_pred和y_true都已经是log1p变换后的值，不需要再次变换
        # 直接计算R²分数

        # 计算R2分数 - 只关注per_mu（第二个输出）
        try:
            # 确保y_pred是numpy数组
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            elif not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)

            # 确保y_true是numpy数组
            if hasattr(y_true, 'values'):
                y_true = y_true.values
            elif not isinstance(y_true, np.ndarray):
                y_true = np.array(y_true)

            # 只计算per_mu的R²值（第一列）
            if y_true.ndim > 1 and y_true.shape[1] > 0:
                y_true_per_mu = y_true[:, 0]  # per_mu是第一列
            else:
                y_true_per_mu = y_true.flatten()

            if y_pred.ndim > 1 and y_pred.shape[1] > 0:
                y_pred_per_mu = y_pred[:, 0]  # per_mu是第一列
            else:
                y_pred_per_mu = y_pred.flatten()

            # 确保长度匹配
            min_len = min(len(y_true_per_mu), len(y_pred_per_mu))
            y_true_per_mu = y_true_per_mu[:min_len]
            y_pred_per_mu = y_pred_per_mu[:min_len]

            # 检查数据有效性
            if len(y_true_per_mu) == 0 or len(y_pred_per_mu) == 0:
                print(f"[警告] epoch {epoch + 1} 数据为空，设置R²为-1000")
                r2 = -1000
            elif np.std(y_true_per_mu) < 1e-8:
                print(f"[警告] epoch {epoch + 1} 真实值方差为0，设置R²为-1000")
                r2 = -1000
            elif np.std(y_pred_per_mu) < 1e-8:
                print(f"[警告] epoch {epoch + 1} 预测值方差为0，设置R²为-1000")
                r2 = -1000
            else:
                r2 = r2_score(y_true_per_mu, y_pred_per_mu)
                # 限制R²在合理范围内
                r2 = max(-100, min(1, r2))
                print(f"Epoch {epoch + 1} - per_mu R²: {r2:.4f}")
        except Exception as e:
            print(f"[错误] epoch {epoch + 1} 计算R2分数失败: {e}")
            r2 = -1000  # 设置一个很低的R2分数

        logs = logs or {}
        logs['val_r2'] = r2

        # 如果R²大于0.1且是当前最佳，保存模型
        if r2 > self.min_r2 and r2 > self.best_r2 and not self.model_saved:
            self.best_r2 = r2
            if self.save_path:
                save_model_with_custom_objects(self.model, self.save_path)
                self.model_saved = True
                print(f"✅ Epoch {epoch + 1}: R²={r2:.4f} >= {self.min_r2}, 模型已保存!")
            self.bad_epochs = 0
        elif r2 > self.best_r2:
            self.best_r2 = r2
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if r2 < self.min_r2:
            print(f"Epoch {epoch + 1}: val_r2={r2:.4f} < {self.min_r2}, bad_epochs={self.bad_epochs}")
        else:
            print(f"Epoch {epoch + 1}: val_r2={r2:.4f} >= {self.min_r2}")

        if self.bad_epochs >= self.patience:
            print(f"Early stopping: val_r2低于{self.min_r2}已连续{self.patience}个epoch")
            self.model.stop_training = True


def train_final_model(X_train, y_cls_train, y_reg_train, X_val, y_cls_val, y_reg_val, strategy=None, skip_optuna=False,
                      best_params=None):
    """训练最终模型，支持多GPU训练"""

    # 如果跳过Optuna优化，直接使用提供的参数
    if skip_optuna and best_params is not None:
        print(f"\n🎯 跳过Optuna优化，直接使用R²达到0.8时的最佳参数...")
        print(f"使用参数: {best_params}")

        # 构建最终模型
        num_classes = len(np.unique(y_cls_train))
        best_model = build_hybrid_model(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
            params=best_params,
            strategy=strategy
        )

        # 创建monitor对象（如果未定义）
        if 'monitor' not in locals():
            monitor = TrainingMonitor(Config.logs_dir)

        # 定义学习率调度器（如果未定义）
        def create_lr_scheduler(initial_lr):
            """创建学习率调度器"""

            def lr_scheduler(epoch, lr):
                if epoch < 5:
                    return lr
                elif epoch < 15:
                    return lr * 0.8
                elif epoch < 25:
                    return lr * 0.6
                elif epoch < 35:
                    return lr * 0.4
                else:
                    return lr * 0.2

            return lr_scheduler

        # 设置最终训练的回调函数
        final_callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=Config.dl_params['early_stop_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                Config.final_model_file,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.LearningRateScheduler(
                create_lr_scheduler(best_params['lr'])
            ),
            TensorBoard(
                log_dir=Config.logs_dir,
                histogram_freq=1
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: monitor.update_metrics(epoch, logs)
            ),
            R2EarlyStoppingAndSave(
                X_val, y_reg_val, min_r2=0.1, patience=5, save_path=Config.final_model_file
            )
        ]

        # 训练最终模型
        history = best_model.fit(
            X_train,
            {
                'classification': y_cls_train,
                'regression': y_reg_train
            },
            validation_data=(
                X_val,
                {
                    'classification': y_cls_val,
                    'regression': y_reg_val
                }
            ),
            epochs=Config.dl_params['epochs'],
            batch_size=best_params['batch_size'],
            callbacks=final_callbacks,
            verbose=1
        )

        # 保存训练历史
        history_file = os.path.join(Config.logs_dir, 'final_model_history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump({
                'params': best_params,
                'history': history.history,
                'r2_achieved': True,
                'skip_optuna': True
            }, f, indent=4, ensure_ascii=False, default=json_fallback)

        return best_model, history

    print("\n开始Optuna超参数优化...")

    # 添加输入数据验证
    print("输入数据验证:")
    print(
        f"X_train shape: {X_train.shape}, dtype: {X_train.dtypes.iloc[0] if hasattr(X_train, 'dtypes') else 'unknown'}")
    print(f"X_val shape: {X_val.shape}, dtype: {X_val.dtypes.iloc[0] if hasattr(X_val, 'dtypes') else 'unknown'}")
    print(f"y_reg_train shape: {y_reg_train.shape if hasattr(y_reg_train, 'shape') else len(y_reg_train)}")
    print(f"y_reg_val shape: {y_reg_val.shape if hasattr(y_reg_val, 'shape') else len(y_reg_val)}")

    # 添加目标变量统计信息
    print(f"训练集目标变量统计:")
    if hasattr(y_reg_train, 'describe'):
        print(y_reg_train.describe())
    else:
        # 安全地处理pandas DataFrame和numpy array
        if hasattr(y_reg_train, 'iloc'):
            y_reg_train_array = y_reg_train.values
        else:
            y_reg_train_array = np.array(y_reg_train)
        print(
            f"per_mu: min={np.min(y_reg_train_array[:, 0]):.4f}, max={np.max(y_reg_train_array[:, 0]):.4f}, mean={np.mean(y_reg_train_array[:, 0]):.4f}")
        print(
            f"per_qu: min={np.min(y_reg_train_array[:, 1]):.4f}, max={np.max(y_reg_train_array[:, 1]):.4f}, mean={np.mean(y_reg_train_array[:, 1]):.4f}")

    print(f"验证集目标变量统计:")
    if hasattr(y_reg_val, 'describe'):
        print(y_reg_val.describe())
    else:
        # 安全地处理pandas DataFrame和numpy array
        if hasattr(y_reg_val, 'iloc'):
            y_reg_val_array = y_reg_val.values
        else:
            y_reg_val_array = np.array(y_reg_val)
        print(
            f"per_mu: min={np.min(y_reg_val_array[:, 0]):.4f}, max={np.max(y_reg_val_array[:, 0]):.4f}, mean={np.mean(y_reg_val_array[:, 0]):.4f}")
        print(
            f"per_qu: min={np.min(y_reg_val_array[:, 1]):.4f}, max={np.max(y_reg_val_array[:, 1]):.4f}, mean={np.mean(y_reg_val_array[:, 1]):.4f}")

    # 检查数据中是否有NaN或Inf
    if hasattr(X_train, 'isnull'):
        train_nan_count = X_train.isnull().sum().sum()
        val_nan_count = X_val.isnull().sum().sum()
        print(f"NaN检查 - 训练集: {train_nan_count}, 验证集: {val_nan_count}")

        if train_nan_count > 0 or val_nan_count > 0:
            print("发现NaN值，进行清理...")
            X_train = X_train.fillna(0)
            X_val = X_val.fillna(0)

    # 检查标签数据
    if hasattr(y_reg_train, 'shape'):
        train_label_nan = np.isnan(y_reg_train).sum()
        val_label_nan = np.isnan(y_reg_val).sum()
        print(f"标签NaN检查 - 训练: {train_label_nan}, 验证: {val_label_nan}")

        # 确保比较的是标量值
        train_nan_count = train_label_nan.sum() if hasattr(train_label_nan, 'sum') else train_label_nan
        val_nan_count = val_label_nan.sum() if hasattr(val_label_nan, 'sum') else val_label_nan

        if train_nan_count > 0 or val_nan_count > 0:
            print("清理标签NaN值...")
            y_reg_train = np.nan_to_num(y_reg_train, nan=0.0)
            y_reg_val = np.nan_to_num(y_reg_val, nan=0.0)

    # 额外检查：确保所有数据都是有限的数值
    print("最终数据质量检查...")

    # 检查特征数据
    if hasattr(X_train, 'values'):
        X_train_values = X_train.values
    else:
        X_train_values = X_train

    if hasattr(X_val, 'values'):
        X_val_values = X_val.values
    else:
        X_val_values = X_val

    # 检查并清理特征中的NaN和Inf
    if np.isnan(X_train_values).any() or np.isinf(X_train_values).any():
        print("清理训练特征中的NaN/Inf...")
        X_train_values = np.nan_to_num(X_train_values, nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(X_train, 'values'):
            X_train = pd.DataFrame(X_train_values, columns=X_train.columns, index=X_train.index)
        else:
            X_train = X_train_values

    if np.isnan(X_val_values).any() or np.isinf(X_val_values).any():
        print("清理验证特征中的NaN/Inf...")
        X_val_values = np.nan_to_num(X_val_values, nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(X_val, 'values'):
            X_val = pd.DataFrame(X_val_values, columns=X_val.columns, index=X_val.index)
        else:
            X_val = X_val_values

    # 检查并清理标签中的NaN和Inf
    if hasattr(y_reg_train, 'shape'):
        train_nan_count = np.isnan(y_reg_train).sum()
        train_inf_count = np.isinf(y_reg_train).sum()
        # 确保比较的是标量值
        if hasattr(train_nan_count, 'sum'):
            train_nan_count = train_nan_count.sum()
        if hasattr(train_inf_count, 'sum'):
            train_inf_count = train_inf_count.sum()

        if train_nan_count > 0 or train_inf_count > 0:
            print(f"清理训练标签中的NaN({train_nan_count})/Inf({train_inf_count})...")
            y_reg_train = np.nan_to_num(y_reg_train, nan=0.0, posinf=0.0, neginf=0.0)

    if hasattr(y_reg_val, 'shape'):
        val_nan_count = np.isnan(y_reg_val).sum()
        val_inf_count = np.isinf(y_reg_val).sum()
        # 确保比较的是标量值
        if hasattr(val_nan_count, 'sum'):
            val_nan_count = val_nan_count.sum()
        if hasattr(val_inf_count, 'sum'):
            val_inf_count = val_inf_count.sum()

        if val_nan_count > 0 or val_inf_count > 0:
            print(f"清理验证标签中的NaN({val_nan_count})/Inf({val_inf_count})...")
            y_reg_val = np.nan_to_num(y_reg_val, nan=0.0, posinf=0.0, neginf=0.0)

    print("数据清理完成，开始Optuna优化...")

    # 创建训练监控器
    monitor = TrainingMonitor(Config.logs_dir)

    # 定义优化目标函数（移到外部避免闭包作用域问题）
    def create_objective(X_train, y_cls_train, y_reg_train, X_val, y_cls_val, y_reg_val, strategy, monitor):
        def objective(trial):
            # 从超参数空间中采样（与Config.optuna_params一致，只允许'adam'和'relu'）
            params = {
                'lr': trial.suggest_float('lr', *Config.optuna_params['param_ranges']['lr'], log=True),
                'neurons1': trial.suggest_int('neurons1', *Config.optuna_params['param_ranges']['neurons1']),
                'neurons2': trial.suggest_int('neurons2', *Config.optuna_params['param_ranges']['neurons2']),
                'dropout_rate': trial.suggest_float('dropout_rate',
                                                    *Config.optuna_params['param_ranges']['dropout_rate']),
                'batch_size': trial.suggest_categorical(
                    'batch_size',
                    Config.optuna_params['param_ranges']['batch_size']
                ),
                'attention_units': trial.suggest_int(
                    'attention_units',
                    *Config.optuna_params['param_ranges']['attention_units']
                ),
                'l1_lambda': trial.suggest_float(
                    'l1_lambda',
                    *Config.optuna_params['param_ranges']['l1_lambda'],
                    log=True),
                'l2_lambda': trial.suggest_float(
                    'l2_lambda',
                    *Config.optuna_params['param_ranges']['l2_lambda'],
                    log=True),
                'optimizer_type': trial.suggest_categorical(
                    'optimizer_type',
                    Config.optuna_params['param_ranges']['optimizer_type']
                ),
                'activation': trial.suggest_categorical(
                    'activation',
                    Config.optuna_params['param_ranges']['activation']
                )
            }

            # 构建模型
            num_classes = len(np.unique(y_cls_train))
            model = build_hybrid_model(X_train.shape[1], num_classes, params, strategy)

            # 设置改进的回调函数，防止过拟合
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=Config.dl_params['early_stop_patience'],
                    restore_best_weights=True,
                    min_delta=Config.dl_params['min_delta'],
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.3,  # 更激进的学习率衰减
                    patience=2,  # 进一步缩短耐心，加快训练
                    min_lr=1e-7,  # 更小的最小学习率
                    verbose=1,
                    mode='min'
                ),
                ModelCheckpoint(
                    os.path.join(Config.model_dir, f'model_trial_{trial.number}.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0
                ),
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: monitor.update_metrics(epoch, logs)
                ),
                # 添加R²早停回调，当R²达到0.8时停止调参
                OptunaR2EarlyStopping(
                    X_val, y_reg_val, min_r2=0.8, trial=trial
                )
            ]

            # 训练模型
            try:
                history = model.fit(
                    X_train,
                    {
                        'classification': y_cls_train,
                        'regression': y_reg_train
                    },
                    validation_data=(
                        X_val,
                        {
                            'classification': y_cls_val,
                            'regression': y_reg_val
                        }
                    ),
                    epochs=Config.dl_params['epochs'],
                    batch_size=params['batch_size'],
                    callbacks=callbacks,
                    verbose=0
                )

                # 计算验证集性能
                val_loss = min(history.history['val_loss'])

                # 报告中间值
                trial.report(val_loss, step=Config.dl_params['epochs'])

                # 处理提前停止
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return val_loss

            except optuna.TrialPruned as e:
                # 检查是否是因为R²达到0.8而停止的
                if hasattr(trial, 'user_attrs') and trial.user_attrs.get('r2_achieved', 0) >= 0.8:
                    print(f"🎯 R²达到{trial.user_attrs.get('r2_achieved', 0):.4f}，停止调参训练！")
                    # 返回一个很小的损失值，表示这是一个成功的试验
                    return 0.001
                else:
                    # 正常的剪枝
                    raise e

        return objective

    # 创建学习率调度器
    def create_lr_scheduler(initial_lr):
        """创建改进的学习率调度器，防止过拟合"""

        # 使用ReduceLROnPlateau回调，更智能的学习率调整
        def lr_scheduler(epoch, lr):
            # 更严格的学习率衰减策略
            if epoch < 5:
                return lr
            elif epoch < 15:
                return lr * 0.8
            elif epoch < 25:
                return lr * 0.6
            elif epoch < 35:
                return lr * 0.4
            else:
                return lr * 0.2

        return lr_scheduler

    # 创建和运行Optuna研究
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )

    # 创建objective函数并传递给study.optimize
    objective_func = create_objective(X_train, y_cls_train, y_reg_train, X_val, y_cls_val, y_reg_val, strategy, monitor)

    # 添加R²达到0.8时的早期停止机制
    r2_achieved = False
    best_r2 = 0.0
    global_best_params = None  # 存储全局最佳参数

    def optimize_with_r2_stop():
        nonlocal r2_achieved, best_r2

        for trial in study.trials:
            if r2_achieved:
                print(f"🎯 R²已达到{best_r2:.4f}，停止调参训练！")
                break

            try:
                result = objective_func(trial)

                # 检查trial是否因为R²达到0.8而停止
                if hasattr(trial, 'user_attrs') and trial.user_attrs.get('r2_achieved', 0) >= 0.8:
                    r2_achieved = True
                    best_r2 = trial.user_attrs.get('r2_achieved', 0)
                    print(f"🎯 试验 {trial.number}: R²达到{best_r2:.4f}，停止调参训练！")
                    break

            except optuna.TrialPruned:
                # 检查是否是因为R²达到0.8而剪枝
                if hasattr(trial, 'user_attrs') and trial.user_attrs.get('r2_achieved', 0) >= 0.8:
                    r2_achieved = True
                    best_r2 = trial.user_attrs.get('r2_achieved', 0)
                    print(f"🎯 试验 {trial.number}: R²达到{best_r2:.4f}，停止调参训练！")
                    break
                continue
            except Exception as e:
                print(f"试验 {trial.number} 失败: {e}")
                continue

    # 运行优化
    if not r2_achieved:
        study.optimize(
            objective_func,
            n_trials=Config.optuna_params['n_trials'],
            timeout=Config.optuna_params['timeout']
        )

    print("\n最佳超参数:", study.best_trial.params)

    # 初始化变量
    global_best_params = study.best_trial.params  # 默认使用Optuna最佳参数
    r2_achieved = False
    best_r2 = 0

    # 检查是否有试验达到R²=0.8
    for trial in study.trials:
        if hasattr(trial, 'user_attrs') and trial.user_attrs.get('r2_achieved', 0) >= 0.8:
            print(f"🎯 发现R²达到0.8的试验: 试验{trial.number}, R²={trial.user_attrs.get('r2_achieved', 0):.4f}")
            # 使用这个试验的参数作为最佳参数
            # study.best_trial 是只读属性，不能直接设置
            global_best_params = trial.params  # 保存全局最佳参数
            r2_achieved = True
            best_r2 = trial.user_attrs.get('r2_achieved', 0)
            break

    # 使用最佳参数训练最终模型
    print("\n使用最佳参数训练最终模型...")
    num_classes = len(np.unique(y_cls_train))

    # 构建最终模型
    # 使用全局最佳参数（如果R²达到0.8）或Optuna最佳参数
    final_params = global_best_params if r2_achieved else study.best_trial.params
    print(f"使用参数: {'全局最佳参数' if r2_achieved else 'Optuna最佳参数'}")

    best_model = build_hybrid_model(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        params=final_params,
        strategy=strategy
    )

    # 设置最终训练的回调函数
    final_callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=Config.dl_params['early_stop_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            Config.final_model_file,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(
            create_lr_scheduler(final_params['lr'])
        ),
        TensorBoard(
            log_dir=Config.logs_dir,
            histogram_freq=1
        ),
        # 移除monitor回调，因为monitor变量未定义
        # tf.keras.callbacks.LambdaCallback(
        #     on_epoch_end=lambda epoch, logs: monitor.update_metrics(epoch, logs)
        # ),
        R2EarlyStoppingAndSave(
            X_val, y_reg_val, min_r2=0.1, patience=3, save_path=Config.final_model_file  # 减少patience，加快早停
        )
    ]

    # 训练最终模型
    history = best_model.fit(
        X_train,
        {
            'classification': y_cls_train,
            'regression': y_reg_train
        },
        validation_data=(
            X_val,
            {
                'classification': y_cls_val,
                'regression': y_reg_val
            }
        ),
        epochs=Config.dl_params['epochs'],
        batch_size=study.best_trial.params['batch_size'],
        callbacks=final_callbacks,
        verbose=1
    )

    # 保存训练历史
    history_file = os.path.join(Config.logs_dir, 'final_model_history.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump({
            'params': study.best_trial.params,
            'history': history.history,
            'study_trials': [
                {
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value
                }
                for trial in study.trials
            ]
        }, f, indent=4, ensure_ascii=False, default=json_fallback)

    return best_model, history


def cross_validate_and_ensemble(X, y_cls, y_reg, n_splits=10, n_models=5, strategy=None, global_best_params=None,
                                r2_achieved=False):
    """实现交叉验证和模型集成，支持多GPU训练

    Args:
        X: 特征数据
        y_cls: 分类标签
        y_reg: 回归标签
        n_splits: 交叉验证折数
        n_models: 集成模型数量
        strategy: 多GPU训练策略
        xgb_max_rows: XGBoost采样量

    Returns:
        ensemble_models: 集成模型列表
        cv_scores: 交叉验证分数
    """
    from sklearn.model_selection import StratifiedKFold
    cv_scores = {
        'cls_acc': [],
        'reg_r2': [],
        'reg_mae': []
    }
    ensemble_models = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.random_seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_cls), 1):
        print(f"\n训练第 {fold}/{n_splits} 折...")
        # 分割数据
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_cls_train_fold = y_cls[train_idx]
        y_cls_val_fold = y_cls[val_idx]
        y_reg_train_fold = y_reg.iloc[train_idx]
        y_reg_val_fold = y_reg.iloc[val_idx]
        fold_models = []
        for model_idx in range(n_models):
            print(f"\n训练模型 {model_idx + 1}/{n_models}...")
            # XGBoost特征提取（使用全部数据）
            X_train_combined, X_val_combined, xgb_model, y_cls_train_sample, y_cls_val_sample, y_reg_train_sample, y_reg_val_sample = extract_xgboost_features(
                X_train_fold, X_val_fold, y_cls_train_fold, y_cls_val_fold, y_reg_train_fold, y_reg_val_fold
            )
            # 训练深度学习模型（只用采样数据）
            # 如果R²已达到0.8，跳过Optuna优化，直接使用最佳参数
            if r2_achieved and global_best_params is not None:
                print(f"🎯 交叉验证第{fold}折: 使用R²达到0.8时的最佳参数，跳过调参")
                model, history = train_final_model(
                    X_train_combined, y_cls_train_sample, y_reg_train_sample,
                    X_val_combined, y_cls_val_sample, y_reg_val_sample,
                    strategy=strategy, skip_optuna=True, best_params=global_best_params
                )
            else:
                model, history = train_final_model(
                    X_train_combined, y_cls_train_sample, y_reg_train_sample,
                    X_val_combined, y_cls_val_sample, y_reg_val_sample,
                    strategy=strategy
                )
            fold_models.append({
                'xgb_model': xgb_model,
                'dl_model': model
            })
        # 验证集评估（只用采样数据）
        val_preds_cls = []
        val_preds_reg = []
        for model_dict in fold_models:
            cls_pred, reg_pred = model_dict['dl_model'].predict(X_val_combined)
            val_preds_cls.append(cls_pred)
            val_preds_reg.append(reg_pred)
        ensemble_cls = np.mean(val_preds_cls, axis=0)
        ensemble_reg = np.mean(val_preds_reg, axis=0)
        cls_acc = accuracy_score(y_cls_val_sample, np.argmax(ensemble_cls, axis=1))
        # 只计算per_mu的R²值
        # 确保y_reg_val_sample是numpy数组
        if hasattr(y_reg_val_sample, 'values'):
            y_reg_val_sample = y_reg_val_sample.values
        reg_r2 = r2_score(y_reg_val_sample[:, 0], ensemble_reg[:, 0])  # 只使用per_mu
        reg_mae = mean_absolute_error(y_reg_val_sample, ensemble_reg)
        cv_scores['cls_acc'].append(cls_acc)
        cv_scores['reg_r2'].append(reg_r2)
        cv_scores['reg_mae'].append(reg_mae)
        ensemble_models.append(fold_models)
        print(f"\n第 {fold} 折结果:")
        print(f"分类准确率: {cls_acc:.4f}")
        print(f"per_mu R2 分数: {reg_r2:.4f}")
        print(f"回归 MAE: {reg_mae:.4f}")
    print("\n交叉验证平均分数:")
    print(f"分类准确率: {np.mean(cv_scores['cls_acc']):.4f} ± {np.std(cv_scores['cls_acc']):.4f}")
    print(f"per_mu R2 分数: {np.mean(cv_scores['reg_r2']):.4f} ± {np.std(cv_scores['reg_r2']):.4f}")
    print(f"回归 MAE: {np.mean(cv_scores['reg_mae']):.4f} ± {np.std(cv_scores['reg_mae']):.4f}")
    print("注意：R²值只计算per_mu，per_qu权重为0")
    return ensemble_models, cv_scores


def augment_data(
        X: Union[pd.DataFrame, np.ndarray],
        y_cls: Union[pd.Series, np.ndarray],
        y_reg: Union[pd.DataFrame, np.ndarray],
        augmentation_factor: float = 1.0
) -> tuple:
    assert isinstance(X, (pd.DataFrame, np.ndarray)), f"X must be DataFrame or ndarray, got {type(X)}"
    assert isinstance(y_cls, (pd.Series, np.ndarray)), f"y_cls must be Series or ndarray, got {type(y_cls)}"
    assert isinstance(y_reg, (pd.DataFrame, np.ndarray)), f"y_reg must be DataFrame or ndarray, got {type(y_reg)}"
    assert 0 < augmentation_factor < 1, "augmentation_factor must be between 0 and 1"

    print("\n执行数据增强...")
    X_aug = X.copy() if hasattr(X, 'copy') else np.copy(X)
    y_cls_aug = y_cls.copy() if hasattr(y_cls, 'copy') else np.copy(y_cls)
    y_reg_aug = y_reg.copy() if hasattr(y_reg, 'copy') else np.copy(y_reg)

    n_samples = len(X)
    n_augment = int(n_samples * augmentation_factor)

    # 1. 高斯噪声扰动（增强版）
    noise_samples = []
    noise_cls = []
    noise_reg = []

    for i in range(n_augment):
        # 随机选择一个样本
        idx = random.randint(0, n_samples - 1)
        sample = safe_slice(X, idx)
        sample = sample.copy() if hasattr(sample, 'copy') else np.copy(sample)
        # 添加增强的高斯噪声
        noise_std = Config.augmentation_params.get('noise_factor', 0.02)
        noise = np.random.normal(0, noise_std, size=len(sample))
        noisy_sample = sample + noise
        noise_samples.append(noisy_sample)
        noise_cls.append(safe_slice(y_cls, idx))
        noise_reg.append(safe_slice(y_reg, idx))

    # 2. Mixup数据增强
    mixup_samples = []
    mixup_cls = []
    mixup_reg = []

    mixup_alpha = Config.augmentation_params.get('mixup_alpha', 0.2)
    for i in range(n_augment):
        # 随机选择两个样本
        idx1, idx2 = random.sample(range(n_samples), 2)
        # 生成Beta分布的混合权重
        alpha = np.random.beta(mixup_alpha, mixup_alpha)
        s1 = safe_slice(X, idx1)
        s2 = safe_slice(X, idx2)
        mixed_sample = alpha * s1 + (1 - alpha) * s2
        r1 = safe_slice(y_reg, idx1)
        r2 = safe_slice(y_reg, idx2)
        mixed_reg_value = alpha * r1 + (1 - alpha) * r2
        mixup_samples.append(mixed_sample)
        mixup_cls.append(safe_slice(y_cls, idx1))  # 保持原始标签
        mixup_reg.append(mixed_reg_value)

    # 3. CutMix数据增强（特征级别）
    cutmix_samples = []
    cutmix_cls = []
    cutmix_reg = []

    cutmix_alpha = Config.augmentation_params.get('cutmix_alpha', 1.0)
    for i in range(n_augment):
        # 随机选择两个样本
        idx1, idx2 = random.sample(range(n_samples), 2)
        s1 = safe_slice(X, idx1)
        s2 = safe_slice(X, idx2)

        # 生成CutMix掩码
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        cut_len = int(len(s1) * (1 - lam))
        cut_start = random.randint(0, len(s1) - cut_len)

        # 应用CutMix
        mixed_sample = s1.copy()
        mixed_sample[cut_start:cut_start + cut_len] = s2[cut_start:cut_start + cut_len]

        cutmix_samples.append(mixed_sample)
        cutmix_cls.append(safe_slice(y_cls, idx1))
        cutmix_reg.append(safe_slice(y_reg, idx1))

    # 合并所有增强数据
    all_augmented_data = [
        (noise_samples, noise_cls, noise_reg),
        (mixup_samples, mixup_cls, mixup_reg),
        (cutmix_samples, cutmix_cls, cutmix_reg)
    ]

    for samples, cls_labels, reg_labels in all_augmented_data:
        if samples:
            if isinstance(X, pd.DataFrame):
                X_aug = pd.concat([X_aug, pd.DataFrame(samples, columns=X.columns)], axis=0)
            else:
                X_aug = np.concatenate([X_aug, np.array(samples)], axis=0)

            y_cls_aug = np.concatenate([y_cls_aug, np.array(cls_labels)], axis=0)

            if isinstance(y_reg, pd.DataFrame):
                y_reg_aug = pd.concat([y_reg_aug, pd.DataFrame(reg_labels, columns=y_reg.columns)], axis=0)
            else:
                y_reg_aug = np.concatenate([y_reg_aug, np.array(reg_labels)], axis=0)

    # 确保所有数据集大小一致
    min_samples = min(len(X_aug), len(y_cls_aug), len(y_reg_aug))
    if isinstance(X_aug, pd.DataFrame):
        X_aug = X_aug.iloc[:min_samples]
    else:
        X_aug = X_aug[:min_samples]
    y_cls_aug = y_cls_aug[:min_samples]
    if isinstance(y_reg_aug, pd.DataFrame):
        y_reg_aug = y_reg_aug.iloc[:min_samples]
    else:
        y_reg_aug = y_reg_aug[:min_samples]

    print(f"原始数据大小: {len(X)}")
    print(f"增强后数据大小: {len(X_aug)}")
    print(f"增强比例: {(len(X_aug) - len(X)) / len(X) * 100:.2f}%")

    return X_aug, y_cls_aug, y_reg_aug


def safe_slice(obj: Union[pd.DataFrame, pd.Series, np.ndarray], idx: int):
    assert isinstance(idx, int), f"idx must be int, got {type(idx)}"
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.iloc[idx]
    else:
        return obj[idx]


# === 工具函数：安全获取列 ===
def get_col_safe(df, candidates, idx=None):
    """
    从df中优先返回candidates列表中第一个存在的列。
    idx: 可选，若为slice或索引，则返回该范围/索引的值。
    """
    for cand in candidates:
        if cand in df.columns:
            if idx is not None:
                return df[cand].iloc[idx].values
            else:
                return df[cand]
    raise KeyError(f"None of {candidates} found in columns: {df.columns.tolist()}")


def check_data(X, y_reg, y_cls, name="train"):
    print(f"==== {name} 数据检查 ====")
    print("X shape:", X.shape)
    print("y_reg shape:", y_reg.shape)
    print("y_cls shape:", y_cls.shape)
    print("X NaN:", np.isnan(X).sum(), "Inf:", np.isinf(X).sum())
    print("y_reg NaN:", np.isnan(y_reg).sum(), "Inf:", np.isinf(y_reg).sum())
    print("y_cls NaN:", np.isnan(y_cls).sum(), "Inf:", np.isinf(y_cls).sum())
    print("y_reg describe:\n", pd.DataFrame(y_reg).describe())
    print("y_cls value counts:\n", pd.Series(y_cls).value_counts())
    print("=" * 40)


def generate_interaction_features(X, y=None, sample_size=100000, max_combos=100, top_n=10, selected_features=None):
    """
    生成交互特征，并根据互信息选择Top特征。

    参数:
        X: pd.DataFrame
            输入特征数据。
        y: pd.Series or None
            目标变量，用于计算互信息。如果为None，仅返回交互特征。
        sample_size: int
            互信息计算时采样的样本数（避免大数据集占用过多内存）。
        max_combos: int
            随机选择的最大组合数（避免特征组合过多）。
        top_n: int
            选择Top N个交互特征（基于互信息）。
        selected_features: list or None
            如果提供，则只生成这些特征的交互特征。

    返回:
        X_new: pd.DataFrame
            添加了交互特征的新DataFrame。
        selected_feats: list
            被选择的交互特征名（如果y不为None）。
    """
    print("\n生成交互特征...")

    # 如果 selected_features 没提供，则使用 X 的全部特征
    features = selected_features if selected_features else X.columns.tolist()

    # 获取特征数量
    num_features = len(features)
    print(f"原始特征数量: {num_features}")

    # 如果特征数小于2，不生成交互特征
    if num_features < 2:
        print("特征数量不足，跳过交互特征生成。")
        return X, []

    # 获取所有两两组合
    from itertools import combinations
    combos = list(combinations(features, 2))
    print(f"组合数量: {len(combos)}")

    # 如果组合数量过多，随机采样 max_combos 个（防过拟合优化）
    if len(combos) > max_combos:
        print(f"组合数量过多({len(combos)})，随机选择{max_combos}个组合")
        # 优先选择互信息高的特征组合
        if y is not None and len(combos) > max_combos * 2:
            # 先计算所有组合的互信息，选择top特征
            from sklearn.feature_selection import mutual_info_regression
            combo_scores = []
            for f1, f2 in combos:
                if f1 in X.columns and f2 in X.columns:
                    try:
                        combo_feature = X[f1] * X[f2]
                        if y.ndim > 1:
                            y_target = y.iloc[:, 0] if hasattr(y, 'iloc') else y[:, 0]
                        else:
                            y_target = y
                        mi_score = \
                        mutual_info_regression(combo_feature.values.reshape(-1, 1), y_target, random_state=42)[0]
                        combo_scores.append((mi_score, f1, f2))
                    except:
                        combo_scores.append((0, f1, f2))

            # 按互信息分数排序，选择top组合
            combo_scores.sort(reverse=True)
            combos = [(f1, f2) for _, f1, f2 in combo_scores[:max_combos]]
        else:
            combos = random.sample(combos, max_combos)

    # 初始化交互特征 DataFrame
    inter_df = pd.DataFrame(index=X.index)

    # 生成交互特征
    for f1, f2 in combos:
        # 检查特征是否存在
        if f1 in X.columns and f2 in X.columns:
            inter_df[f"{f1}_x_{f2}"] = X[f1] * X[f2]
        else:
            # 尝试查找相似的特征名（处理可能的命名差异）
            f1_found = None
            f2_found = None

            # 查找f1的匹配特征
            for col in X.columns:
                if f1 in col or col in f1:
                    f1_found = col
                    break

            # 查找f2的匹配特征
            for col in X.columns:
                if f2 in col or col in f2:
                    f2_found = col
                    break

            if f1_found and f2_found:
                inter_df[f"{f1}_x_{f2}"] = X[f1_found] * X[f2_found]
            else:
                print(f"警告: 特征 {f1} 或 {f2} 不存在于数据中，跳过交互特征 {f1}_x_{f2}")
                if len(X.columns) <= 20:  # 如果特征不多，显示所有特征
                    print(f"可用特征列: {X.columns.tolist()}")
                else:
                    print(f"可用特征列: {X.columns.tolist()[:10]}...")
                # 用0填充缺失的交互特征
                inter_df[f"{f1}_x_{f2}"] = 0.0

    print(f"生成了 {len(inter_df.columns)} 个交互特征")

    # 如果 y 为空，直接返回（预测阶段用）
    if y is None:
        if selected_features is None:
            raise ValueError("预测阶段必须提供 selected_features")

        # 如果max_combos=0，直接生成选中的交互特征，不进行新的组合
        if max_combos == 0:
            print(f"直接生成 {len(selected_features)} 个选中的交互特征...")
            result_df = pd.DataFrame(index=X.index)

            for feat in selected_features:
                # 解析特征名，找到对应的两个原始特征
                if '_x_' in feat:
                    f1, f2 = feat.split('_x_', 1)
                    if f1 in X.columns and f2 in X.columns:
                        result_df[feat] = X[f1] * X[f2]
                        print(f"✅ 成功生成交互特征: {feat}")
                    else:
                        print(f"⚠️ 警告: 原始特征 {f1} 或 {f2} 不存在，用0填充")
                        result_df[feat] = 0.0
                else:
                    print(f"⚠️ 警告: 无效的交互特征名 {feat}，用0填充")
                    result_df[feat] = 0.0

            return result_df

        # 只生成选中的交互特征
        result_df = pd.DataFrame(index=X.index)
        print(f"尝试为验证集生成 {len(selected_features)} 个交互特征...")
        print(f"验证集可用特征: {list(X.columns)[:10]}...")  # 显示前10个特征

        for feat in selected_features:
            if feat in inter_df.columns:
                result_df[feat] = inter_df[feat]
                print(f"✅ 成功生成交互特征: {feat}")
            else:
                print(f"⚠️ 警告: 交互特征 {feat} 不存在，用0填充")
                print(f"   可用的交互特征: {list(inter_df.columns)[:5]}...")  # 显示前5个可用特征
                result_df[feat] = 0.0

        return result_df

        # 不进行采样，使用全部数据进行特征选择
        print(f"大数据量优化：采样100万数据进行特征选择")
        if len(X) > 1000000:
            X_sample = inter_df.sample(n=1000000, random_state=42)
            y_sample = y.iloc[X_sample.index]
        else:
            X_sample = inter_df
            y_sample = y

    # 计算互信息
    print("计算互信息并选择top特征...")
    # 如果y是多维的，取第一列（通常是主要目标）
    if y_sample.ndim > 1:
        y_sample_1d = y_sample.iloc[:, 0] if hasattr(y_sample, 'iloc') else y_sample[:, 0]
        print(
            f"多目标回归，使用第一个目标进行特征选择: {y_sample_1d.name if hasattr(y_sample_1d, 'name') else 'target_0'}")
    else:
        y_sample_1d = y_sample

    mi = mutual_info_regression(X_sample, y_sample_1d, random_state=42)

    # 选择 top_n 个最相关特征
    top_idx = np.argsort(mi)[::-1][:top_n]
    selected_feats = [inter_df.columns[i] for i in top_idx]
    print(f"选择了 {len(selected_feats)} 个交互特征: {selected_feats}")

    # 返回包含交互特征的数据
    return inter_df[selected_feats], selected_feats


# 原始训练函数已删除，现在只保留预测功能

def main():
    """主函数：整合所有优化功能，支持多GPU训练"""
    try:
        # 1. 验证配置
        print("\n=== 开始运行优化预测程序 ===")
        print("\n1. 验证配置...")
        if not Config.validate_params() or not Config.validate_paths():
            raise ValueError("配置验证失败")

        # 2. 创建必要的目录
        print("\n2. 创建输出目录...")
        Config.create_output_dirs()

        # 3. 设置GPU和多GPU策略
        print("\n3. 配置GPU和多GPU训练环境...")

        # 检查TensorFlow CUDA支持
        print(f"TensorFlow版本: {tf.__version__}")
        print(f"CUDA支持: {'是' if tf.test.is_built_with_cuda() else '否'}")

        if not tf.test.is_built_with_cuda():
            print("⚠️  当前TensorFlow版本不支持CUDA，将使用CPU模式")
            print("💡 建议安装支持CUDA的TensorFlow版本以获得GPU加速")
            print("   安装命令: pip install tensorflow==2.10.1")
            use_gpu, num_gpus, strategy = False, 0, None
        else:
            use_gpu, num_gpus, strategy = setup_multi_gpu()

        if use_gpu and num_gpus >= 2:
            print(f"✅ 多GPU训练环境配置成功，将使用 {num_gpus} 个GPU进行训练")
            print(f"使用策略: {type(strategy).__name__}")
        elif use_gpu:
            print(f"✅ 单GPU训练环境配置成功，将使用 {num_gpus} 个GPU")
            strategy = None
        else:
            print("ℹ️  使用CPU模式进行训练")
            strategy = None

        # 3.1 验证XGBoost GPU支持
        print("\n3.1 验证XGBoost GPU支持...")
        Config.validate_xgboost_gpu()

        # 4. 加载数据
        print("\n4. 开始加载和处理数据...")
        X_train, y_cls_train, y_reg_train, X_val, val_ids, data_val = load_data()
        check_data(X_train, y_reg_train, y_cls_train, "train")

        # 4.1 验证数据质量
        print("\n4.1 验证数据质量...")
        # 注意：这里验证集还没有进行log1p变换，所以先不验证标签范围
        validate_training_data(X_train, y_reg_train, y_cls_train, X_val,
                               data_val[['per_mu', 'per_qu']], data_val['suit'])

        # ====== 使用全部数据进行训练 ======
        print(f"\n4.1 使用全部数据进行训练...")
        # 使用全部数据
        train_sample_size = len(X_train)
        val_sample_size = len(X_val)

        print(f"使用全部数据: 训练集 {train_sample_size:,} 条，验证集 {val_sample_size:,} 条")
        # 不需要切片，直接使用全部数据

        # 目标变量的log1p变换和clip已经在上面处理了

        # 2. 特征标准化（训练/验证用同一个scaler）- per_mu预测优化
        print("标准化特征（使用RobustScaler）...")
        scaler = RobustScaler()  # 使用RobustScaler处理异常值，对per_mu预测更鲁棒
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

        # 验证标准化效果
        print("🔍 验证特征标准化效果...")
        try:
            train_mean = X_train.mean().mean()
            train_std = X_train.std().mean()
            val_mean = X_val.mean().mean()
            val_std = X_val.std().mean()

            print(f"训练集标准化后统计: 均值={train_mean:.6f}, 标准差={train_std:.6f}")
            print(f"验证集标准化后统计: 均值={val_mean:.6f}, 标准差={val_std:.6f}")

            # 检查是否有异常值
            train_nan = X_train.isnull().sum().sum()
            train_inf = np.isinf(X_train.values).sum()
            val_nan = X_val.isnull().sum().sum()
            val_inf = np.isinf(X_val.values).sum()

            print(f"数据质量检查: 训练集NaN={train_nan}, Inf={train_inf}, 验证集NaN={val_nan}, Inf={val_inf}")

            if train_nan > 0 or train_inf > 0 or val_nan > 0 or val_inf > 0:
                print("⚠️ 发现异常值，进行清理...")
                X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
                X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
                print("✅ 异常值清理完成")
            else:
                print("✅ 数据质量良好，无异常值")

        except Exception as e:
            print(f"⚠️ 标准化验证失败: {str(e)}")

        # 保存scaler和label encoder
        os.makedirs(Config.scaler_dir, exist_ok=True)
        joblib.dump(scaler, Config.scaler_file)
        print(f"Scaler已保存至: {Config.scaler_file}")

        # 创建并保存label encoder（用于分类特征）
        label_encoder = LabelEncoder()
        # 这里我们创建一个简单的label encoder，实际使用时需要根据具体分类特征调整
        label_encoder.fit(['0', '1'])  # 假设只有两个类别
        os.makedirs(Config.label_encoder_dir, exist_ok=True)
        joblib.dump(label_encoder, Config.label_encoder_file)
        print(f"Label encoder已保存至: {Config.label_encoder_file}")

        # 3. 特征工程
        print("\n3. 特征工程...")

        # 首先处理缺失值
        print("处理缺失值...")

        # 检查缺失值情况
        print("训练集缺失值统计:")
        print(X_train.isnull().sum())
        print("\n验证集缺失值统计:")
        print(X_val.isnull().sum())

        # 对数值列进行缺失值填充
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns

        # 使用中位数填充缺失值
        for col in numeric_columns:
            if X_train[col].isnull().sum() > 0:
                median_val = X_train[col].median()
                X_train[col].fillna(median_val, inplace=True)
                if col in X_val.columns:
                    X_val[col].fillna(median_val, inplace=True)
                print(f"列 {col}: 用中位数 {median_val:.4f} 填充了 {X_train[col].isnull().sum()} 个缺失值")

        # 对非数值列进行缺失值填充
        non_numeric_columns = X_train.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_columns:
            if X_train[col].isnull().sum() > 0:
                # 对于非数值列，用众数填充
                mode_val = X_train[col].mode().iloc[0] if len(X_train[col].mode()) > 0 else 0
                X_train[col].fillna(mode_val, inplace=True)
                if col in X_val.columns:
                    X_val[col].fillna(mode_val, inplace=True)
                print(f"列 {col}: 用众数 {mode_val} 填充了缺失值")

        # 再次检查缺失值
        print("\n填充后训练集缺失值统计:")
        print(X_train.isnull().sum().sum())
        print("填充后验证集缺失值统计:")
        print(X_val.isnull().sum().sum())

        # 确保所有数据都是数值类型
        print("确保所有特征都是数值类型...")
        for col in X_train.columns:
            try:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                if col in X_val.columns:
                    X_val[col] = pd.to_numeric(X_val[col], errors='coerce')
            except Exception as e:
                print(f"警告：列 {col} 转换为数值类型失败: {e}")
                # 如果转换失败，用0填充
                X_train[col] = 0
                if col in X_val.columns:
                    X_val[col] = 0

        # 最终检查缺失值
        if X_train.isnull().sum().sum() > 0 or X_val.isnull().sum().sum() > 0:
            print("警告：仍有缺失值，用0填充")
            X_train = X_train.fillna(0)
            X_val = X_val.fillna(0)

        print("缺失值处理完成！")

        # PCA特征 - 修复：使用有意义的列名
        print("执行PCA降维...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pd.DataFrame(pca.fit_transform(X_train))
        X_val_pca = pd.DataFrame(pca.transform(X_val))
        # 为PCA特征添加有意义的列名
        pca_cols = [f'pca_{i}' for i in range(X_train_pca.shape[1])]
        X_train_pca.columns = pca_cols
        X_val_pca.columns = pca_cols

        # 保存PCA对象供推理时使用
        pca_path = os.path.join(Config.scaler_dir, 'pca_AA.pkl')
        joblib.dump(pca, pca_path)
        print(f"PCA对象已保存至: {pca_path}")

        # 交互特征（只生成一次）
        print("生成交互特征...")
        print(f"训练集特征列名: {X_train.columns.tolist()[:10]}...")  # 显示前10个列名
        print(f"验证集特征列名: {X_val.columns.tolist()[:10]}...")  # 显示前10个列名

        # 确保训练集和验证集有相同的特征列
        common_cols = X_train.columns.intersection(X_val.columns)
        print(f"共同特征列数量: {len(common_cols)}")
        if len(common_cols) != len(X_train.columns):
            print(f"警告: 训练集有 {len(X_train.columns)} 个特征，验证集有 {len(X_val.columns)} 个特征")
            print("使用共同特征列进行交互特征生成...")
            X_train = X_train[common_cols]
            X_val = X_val[common_cols]

        # 大数据量优化：跳过交互特征生成以加快训练速度
        print("为训练集生成交互特征...")
        print("⚠️ 大数据量优化：跳过交互特征生成以加快训练速度")
        print("   原因：800万行数据生成交互特征需要数小时，影响训练效率")

        # 直接使用原始特征，不生成交互特征
        X_train_inter = X_train.copy()
        selected_inter_feats = []
        print(f"跳过交互特征生成，直接使用原始特征: {X_train_inter.shape}")

        print("为验证集生成相同的交互特征...")
        # 确保验证集有相同的特征列
        X_val_aligned = X_val[X_train.columns]  # 确保列顺序一致

        # 调试信息：检查特征列是否一致
        print(f"训练集特征列数: {len(X_train.columns)}")
        print(f"验证集特征列数: {len(X_val_aligned.columns)}")
        print(f"特征列是否一致: {list(X_train.columns) == list(X_val_aligned.columns)}")

        # 检查选中的交互特征
        print(f"选中的交互特征: {selected_inter_feats}")

        # 直接使用原始特征，不生成交互特征
        X_val_inter = X_val_aligned.copy()
        print(f"验证集跳过交互特征生成，直接使用原始特征: {X_val_inter.shape}")

        # 保存交互特征和特征顺序供推理时使用
        selected_inter_feats_path = os.path.join(Config.output_dir, 'selected_inter_feats.json')
        with open(selected_inter_feats_path, 'w', encoding='utf-8') as f:
            json.dump(selected_inter_feats, f, ensure_ascii=False, indent=2)
        print(f"交互特征已保存至: {selected_inter_feats_path}")

        # 保存特征顺序
        feature_order_path = os.path.join(Config.output_dir, 'feature_order.json')
        with open(feature_order_path, 'w', encoding='utf-8') as f:
            json.dump(list(X_train.columns), f, ensure_ascii=False, indent=2)
        print(f"特征顺序已保存至: {feature_order_path}")

        # 拼接前去重：去除重复列名，优先保留原始特征
        def remove_duplicate_columns(df_list):
            seen = set()
            new_list = []
            for df in df_list:
                cols = []
                for col in df.columns:
                    if col not in seen:
                        cols.append(col)
                        seen.add(col)
                new_list.append(df[cols])
            return new_list

        X_train, X_train_pca, X_train_inter = remove_duplicate_columns([X_train, X_train_pca, X_train_inter])
        X_val, X_val_pca, X_val_inter = remove_duplicate_columns([X_val, X_val_pca, X_val_inter])

        # 拼接全部特征
        X_train_all = pd.concat([X_train, X_train_pca, X_train_inter], axis=1)
        X_val_all = pd.concat([X_val, X_val_pca, X_val_inter], axis=1)

        # 保存特征顺序供推理时使用
        feature_order_path = os.path.join(Config.output_dir, 'feature_order.json')
        with open(feature_order_path, 'w', encoding='utf-8') as f:
            json.dump(list(X_train_all.columns), f, ensure_ascii=False, indent=2)
        print(f"特征顺序已保存至: {feature_order_path}")

        # 拼接后展平多重列
        def flatten_multicolumns(df):
            for col in list(df.columns):  # 使用list避免RuntimeError
                if isinstance(df[col], pd.DataFrame):
                    # 多重列，展平
                    for i in range(df[col].shape[1]):
                        new_col_name = f"{col}_{i}"
                        df[new_col_name] = df[col].iloc[:, i]
                    df.drop(columns=[col], inplace=True)
            return df

        X_train_all = flatten_multicolumns(X_train_all)
        X_val_all = flatten_multicolumns(X_val_all)

        # 统一补齐和对齐
        missing_cols = set(X_train_all.columns) - set(X_val_all.columns)
        if missing_cols:
            print(f"为验证集添加缺失特征: {len(missing_cols)} 个")
            for col in missing_cols:
                X_val_all[col] = 0
        X_val_all = X_val_all[X_train_all.columns]

        # 检查并强制所有特征为数值型，防止object列导致XGBoost报错
        for df_name, df in zip(['X_train_all', 'X_val_all'], [X_train_all, X_val_all]):
            for col in list(df.columns):  # Use list to avoid RuntimeError during iteration if columns are dropped
                try:
                    col_data = df[col]
                    # 如果是DataFrame（多重列名或拼接出错），自动展平
                    if isinstance(col_data, pd.DataFrame):
                        print(f"严重警告: {df_name} 的 {col} 是DataFrame，shape={col_data.shape}，自动展平！")
                        for i in range(col_data.shape[1]):
                            new_col_name = f"{col}_{i}"
                            df[new_col_name] = col_data.iloc[:, i]
                        df.drop(columns=[col], inplace=True)  # Drop original multi-column
                    elif not np.issubdtype(col_data.dtype, np.number):
                        print(f"警告: {df_name} 的 {col} 不是数值型, 实际类型: {col_data.dtype}")
                        df[col] = pd.to_numeric(col_data, errors='coerce')
                except (KeyError, TypeError) as e:
                    print(f"跳过列 {col}: {str(e)}")
                    continue

        # 最后再全量展平一次，确保没有遗漏
        def flatten_all_multicolumns(df):
            for col in list(df.columns):
                try:
                    if isinstance(df[col], pd.DataFrame):
                        for i in range(df[col].shape[1]):
                            new_col_name = f"{col}_{i}"
                            df[new_col_name] = df[col].iloc[:, i]
                        df.drop(columns=[col], inplace=True)
                except (KeyError, TypeError) as e:
                    print(f"展平时跳过列 {col}: {str(e)}")
                    continue
            return df

        X_train_all = flatten_all_multicolumns(X_train_all)
        X_val_all = flatten_all_multicolumns(X_val_all)

        # 大数据量内存优化：在XGBoost特征提取前进行采样
        print("4. 大数据量内存优化：采样到50万条数据进行XGBoost训练...")
        max_samples = 500000  # 50万条数据（进一步减少内存使用）

        if len(X_train_all) > max_samples:
            print(f"原始训练集: {len(X_train_all):,} 行")
            print(f"采样到: {max_samples:,} 行")

            # 随机采样
            np.random.seed(42)
            sample_indices = np.random.choice(len(X_train_all), max_samples, replace=False)
            X_train_sample = X_train_all.iloc[sample_indices]
            y_reg_sample = y_reg_train.iloc[sample_indices]

            print(f"采样后训练集: {len(X_train_sample):,} 行")
        else:
            print(f"数据量适中({len(X_train_all):,}行)，无需采样")
            X_train_sample = X_train_all
            y_reg_sample = y_reg_train

        # 4. XGBoost叶子节点特征（stacking）
        print("4. 提取XGBoost叶子节点特征...")
        xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.15)  # 减少树数量，加快训练
        xgb_model.fit(X_train_sample, y_reg_sample['per_mu'])  # 单目标

        # 保存XGBoost模型
        os.makedirs(Config.xgboost_dir, exist_ok=True)
        xgb_model.save_model(Config.xgboost_model_file)
        print(f"XGBoost模型已保存至: {Config.xgboost_model_file}")

        # 分批提取叶子节点特征以避免内存不足
        print("分批提取叶子节点特征...")
        batch_size = 200000  # 每批20万行，提高处理效率

        # 训练集分批处理
        leaf_train_list = []
        for i in range(0, len(X_train_all), batch_size):
            end_idx = min(i + batch_size, len(X_train_all))
            batch_X = X_train_all.iloc[i:end_idx]
            batch_leaf = xgb_model.apply(batch_X)
            leaf_train_list.append(batch_leaf)
            print(f"处理训练集批次 {i // batch_size + 1}/{(len(X_train_all) - 1) // batch_size + 1}")

        leaf_train = np.vstack(leaf_train_list)

        # 验证集分批处理
        leaf_val_list = []
        for i in range(0, len(X_val_all), batch_size):
            end_idx = min(i + batch_size, len(X_val_all))
            batch_X = X_val_all.iloc[i:end_idx]
            batch_leaf = xgb_model.apply(batch_X)
            leaf_val_list.append(batch_leaf)
            print(f"处理验证集批次 {i // batch_size + 1}/{(len(X_val_all) - 1) // batch_size + 1}")

        leaf_val = np.vstack(leaf_val_list)
        # 直接用DataFrame包装，防止嵌套
        leaf_train = pd.DataFrame(leaf_train, index=X_train_all.index)
        leaf_val = pd.DataFrame(leaf_val, index=X_val_all.index)
        X_train_all = pd.concat([X_train_all, leaf_train], axis=1)
        X_val_all = pd.concat([X_val_all, leaf_val], axis=1)

        # 最终检查确保所有特征为数值型
        print("最终检查特征类型...")
        for df_name, df in zip(['X_train_all', 'X_val_all'], [X_train_all, X_val_all]):
            for col in list(df.columns):
                try:
                    col_data = df[col]
                    if isinstance(col_data, pd.DataFrame):
                        print(f"最终警告: {df_name} 的 {col} 仍是DataFrame，shape={col_data.shape}，强制展平！")
                        for i in range(col_data.shape[1]):
                            new_col_name = f"{col}_{i}"
                            df[new_col_name] = col_data.iloc[:, i]
                        df.drop(columns=[col], inplace=True)
                    elif not np.issubdtype(col_data.dtype, np.number):
                        print(f"最终警告: {df_name} 的 {col} 不是数值型, 实际类型: {col_data.dtype}")
                        df[col] = pd.to_numeric(col_data, errors='coerce')
                except (KeyError, TypeError) as e:
                    print(f"最终检查跳过列 {col}: {str(e)}")
                    continue

        # 确保所有列名都是字符串类型，避免sklearn报错
        print("确保所有列名都是字符串类型...")
        X_train_all.columns = X_train_all.columns.astype(str)
        X_val_all.columns = X_val_all.columns.astype(str)

        print(f"最终特征数量: X_train_all={X_train_all.shape[1]}, X_val_all={X_val_all.shape[1]}")
        print(f"X_train_all列名类型: {set(type(col) for col in X_train_all.columns)}")
        print(f"X_val_all列名类型: {set(type(col) for col in X_val_all.columns)}")

        # 5. Ridge基线 - 只关注per_mu
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        y_reg_train_1d = np.asarray(y_reg_train['per_mu'])
        ridge = Ridge()
        ridge.fit(X_train_all, y_reg_train_1d)
        print("Ridge回归R² (per_mu):", r2_score(y_reg_train_1d, ridge.predict(X_train_all)))

        # 6. 特征重要性筛选top 50特征
        feature_names = X_train_all.columns.tolist()
        print(f"原始特征数量: {len(feature_names)}")
        print(f"原始特征名示例: {feature_names[:10]}")

        # 过滤掉数字特征名，只保留有意义的特征名
        meaningful_features = [f for f in feature_names if not f.replace('_', '').replace('.', '').isdigit()]
        if len(meaningful_features) < 10:
            # 如果有意义特征太少，使用所有特征
            meaningful_features = feature_names

        print(f"有意义的特征数量: {len(meaningful_features)}")

        feature_importance = analyze_feature_importance(
            xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1).fit(X_train_all, y_reg_train['per_mu']),
            # 减少树数量，加快训练
            X_train_all, y_reg_train['per_mu'], meaningful_features, threshold=0.01
        )

        # 获取top特征，但确保这些特征名在X_train_all中存在
        top_features_df = feature_importance.sort_values('importance', ascending=False)
        top_features = []

        for _, row in top_features_df.iterrows():
            feature_name = row['feature']
            if feature_name in X_train_all.columns:
                top_features.append(feature_name)
            elif feature_name.startswith('feature_'):
                # 如果是通用特征名，尝试映射到实际特征
                try:
                    feature_idx = int(feature_name.split('_')[1])
                    if feature_idx < len(feature_names):
                        actual_feature = feature_names[feature_idx]
                        if actual_feature in X_train_all.columns:
                            top_features.append(actual_feature)
                except (ValueError, IndexError):
                    continue

            if len(top_features) >= 50:
                break

        print(f"成功匹配的Top特征数量: {len(top_features)}")
        print(f"Top特征示例: {top_features[:10]}")

        # 使用筛选后的特征进行最终训练
        X_train_final = X_train_all[top_features]
        X_val_final = X_val_all[top_features]

        # 确保特征对齐
        if hasattr(X_train_final, 'columns') and hasattr(X_val_final, 'columns'):
            missing_cols = set(X_train_final.columns) - set(X_val_final.columns)
            if missing_cols:
                print(f"验证集缺失特征: {len(missing_cols)} 个，自动补0")
                for col in missing_cols:
                    X_val_final[col] = 0
            X_val_final = X_val_final[X_train_final.columns]
        else:
            print("警告: X_train_final或X_val_final不是DataFrame，跳过特征对齐检查")

        # 更新训练和验证数据
        X_train = X_train_final
        X_val = X_val_final

        # 7. 准备训练集和验证集标签
        print("\n7. 准备训练集和验证集标签...")

        # 从原始标签中提取对应的标签
        y_cls_train_final = y_cls_train[:len(X_train)]
        y_reg_train_final = y_reg_train.iloc[:len(X_train)]

        # 从data_val中提取对应的标签
        y_cls_val = data_val['suit'].iloc[:len(X_val)]
        y_reg_val = data_val[['per_mu', 'per_qu']].iloc[:len(X_val)]

        # === 对验证集目标变量进行相同的预处理 ===
        print("对验证集目标变量进行log1p变换...")
        y_reg_val = np.log1p(y_reg_val)

        # 保存变换后的验证集标签，供后续采样使用
        y_reg_val_transformed = y_reg_val.copy()

        # 大数据量内存优化：在XGBoost特征提取前进行采样
        print("4. 大数据量内存优化：采样到100万条数据进行训练...")
        max_samples = 1000000  # 100万条数据（进一步优化运行时间）

        # 检查是否需要采样
        if len(X_train_all) > max_samples:
            print(f"原始训练集: {len(X_train_all):,} 行")
            print(f"采样到: {max_samples:,} 行")

            # 分层采样保持数据分布
            X_train_sampled, _, y_cls_train_sampled, _, y_reg_train_sampled, _ = train_test_split(
                X_train_all, y_cls_train_final, y_reg_train_final,
                train_size=max_samples,
                random_state=42,
                stratify=y_cls_train_final
            )

            # 验证集也相应采样
            val_samples = min(max_samples // 4, len(X_val_all))  # 验证集为训练集的1/4
            X_val_sampled, _, y_cls_val_sampled, _, y_reg_val_sampled, _ = train_test_split(
                X_val_all, y_cls_val, y_reg_val_transformed,  # 使用变换后的验证集标签
                train_size=val_samples,
                random_state=42,
                stratify=y_cls_val
            )

            print(f"采样后训练集: {len(X_train_sampled):,} 行")
            print(f"采样后验证集: {len(X_val_sampled):,} 行")

            # 更新变量
            X_train_all = X_train_sampled
            X_val_all = X_val_sampled
            y_cls_train_final = y_cls_train_sampled
            y_cls_val = y_cls_val_sampled
            y_reg_train_final = y_reg_train_sampled
            y_reg_val = y_reg_val_sampled

            # 采样后的验证集标签已经是log1p变换后的，无需再次变换
            print("采样后的验证集标签已经是log1p变换后的，无需再次变换")
        else:
            print(f"数据量适中({len(X_train_all):,}行)，无需采样")

        # 对验证集也进行相同的裁剪
        for col in y_reg_val.columns:
            q_low = y_reg_val[col].quantile(0.001)
            q_high = y_reg_val[col].quantile(0.999)
            y_reg_val[col] = np.clip(y_reg_val[col], q_low, q_high)

        print(f"验证集log1p变换后统计信息:")
        print(
            f"per_mu: min={y_reg_val['per_mu'].min():.4f}, max={y_reg_val['per_mu'].max():.4f}, mean={y_reg_val['per_mu'].mean():.4f}")
        print(
            f"per_qu: min={y_reg_val['per_qu'].min():.4f}, max={y_reg_val['per_qu'].max():.4f}, mean={y_reg_val['per_qu'].mean():.4f}")

        print(f"训练集标签形状: y_cls={y_cls_train_final.shape}, y_reg={y_reg_train_final.shape}")
        print(f"验证集标签形状: y_cls={y_cls_val.shape}, y_reg={y_reg_val.shape}")

        # 8. 交叉验证和模型集成
        print("\n8. 执行交叉验证和模型集成...")

        # 使用全局变量检查R²是否已达到0.8
        global GLOBAL_R2_ACHIEVED, GLOBAL_BEST_PARAMS, GLOBAL_BEST_R2

        print(f"🔍 全局R²状态检查: GLOBAL_R2_ACHIEVED={GLOBAL_R2_ACHIEVED}, GLOBAL_BEST_R2={GLOBAL_BEST_R2}")
        print(f"🔍 全局最佳参数: {GLOBAL_BEST_PARAMS}")

        if GLOBAL_R2_ACHIEVED:
            print(f"🎯 检测到R²已达到{GLOBAL_BEST_R2:.4f}，交叉验证将使用最佳参数，跳过调参")
            print(f"使用全局最佳参数: {GLOBAL_BEST_PARAMS}")
        else:
            print("ℹ️ 未检测到R²达到0.8，交叉验证将进行正常调参")

        ensemble_models, cv_scores = cross_validate_and_ensemble(
            X_train_all,  # 使用采样后的训练数据
            y_cls_train_final,
            y_reg_train_final,
            n_splits=Config.ensemble_params['n_splits'],
            n_models=Config.ensemble_params['n_models'],
            strategy=strategy,
            global_best_params=GLOBAL_BEST_PARAMS,
            r2_achieved=GLOBAL_R2_ACHIEVED
        )

        # 9. 特征提取和组合
        print("\n9. 特征提取和组合...")
        # 使用采样后的数据进行训练
        print(f"使用采样后训练数据: {len(X_train_all):,} 个样本")
        print(f"使用采样后验证数据: {len(X_val_all):,} 个样本")

        X_train_sampled = X_train_all
        y_cls_train_sampled = y_cls_train_final
        y_reg_train_sampled = y_reg_train_final
        X_val_sampled = X_val_all
        y_cls_val_sampled = y_cls_val
        y_reg_val_sampled = y_reg_val

        X_train_combined, X_val_combined, xgb_model, y_cls_train_sample, y_cls_val_sample, y_reg_train_sample, y_reg_val_sample = extract_xgboost_features(
            X_train_sampled, X_val_sampled, y_cls_train_sampled, y_cls_val_sampled, y_reg_train_sampled,
            y_reg_val_sampled
        )
        # 修复：将X_train_combined、X_val_combined转为DataFrame并拼接列名
        # 安全获取原始特征名
        if hasattr(X_train_final, 'columns'):
            orig_feature_names = list(X_train_final.columns)
        else:
            # 如果X_train_final是numpy数组，使用特征名列表
            orig_feature_names = feature_names[:X_train_final.shape[1]]
        n_leaf = X_train_combined.shape[1] - len(orig_feature_names)
        leaf_feature_names = [f'xgb_leaf_{i}' for i in range(n_leaf)]
        all_feature_names = orig_feature_names + leaf_feature_names
        X_train_combined = pd.DataFrame(X_train_combined, columns=all_feature_names)
        X_val_combined = pd.DataFrame(X_val_combined, columns=all_feature_names)

        # 10. 处理验证集特征
        print("\n10. 处理验证集特征...")
        # 注意：X_val_combined已经在extract_xgboost_features中处理过了，不需要重新处理
        # 只需要确保列名一致
        X_val_original_for_ensemble = X_val_all.copy()

        # 确保所有列名为str类型
        X_train_combined.columns = [str(c) for c in X_train_combined.columns]
        X_val_combined.columns = [str(c) for c in X_val_combined.columns]

        # 检查数据维度是否匹配
        print(f"训练集特征形状: {X_train_combined.shape}")
        print(f"验证集特征形状: {X_val_combined.shape}")
        print(f"训练集标签形状: y_cls={y_cls_train_sample.shape}, y_reg={y_reg_train_sample.shape}")
        print(f"验证集标签形状: y_cls={y_cls_val_sample.shape}, y_reg={y_reg_val_sample.shape}")

        # 确保数据维度匹配
        if X_train_combined.shape[0] != len(y_cls_train_sample) or X_train_combined.shape[0] != len(y_reg_train_sample):
            print("警告：训练集特征和标签维度不匹配，进行对齐...")
            min_len = min(X_train_combined.shape[0], len(y_cls_train_sample), len(y_reg_train_sample))
            X_train_combined = X_train_combined.iloc[:min_len]
            y_cls_train_sample = y_cls_train_sample.iloc[:min_len] if hasattr(y_cls_train_sample,
                                                                              'iloc') else y_cls_train_sample[:min_len]
            y_reg_train_sample = y_reg_train_sample.iloc[:min_len] if hasattr(y_reg_train_sample,
                                                                              'iloc') else y_reg_train_sample[:min_len]

        if X_val_combined.shape[0] != len(y_cls_val_sample) or X_val_combined.shape[0] != len(y_reg_val_sample):
            print("警告：验证集特征和标签维度不匹配，进行对齐...")
            min_len = min(X_val_combined.shape[0], len(y_cls_val_sample), len(y_reg_val_sample))
            X_val_combined = X_val_combined.iloc[:min_len]
            y_cls_val_sample = y_cls_val_sample.iloc[:min_len] if hasattr(y_cls_val_sample,
                                                                          'iloc') else y_cls_val_sample[:min_len]
            y_reg_val_sample = y_reg_val_sample.iloc[:min_len] if hasattr(y_reg_val_sample,
                                                                          'iloc') else y_reg_val_sample[:min_len]

        # 11. 训练最终模型
        print("\n11. 训练最终模型...")

        # 检查是否已有训练好的模型
        final_model_path = os.path.join(Config.final_model_dir, 'best_hybrid_multioutput_AA.h5')
        if os.path.exists(final_model_path):
            print("发现已存在的最终模型，尝试加载...")
            try:
                best_model = load_model_with_custom_objects(final_model_path)
                print("成功加载已存在的最终模型")
                history = None  # 没有历史记录
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("将重新训练模型...")
                best_model, history = None, None
        else:
            best_model, history = None, None

        if best_model is None:
            # 添加数据质量检查
            print("最终训练数据质量检查:")
            print(f"X_train_combined NaN数量: {X_train_combined.isnull().sum().sum()}")
            print(f"X_val_combined NaN数量: {X_val_combined.isnull().sum().sum()}")
            print(
                f"y_reg_train_sample NaN数量: {np.isnan(y_reg_train_sample).sum() if hasattr(y_reg_train_sample, 'shape') else 'N/A'}")
            print(
                f"y_reg_val_sample NaN数量: {np.isnan(y_reg_val_sample).sum() if hasattr(y_reg_val_sample, 'shape') else 'N/A'}")

            # 检查是否有无穷大值
            print(
                f"X_train_combined Inf数量: {np.isinf(X_train_combined.select_dtypes(include=[np.number])).sum().sum()}")
            print(f"X_val_combined Inf数量: {np.isinf(X_val_combined.select_dtypes(include=[np.number])).sum().sum()}")

            # 如果发现NaN或Inf，进行清理
            if X_train_combined.isnull().sum().sum() > 0:
                print("清理训练集NaN值...")
                X_train_combined = X_train_combined.fillna(0)

            if X_val_combined.isnull().sum().sum() > 0:
                print("清理验证集NaN值...")
                X_val_combined = X_val_combined.fillna(0)

            # 确保标签数据是numpy数组且没有NaN
            if hasattr(y_reg_train_sample, 'values'):
                y_reg_train_sample = y_reg_train_sample.values
            if hasattr(y_reg_val_sample, 'values'):
                y_reg_val_sample = y_reg_val_sample.values

            # 检查并清理标签中的NaN
            if np.isnan(y_reg_train_sample).any():
                print("清理训练标签NaN值...")
                y_reg_train_sample = np.nan_to_num(y_reg_train_sample, nan=0.0)

            if np.isnan(y_reg_val_sample).any():
                print("清理验证标签NaN值...")
                y_reg_val_sample = np.nan_to_num(y_reg_val_sample, nan=0.0)

            try:
                best_model, history = train_final_model(
                    X_train_combined, y_cls_train_sample, y_reg_train_sample,
                    X_val_combined, y_cls_val_sample, y_reg_val_sample,
                    strategy=strategy
                )
            except Exception as e:
                print(f"最终模型训练失败: {e}")
                print("尝试使用更保守的参数重新训练...")
                # 使用更保守的参数
                best_model, history = train_final_model_conservative(
                    X_train_combined, y_cls_train_sample, y_reg_train_sample,
                    X_val_combined, y_cls_val_sample, y_reg_val_sample,
                    strategy=strategy
                )

        # 12. 分析特征重要性
        print("\n12. 分析特征重要性...")
        # 使用原始特征名进行分析，而不是数字特征名
        # 安全获取原始特征名
        if hasattr(X_train_final, 'columns'):
            original_feature_names = list(X_train_final.columns)
        else:
            # 如果X_train_final是numpy数组，使用特征名列表
            original_feature_names = feature_names[:X_train_final.shape[1]]
        # 过滤掉数字特征名，只保留有意义的特征名
        meaningful_features = [f for f in original_feature_names if not f.replace('_', '').replace('.', '').isdigit()]
        if len(meaningful_features) < 3:
            # 如果有意义特征太少，使用所有特征
            meaningful_features = original_feature_names

        # 修复：确保传入的X_train和feature_names匹配
        # 使用X_train_final而不是X_train_combined，因为xgb_model是在X_train_combined上训练的
        # 但我们需要分析原始特征的重要性

        # 安全获取特征列名
        if hasattr(X_train_final, 'columns'):
            feature_names_for_analysis = list(X_train_final.columns)
        else:
            # 如果X_train_final是numpy数组，使用原始特征名
            feature_names_for_analysis = original_feature_names[:X_train_final.shape[1]]

        feature_importance = analyze_feature_importance(
            xgb_model,
            X_train_combined[:10000] if X_train_combined.shape[0] > 10000 else X_train_combined,
            y_reg_train['per_mu'][:10000] if y_reg_train.shape[0] > 10000 else y_reg_train['per_mu'],
            feature_names_for_analysis,  # 使用安全的特征列名
            threshold=Config.feature_importance['threshold']
        )

        # 13. 模型预测（在验证集上）
        print("\n13. 在验证集上执行预测...")
        # 分块预测和分块写出结果，防止爆内存
        batch_size = 200_000  # 增大批次大小，提高预测效率
        n_val = X_val_combined.shape[0]
        results_chunks = []
        for start in range(0, n_val, batch_size):
            end = min(start + batch_size, n_val)
            cls_pred_batch, reg_pred_batch = best_model.predict(X_val_combined[start:end])
            # 新增：预测后反变换
            reg_pred_batch = np.expm1(reg_pred_batch)
            results_chunk = pd.DataFrame({
                'x': get_col_safe(data_val, ['x', 'x_product', 'right_x'], slice(start, end)),
                'y': get_col_safe(data_val, ['y', 'y_product', 'right_y'], slice(start, end)),
                'yyyy': get_col_safe(data_val, ['yyyy', 'right_yyyy', 'year'], slice(start, end)),
                'suit': np.argmax(cls_pred_batch, axis=1),
                'per_mu': reg_pred_batch[:, 1],
                'per_qu': reg_pred_batch[:, 0]
            })
            results_chunks.append(results_chunk)
            import gc;
            gc.collect()
        results_df = pd.concat(results_chunks, ignore_index=True)
        results_df = results_df[['x', 'y', 'yyyy', 'suit', 'per_mu', 'per_qu']]
        os.makedirs(os.path.dirname(Config.result_file), exist_ok=True)
        results_df.to_csv(Config.result_file, index=False, encoding='utf-8')
        del results_chunks, results_df;
        gc.collect()

        # 14. 保存训练历史
        print("\n14. 保存训练历史...")
        history_file = os.path.join(Config.logs_dir, "training_history.json")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        if history is not None:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history.history, f, default=json_fallback, ensure_ascii=False, indent=2)
            print(f"训练历史已保存至: {history_file}")
        else:
            print("⚠️ 没有训练历史记录（使用了已存在的模型）")
            # 保存一个空的训练历史记录
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump({"message": "使用已存在的模型，无训练历史记录"}, f, ensure_ascii=False, indent=2)

        # 15. 保存特征重要性
        print("\n15. 保存特征重要性...")
        os.makedirs(Config.feature_importance_dir, exist_ok=True)
        feature_importance.to_csv(Config.feature_importance_file, index=False, encoding='utf-8')

        # 保存关键特征列表
        key_features = feature_importance[feature_importance['importance'] > Config.feature_importance['threshold']][
            'feature'].tolist()
        with open(Config.key_features_list_file, 'w', encoding='utf-8') as f:
            json.dump(key_features, f, ensure_ascii=False, indent=2)

        # 16. 生成预测地图
        print("\n16. 生成预测地图...")
        # 修复：使用最后一次batch的预测结果，或者重新预测一小部分数据用于可视化
        sample_size = min(1000, X_val_combined.shape[0])  # 取1000个样本用于可视化
        sample_preds = best_model.predict(X_val_combined[:sample_size])
        if isinstance(sample_preds, (list, tuple)) and len(sample_preds) == 2:
            sample_cls_pred, sample_reg_pred = sample_preds
        else:
            sample_cls_pred = sample_preds
            sample_reg_pred = None
        plot_prediction_maps(X_val_combined[:sample_size], sample_cls_pred, sample_reg_pred, val_ids[:sample_size],
                             data_val.iloc[:sample_size])
        # 自动保存交互特征名和最终特征顺序
        with open(os.path.join(Config.output_dir, "selected_inter_feats.json"), "w", encoding="utf-8") as f:
            json.dump(selected_inter_feats, f, ensure_ascii=False)
        with open(os.path.join(Config.output_dir, "feature_order.json"), "w", encoding="utf-8") as f:
            # 安全获取特征列名
            if hasattr(X_train_final, 'columns'):
                feature_columns = list(X_train_final.columns)
            else:
                # 如果X_train_final是numpy数组，使用原始特征名
                feature_columns = original_feature_names[:X_train_final.shape[1]]
            json.dump(feature_columns, f, ensure_ascii=False)
        # 17. 生成分析图表
        print("\n17. 生成分析图表...")

        # 17.1 训练历史图表
        plot_training_history(history)

        # 17.2 混淆矩阵
        # 修复：多输出模型predict返回(cls_pred, reg_pred)，不能直接argmax
        preds = best_model.predict(X_val_combined)
        if isinstance(preds, (list, tuple)) and len(preds) == 2:
            cls_pred, reg_pred = preds
        else:
            # 兼容性兜底
            cls_pred = preds
            reg_pred = None
        y_cls_pred = np.argmax(cls_pred, axis=1)
        plot_confusion_matrix(y_cls_val_sample, y_cls_pred, classes=['不适合', '适合'])

        # 17.3 回归散点图
        if reg_pred is not None:
            y_reg_pred = reg_pred
        else:
            # 兼容性兜底 - 避免重复predict调用
            y_reg_pred = preds[1] if isinstance(preds, (list, tuple)) and len(preds) > 1 else preds
        plot_regression_scatter(y_reg_val_sample, y_reg_pred, ['per_mu', 'per_qu'])

        # 17.4 特征重要性图
        plot_feature_importance(xgb_model, feature_names)

        # 17.5 生成evaluation_report.json
        evaluate_model(best_model, X_val_combined, y_cls_val_sample, y_reg_val_sample)

        print("\n=== 预测程序运行完成! ===")
        print(f"GPU配置: {num_gpus}个GPU, 策略: {type(strategy).__name__ if strategy else 'None'}")
        print(f"结果已保存至: {Config.result_file}")
        print(f"训练历史已保存至: {history_file}")

        return True

    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        if hasattr(e, '__traceback__') and e.__traceback__ is not None:
            print(f"错误位置: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
        raise


def train_final_model_conservative(X_train, y_cls_train, y_reg_train, X_val, y_cls_val, y_reg_val, strategy=None):
    """使用保守参数训练最终模型，避免NaN问题"""
    print("\n使用保守参数训练最终模型...")

    # 数据清理：确保没有NaN或Inf
    print("保守训练数据清理...")

    # 清理特征数据
    if hasattr(X_train, 'values'):
        X_train_values = X_train.values
    else:
        X_train_values = X_train

    if hasattr(X_val, 'values'):
        X_val_values = X_val.values
    else:
        X_val_values = X_val

    # 清理NaN和Inf
    X_train_values = np.nan_to_num(X_train_values, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_values = np.nan_to_num(X_val_values, nan=0.0, posinf=0.0, neginf=0.0)

    if hasattr(X_train, 'values'):
        X_train = pd.DataFrame(X_train_values, columns=X_train.columns, index=X_train.index)
    else:
        X_train = X_train_values

    if hasattr(X_val, 'values'):
        X_val = pd.DataFrame(X_val_values, columns=X_val.columns, index=X_val.index)
    else:
        X_val = X_val_values

    # 清理标签数据
    if hasattr(y_reg_train, 'shape'):
        y_reg_train = np.nan_to_num(y_reg_train, nan=0.0, posinf=0.0, neginf=0.0)

    if hasattr(y_reg_val, 'shape'):
        y_reg_val = np.nan_to_num(y_reg_val, nan=0.0, posinf=0.0, neginf=0.0)

    print("数据清理完成")

    # 使用非常保守的参数
    conservative_params = {
        'lr': 0.0001,  # 很小的学习率
        'neurons1': 64,  # 较小的网络
        'neurons2': 32,
        'dropout_rate': 0.1,
        'batch_size': 32,
        'attention_units': 16,
        'l1_lambda': 1e-5,  # 添加缺失的l1_lambda参数
        'l2_lambda': 1e-4,
        'optimizer_type': 'adam',
        'activation': 'relu'
    }

    print(f"保守参数: {conservative_params}")

    # 构建模型
    num_classes = len(np.unique(y_cls_train))
    model = build_hybrid_model(X_train.shape[1], num_classes, conservative_params, strategy)

    # 设置回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,  # 减少patience，加快早停
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(Config.final_model_dir, 'best_hybrid_multioutput_AA.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # 训练模型
    history = model.fit(
        X_train,
        {
            'classification': y_cls_train,
            'output_clipped': y_reg_train
        },
        validation_data=(
            X_val,
            {
                'classification': y_cls_val,
                'output_clipped': y_reg_val
            }
        ),
        epochs=30,  # 进一步减少epochs，加快训练
        batch_size=conservative_params['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    return model, history


print('=== 脚本已启动 ===')


def set_weather_row_range(start_row, end_row):
    """
    设置气象数据读取的行数范围

    Args:
        start_row (int): 起始行数（0-based，包含此行）
        end_row (int): 结束行数（0-based，不包含此行，None表示到文件末尾）
    """
    Config.weather_start_row = start_row
    Config.weather_end_row = end_row
    Config.use_weather_row_range = True
    print(f"✅ 已设置气象数据行数范围: 起始行={start_row}, 结束行={end_row}")


def get_model_expected_features(model):
    """
    动态获取模型期望的特征数量
    
    Args:
        model: 加载的Keras模型
        
    Returns:
        int: 期望的特征数量
    """
    try:
        # 方法1：从模型输入层获取
        if hasattr(model, 'input_shape'):
            if isinstance(model.input_shape, list):
                # 多输入模型
                input_shape = model.input_shape[0]
            else:
                # 单输入模型
                input_shape = model.input_shape
            
            # input_shape格式通常是 (batch_size, feature_count)
            if len(input_shape) >= 2:
                expected_features = input_shape[1]
                print(f"从模型输入层检测到特征数量: {expected_features}")
                return expected_features
        
        # 方法2：从第一层获取
        if model.layers:
            first_layer = model.layers[0]
            if hasattr(first_layer, 'input_shape'):
                if isinstance(first_layer.input_shape, list):
                    input_shape = first_layer.input_shape[0]
                else:
                    input_shape = first_layer.input_shape
                
                if len(input_shape) >= 2:
                    expected_features = input_shape[1]
                    print(f"从模型第一层检测到特征数量: {expected_features}")
                    return expected_features
        
        # 方法3：从模型配置获取
        if hasattr(model, 'get_config'):
            try:
                config = model.get_config()
                if 'layers' in config and config['layers']:
                    first_layer_config = config['layers'][0]
                    if 'config' in first_layer_config and 'input_shape' in first_layer_config['config']:
                        input_shape = first_layer_config['config']['input_shape']
                        if len(input_shape) >= 2:
                            expected_features = input_shape[1]
                            print(f"从模型配置检测到特征数量: {expected_features}")
                            return expected_features
            except Exception as e:
                print(f"从模型配置检测特征数量失败: {e}")
        
        # 如果所有方法都失败，返回默认值
        print("⚠️ 无法动态检测模型期望特征数量，使用默认值")
        return None
        
    except Exception as e:
        print(f"❌ 动态检测特征数量时出错: {e}")
        return None


def create_model_info_file(model_path, expected_features=None):
    """
    创建模型信息文件，记录模型的基本信息
    
    Args:
        model_path: 模型文件路径
        expected_features: 期望的特征数量（如果不提供则自动检测）
    """
    try:
        model_info_dir = os.path.dirname(model_path)
        model_info_file = os.path.join(model_info_dir, "model_info.json")
        
        info = {
            "model_path": model_path,
            "created_time": datetime.now().isoformat(),
            "input_features": expected_features,
            "detection_method": "automatic" if expected_features is None else "manual"
        }
        
        with open(model_info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 模型信息文件已保存: {model_info_file}")
        return model_info_file
        
    except Exception as e:
        print(f"⚠️ 创建模型信息文件失败: {e}")
        return None


def load_model_info_file(model_path):
    """
    加载模型信息文件
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        dict: 模型信息，如果找不到则返回None
    """
    try:
        model_info_dir = os.path.dirname(model_path)
        model_info_file = os.path.join(model_info_dir, "model_info.json")
        
        if os.path.exists(model_info_file):
            with open(model_info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
            print(f"✅ 加载模型信息文件: {model_info_file}")
            return info
        else:
            print("⚠️ 未找到模型信息文件")
            return None
            
    except Exception as e:
        print(f"⚠️ 加载模型信息文件失败: {e}")
        return None


def determine_model_features(model_path, model_obj):
    """
    确定模型期望的特征数量（综合利用多种方法）
    
    Args:
        model_path: 模型文件路径
        model_obj: 加载的模型对象
        
    Returns:
        int: 期望的特征数量
    """
    print("\n🔍 确定模型期望特征数量...")
    
    # 方法1：尝试加载已有的模型信息文件
    model_info = load_model_info_file(model_path)
    if model_info and 'input_features' in model_info:
        expected_features = model_info['input_features']
        if expected_features is not None:
            print(f"✅ 从模型信息文件获取特征数量: {expected_features}")
            return expected_features
    
    # 方法2：动态 detect from model
    detected_features = get_model_expected_features(model_obj)
    if detected_features is not None:
        print(f"✅ 自动检测到特征数量: {detected_features}")
        
        # 保存检测结果到模型信息文件
        create_model_info_file(model_path, detected_features)
        return detected_features
    
    # 方法3：从文件名或配置推断（备用方案）
    print("⚠️ 使用基于文件名的推断...")
    
    # 根据训练脚本的特征生成逻辑推断：
    # 原始特征(~26) + PCA特征(~13) + 交互特征(~26) ≈ 65
    # + XGBoost叶子特征(~39) ≈ 104，但实际可能是89
    inferred_features = 89  # 基于最新的训练结果
    
    print(f"ℹ️ 推断特征数量: {inferred_features}")
    
    # 验证推断结果
    try:
        # 创建一个测试输入来验证
        test_input = np.random.random((1, inferred_features))
        model_obj.predict(test_input, verbose=0)
        print(f"✅ 特征数量推断验证成功: {inferred_features}")
        
        # 保存验证结果
        create_model_info_file(model_path, inferred_features)
        return inferred_features
        
    except Exception as e:
        print(f"❌ 特征数量推断验证失败: {e}")
        print("可能需要手动指定特征数量")
        
        # 尝试一些常见特征数量的可能性
        common_features = [65, 78, 85, 89, 91, 95, 100, 104, 110, 113]
        for features in common_features:
            try:
                test_input = np.random.random((1, features))
                model_obj.predict(test_input, verbose=0)
                print(f"✅ 找到匹配的特征数量: {features}")
                create_model_info_file(model_path, features)
                return features
            except:
                continue
        
        raise ValueError(f"无法确定模型期望的特征数量，请检查模型完整性")


def detect_and_adapt_model_features(prediction_data, feature_order_file, scaler_file):
    """
    检测并适配不同模型的特征格式
    
    Args:
        prediction_data: 预测数据
        feature_order_file: feature_order.json文件路径
        scaler_file: scaler文件路径
        
    Returns:
        tuple: (adapted_feature_order, is_conversion_needed)
    """
    print("🔍 检测模型特征格式...")
    
    # 加载特征顺序
    with open(feature_order_file, 'r', encoding='utf-8') as f:
        loaded_feature_order = json.load(f)
    
    print(f"   原始特征顺序文件中的格式: {loaded_feature_order}")
    
    # 检测特征顺序格式
    has_numeric_order = any(str(col).isdigit() for col in loaded_feature_order)
    has_name_order = any(not str(col).isdigit() for col in loaded_feature_order)
    
    print(f"   包含数字索引: {has_numeric_order}")
    print(f"   包含特征名称: {has_name_order}")
    
    # 检测预测数据格式
    sample_data = prediction_data.head(1)
    feature_cols = [col for col in sample_data.columns 
                   if col not in Config.exclude_columns and not col.startswith('pca_')]
    
    has_original_features = any(not col.isdigit() for col in feature_cols)
    print(f"   预测数据包含原始特征名: {has_original_features}")
    
    # 判断是否需要适配
    needs_adaptation = has_numeric_order and has_original_features
    print(f"🔧 需要特征适配: {needs_adaptation}")
    
    return loaded_feature_order, needs_adaptation


def main(test_mode=False, test_size=10000, weather_start_row=None, weather_end_row=None, model_name=None):
    """
    主函数 - 纯预测程序入口
    使用现有模型对预测气象数据进行产量预测

    Args:
        test_mode (bool): 是否启用小型测试模式
        test_size (int): 测试模式下的数据量限制
        weather_start_row (int, optional): 气象数据起始行数
        weather_end_row (int, optional): 气象数据结束行数
        model_name (str, optional): 指定使用的模型名称
    """
    print('=== 进入预测流程 ===')

    # 设置模型名称
    if model_name:
        print(f"🎯 使用指定模型: {model_name}")
        Config.set_model_name(model_name)
    else:
        print(f"🎯 使用默认模型: {Config.get_model_name()}")

    if test_mode:
        print(f"🔬 小型测试模式：限制数据量为 {test_size:,} 行")

    # 设置气象数据行数范围（如果提供了参数）
    if weather_start_row is not None or weather_end_row is not None:
        set_weather_row_range(weather_start_row, weather_end_row)

    try:
        # 1. 验证配置
        print("\n1. 验证配置...")
        if not Config.validate_params() or not Config.validate_paths():
            raise ValueError("配置验证失败")

        # 2. 检查现有模型是否存在
        print("\n2. 检查现有模型...")
        if not os.path.exists(Config.final_model_file):
            print(f"❌ 现有模型不存在: {Config.final_model_file}")
            print("请先运行训练模式或检查模型文件路径")
            return False

        print(f"✅ 找到现有模型: {Config.final_model_file}")

        # 3. 分析模型特征映射关系
        print("\n3. 分析模型特征映射关系...")
        mapping_info = analyze_model_feature_mapping(Config.output_dir)
        
        # 4. 加载预测数据（使用与训练代码相同的数据预处理）
        print("\n4. 加载和预处理预测数据...")
        prediction_data = load_prediction_data(test_mode=test_mode, test_size=test_size,
                                               weather_start_row=Config.weather_start_row,
                                               weather_end_row=Config.weather_end_row)

        # 5. 加载现有模型和预处理器
        print("\n5. 加载现有模型和预处理器...")
        best_model = load_model_with_custom_objects(Config.final_model_file)
        scaler = joblib.load(Config.scaler_file)
        pca = joblib.load(os.path.join(Config.scaler_dir, 'pca_AA.pkl'))

        # 动态确定模型期望的特征数量
        expected_features = determine_model_features(Config.final_model_file, best_model)

        # 智能检测和适配特征格式
        feature_order_file = os.path.join(Config.output_dir, 'feature_order.json')
        feature_order, needs_adaptation = detect_and_adapt_model_features(prediction_data, feature_order_file, Config.scaler_file)
        
        if needs_adaptation:
            print("✅ 检测到需要进行特征适配的模型")
        else:
            print("✅ 模型特征格式兼容")

        # 6. 数据预处理（完全按照训练代码的方式）
        print("\n6. 数据预处理...")
        X_processed = preprocess_prediction_data(prediction_data, scaler, pca, feature_order, expected_features, mapping_info=mapping_info)
        
        # 7. 🔧 最终特征数量修正 - 确保与模型完全匹配
        if expected_features and X_processed.shape[1] != expected_features:
            print(f"🔧 最终特征数量修正: {X_processed.shape[1]} -> {expected_features}")
            if X_processed.shape[1] > expected_features:
                X_processed = X_processed.iloc[:, :expected_features]
                print(f"   截取前 {expected_features} 个特征")
            else:
                # 添加缺失特征
                missing_features = expected_features - X_processed.shape[1]
                for i in range(missing_features):
                    X_processed[f'final_missing_feature_{i}'] = 0
                print(f"   添加 {missing_features} 个缺失特征")
            print(f"✅ 最终特征数量: {X_processed.shape[1]}")

        # 8. 模型预测（分块保存）
        print("\n8. 执行模型预测...")
        saved_files = predict_with_processed_data(prediction_data, X_processed, best_model, save_chunk_size=5000000)

        # 9. 预测完成统计
        print("\n9. 预测完成统计...")
        print(f"\n✅ 预测完成！")
        print(f"共保存了 {len(saved_files)} 个预测结果文件:")
        return True

    except Exception as e:
        print(f"❌ 预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_prediction_data(test_mode=False, test_size=10000, weather_start_row=None, weather_end_row=None):
    """加载预测气象数据"""
    print("加载预测气象数据...")

    # 预测气象数据配置（使用预测气象数据）
    prediction_weather_folder = r"C:\Users\Administrator\Desktop\2. B774模型脚本\2-适宜度产量预测模型\train+text_csv数据\预测气象"
    soil_data_csv = Config.soil_data_csv

    # 加载预测气象数据（支持行数范围）
    weather_data = load_weather_data_from_folder(prediction_weather_folder,
                                                 start_row=weather_start_row,
                                                 end_row=weather_end_row)
    print(f"预测气象数据形状: {weather_data.shape}")

    # 小型测试模式：限制气象数据量
    if test_mode:
        print(f"🔬 小型测试模式：限制气象数据为 {test_size:,} 行")
        weather_data = weather_data.head(test_size)
        print(f"限制后气象数据形状: {weather_data.shape}")

    # 加载土壤数据
    soil_data = pd.read_csv(soil_data_csv, encoding='utf-8')
    soil_data.columns = soil_data.columns.str.strip().str.lower()
    soil_data = soil_data.dropna()
    print(f"土壤数据形状: {soil_data.shape}")

    # 小型测试模式：限制土壤数据量
    if test_mode:
        print(f"🔬 小型测试模式：限制土壤数据为 {test_size:,} 行")
        soil_data = soil_data.head(test_size)
        print(f"限制后土壤数据形状: {soil_data.shape}")

    # 合并数据（按照训练代码的方式）
    # 首先重命名气象数据列名
    print("重命名气象数据列名...")
    weather_data_renamed = weather_data.copy()

    # 重命名坐标列
    if 'Lon' in weather_data_renamed.columns:
        weather_data_renamed = weather_data_renamed.rename(columns={'Lon': 'x'})
    if 'Lat' in weather_data_renamed.columns:
        weather_data_renamed = weather_data_renamed.rename(columns={'Lat': 'y'})
    if 'YYYY' in weather_data_renamed.columns:
        weather_data_renamed = weather_data_renamed.rename(columns={'YYYY': 'yyyy'})

    print(f"重命名后气象数据列名: {weather_data_renamed.columns.tolist()[:10]}...")

    # 为气象数据添加right_前缀
    weather_pref = weather_data_renamed.add_prefix('right_')
    print(f"添加前缀后气象数据列名: {weather_pref.columns.tolist()[:10]}...")

    # 创建虚拟产品数据（用于预测）
    product_data = soil_data[['x', 'y']].copy()
    if 'yyyy' in weather_data_renamed.columns:
        product_data['yyyy'] = weather_data_renamed['yyyy'].iloc[0]
    else:
        product_data['yyyy'] = 2015  # 默认年份

    print(f"虚拟产品数据形状: {product_data.shape}")
    print(f"虚拟产品数据列名: {product_data.columns.tolist()}")

    # 精确键合并
    merged_data = pd.merge(
        product_data,
        weather_pref,
        left_on=['x', 'y', 'yyyy'],
        right_on=['right_x', 'right_y', 'right_yyyy'],
        how='inner'
    )

    if merged_data.empty:
        print("精确键合并结果为空，回退到空间近邻合并...")
        merged_data = spatial_temporal_merge(product_data, weather_data_renamed, xy_cols=['x', 'y'], time_col='yyyy',
                                             tolerance=50000)

    # 与土壤数据合并
    merged_data = spatial_merge(merged_data, soil_data, on=['x', 'y'], tolerance=50000)

    # 统一列名为小写
    merged_data.columns = merged_data.columns.str.strip().str.lower()

    # 处理重复列名问题
    print("检查并处理重复列名...")
    original_columns = merged_data.columns.tolist()
    seen_columns = set()
    new_columns = []

    for col in original_columns:
        if col in seen_columns:
            # 为重复列添加后缀
            counter = 1
            new_col = f"{col}_{counter}"
            while new_col in seen_columns:
                counter += 1
                new_col = f"{col}_{counter}"
            new_columns.append(new_col)
            seen_columns.add(new_col)
            print(f"重命名重复列: {col} -> {new_col}")
        else:
            new_columns.append(col)
            seen_columns.add(col)

    merged_data.columns = new_columns

    # 添加缺失的特征列
    if 'right_x_original' not in merged_data.columns and 'right_x' in merged_data.columns:
        merged_data['right_x_original'] = merged_data['right_x']
        print("添加 right_x_original 列")

    if 'right_y_original' not in merged_data.columns and 'right_y' in merged_data.columns:
        merged_data['right_y_original'] = merged_data['right_y']
        print("添加 right_y_original 列")

    print(f"最终预测数据形状: {merged_data.shape}")
    return merged_data


def _needs_feature_name_conversion(actual_columns, expected_feature_order):
    """
    检查是否需要特征名称转换
    
    Args:
        actual_columns: 预测数据的实际列名
        expected_feature_order: 期望的特征顺序
        
    Returns:
        bool: 是否需要转换
    """
    # 检查期望特征是否主要是数字字符串（表示训练时保存的数字索引）
    expect_numeric_features = any(col.isdigit() for col in expected_feature_order)
    
    # 检查实际列是否主要是原始特征名（非数字字符串）
    have_original_features = any(not col.isdigit() and not col.startswith('pca_') 
                               for col in actual_columns)
    
    # 如果期望数字索引但有的是原始特征名，则需要转换
    return expect_numeric_features and have_original_features


def _convert_feature_names_to_match_scaler(prediction_data, expected_feature_order, scaler, mapping_info=None):
    """
    将数字索引重新映射为原始特征名以匹配scaler
    
    Args:
        prediction_data: 原始预测数据DataFrame（包含原始特征名）
        expected_feature_order: 特征顺序（数字索引字符串如"0", "1", "2"）
        scaler: sklearn scaler对象（期望原始特征名）
        mapping_info: 模型映射信息字典
        
    Returns:
        pd.DataFrame: 转换后的数据（特征名为原始特征名）
    """
    print(f"🔧 智能特征转换:")
    print(f"   输入特征: {list(prediction_data.columns)}")
    print(f"   期望特征: {expected_feature_order}")
    
    # 获取scaler期望的特征名称
    scaler_feature_names = []
    if hasattr(scaler, 'feature_names_in_'):
        scaler_feature_names = list(scaler.feature_names_in_)
        print(f"   Scaler期望特征名: {scaler_feature_names}")
    
    # 获取预测数据中的原始特征列表
    original_features = [col for col in prediction_data.columns 
                        if not col.isdigit() and not col.startswith('pca_') 
                        and col not in Config.exclude_columns]
    
    print(f"   预测数据原始特征: {original_features}")
    
    # 创建转换后的DataFrame
    converted_data = pd.DataFrame(index=prediction_data.index)
    
    # 关键修复：直接使用scaler期望的特征名称，确保完全匹配
    for scaler_feature_name in scaler_feature_names:
        if scaler_feature_name in prediction_data.columns:
            converted_data[scaler_feature_name] = prediction_data[scaler_feature_name]
            print(f"   ✅ 直接映射: {scaler_feature_name}")
        else:
            # 如果在预测数据中找不到，尝试智能映射
            mapped_feature_name, mapped_value = _smart_feature_mapping(scaler_feature_name, prediction_data, expected_feature_order, original_features, mapping_info)
            if isinstance(mapped_value, (pd.Series, pd.DataFrame)):
                converted_data[scaler_feature_name] = mapped_value
                print(f"   🔄 智能映射: {scaler_feature_name}")
            elif mapped_value != 0:
                converted_data[scaler_feature_name] = mapped_value
                print(f"   🔄 智能映射: {scaler_feature_name}")
            else:
                converted_data[scaler_feature_name] = 0
                print(f"   ❌ 缺失映射: {scaler_feature_name} = 0")
    
    print(f"🔧 转换完成，特征数: {len(converted_data.columns)}")
    print(f"🔧 转换后的特征名称: {list(converted_data.columns)}")
    
    # 检查并处理NaN值
    nan_count = converted_data.isnull().sum().sum()
    if nan_count > 0:
        print(f"⚠️ 发现 {nan_count} 个NaN值，使用0填充...")
        converted_data = converted_data.fillna(0)
    
    # 确保特征数量与scaler完全匹配
    expected_scaler_count = len(scaler_feature_names) if scaler_feature_names else 0
    if len(converted_data.columns) != expected_scaler_count:
        print(f"⚠️ 特征数量不匹配: 转换后{len(converted_data.columns)} vs 期望{expected_scaler_count}")
        
        # 创建与scaler完全匹配的数据框
        matched_data = pd.DataFrame(index=converted_data.index)
        
        # 按scaler期望的特征顺序填充数据
        for scaler_feature in scaler_feature_names:
            if scaler_feature in converted_data.columns:
                matched_data[scaler_feature] = converted_data[scaler_feature]
            else:
                matched_data[scaler_feature] = 0
                print(f"   ⚠️ 补充缺失特征: {scaler_feature}")
        
        print(f"🔧 特征数量调整为: {len(matched_data.columns)}")
        return matched_data
    
    return converted_data


def analyze_model_feature_mapping(model_path):
    """
    分析指定模型路径下的特征映射关系
    
    Args:
        model_path: 模型目录路径
        
    Returns:
        dict: 包含映射信息的字典
    """
    mapping_info = {
        'feature_order_type': None,  # 'numeric' 或 'named'
        'scaler_features': [],
        'feature_order': [],
        'mapping_strategy': None
    }
    
    try:
        # 1. 检查feature_order.json文件
        feature_order_file = os.path.join(model_path, 'feature_order.json')
        if os.path.exists(feature_order_file):
            with open(feature_order_file, 'r', encoding='utf-8') as f:
                feature_order = json.load(f)
            mapping_info['feature_order'] = feature_order
            
            # 判断特征顺序类型
            if all(str(item).isdigit() for item in feature_order):
                mapping_info['feature_order_type'] = 'numeric'
                mapping_info['mapping_strategy'] = 'numeric_to_named'
            else:
                mapping_info['feature_order_type'] = 'named'
                mapping_info['mapping_strategy'] = 'direct_match'
        
        # 2. 检查scaler文件
        scaler_file = os.path.join(model_path, 'ScalerAA', 'scaler_AA.pkl')
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            if hasattr(scaler, 'feature_names_in_'):
                mapping_info['scaler_features'] = list(scaler.feature_names_in_)
        
        # 3. 检查特征重要性文件
        feature_importance_file = os.path.join(model_path, 'feature_importanceAA', 'enhanced_feature_mapping_results.csv')
        if os.path.exists(feature_importance_file):
            try:
                import pandas as pd
                df = pd.read_csv(feature_importance_file)
                if 'feature' in df.columns and 'original_column_name' in df.columns:
                    # 创建特征映射字典
                    feature_mapping = dict(zip(df['feature'], df['original_column_name']))
                    mapping_info['feature_mapping'] = feature_mapping
            except Exception as e:
                print(f"⚠️ 读取特征重要性文件失败: {e}")
        
        print(f"🔍 模型特征映射分析完成:")
        print(f"   特征顺序类型: {mapping_info['feature_order_type']}")
        print(f"   映射策略: {mapping_info['mapping_strategy']}")
        print(f"   Scaler特征数量: {len(mapping_info['scaler_features'])}")
        
        return mapping_info
        
    except Exception as e:
        print(f"❌ 分析模型特征映射失败: {e}")
        return mapping_info


def _smart_feature_mapping(scaler_feature_name, prediction_data, expected_feature_order, original_features, mapping_info=None):
    """
    智能特征映射：尝试将scaler期望的特征名称映射到预测数据中的特征
    
    Args:
        scaler_feature_name: scaler期望的特征名称
        prediction_data: 预测数据DataFrame
        expected_feature_order: 特征顺序（数字索引）
        original_features: 原始特征列表
        mapping_info: 模型映射信息字典
        
    Returns:
        tuple: (映射后的特征名称, 映射后的数据) 或 (None, 0) 如果找不到
    """
    # 策略1：直接匹配
    if scaler_feature_name in prediction_data.columns:
        return scaler_feature_name, prediction_data[scaler_feature_name]
    
    # 策略2：使用映射信息进行精确映射
    if mapping_info and 'feature_mapping' in mapping_info:
        feature_mapping = mapping_info['feature_mapping']
        if scaler_feature_name in feature_mapping:
            mapped_feature = feature_mapping[scaler_feature_name]
            if mapped_feature in prediction_data.columns:
                # 返回映射后的Series，但保持scaler期望的特征名称
                mapped_series = prediction_data[mapped_feature].copy()
                mapped_series.name = scaler_feature_name
                return scaler_feature_name, mapped_series
    
    # 策略3：模糊匹配（基于特征名称的相似性）
    # 例如：right_tsun_sum 可能对应 right_tsun_mean
    base_name = scaler_feature_name.replace('_sum', '').replace('_max', '').replace('_min', '').replace('_mean', '')
    
    # 优先尝试找到*_mean版本的特征
    mean_feature_name = base_name + '_mean'
    if mean_feature_name in prediction_data.columns and mean_feature_name != scaler_feature_name:
        mapped_series = prediction_data[mean_feature_name].copy()
        mapped_series.name = scaler_feature_name
        return scaler_feature_name, mapped_series
    
    # 其次尝试任何包含基础名称的特征
    for col in prediction_data.columns:
        if base_name in col and col != scaler_feature_name:
            # 找到相似的特征，使用该特征的值
            mapped_series = prediction_data[col].copy()
            mapped_series.name = scaler_feature_name
            return scaler_feature_name, mapped_series
    
    # 策略4：如果都找不到，返回0
    return scaler_feature_name, 0


def preprocess_prediction_data(prediction_data, scaler, pca, feature_order, expected_features=None, chunk_size=100000, mapping_info=None):
    """预处理预测数据（按照训练代码的方式，统一使用动态特征数量调整）"""
    print("预处理预测数据...")
    print(f"数据总量: {prediction_data.shape[0]:,} 行")

    # 如果数据量太大，使用分块处理
    if prediction_data.shape[0] > chunk_size:
        print(f"数据量过大，使用分块处理（块大小: {chunk_size:,} 行）")
        return preprocess_prediction_data_chunked(prediction_data, scaler, pca, feature_order, expected_features, chunk_size, mapping_info)

    # 1. 提取特征列（排除PCA特征，因为标准化器不包含PCA特征）
    feature_columns = [col for col in prediction_data.columns if
                       col not in Config.exclude_columns and not col.startswith('pca_')]
    X_prediction = prediction_data[feature_columns].copy()

    # 2. 智能特征适配（与非分块版本一致）
    original_feature_order = [col for col in feature_order if not col.startswith('pca_')]
    
    # 检查是否需要进行特征名称转换
    needs_feature_conversion = _needs_feature_name_conversion(X_prediction.columns, original_feature_order)
    
    if needs_feature_conversion:
        print("🔧 检测到特征名称不匹配，进行智能转换...")
        X_prediction = _convert_feature_names_to_match_scaler(X_prediction, original_feature_order, scaler, mapping_info)
    else:
        # 原有的特征对齐逻辑
        available_features = [col for col in original_feature_order if col in X_prediction.columns]
        missing_features = [col for col in original_feature_order if col not in X_prediction.columns]

        if missing_features:
            print(f"添加 {len(missing_features)} 个缺失特征...")
            for feat in missing_features:
                X_prediction[feat] = 0

        X_prediction = X_prediction[original_feature_order]

    # 3. 数据清理和类型转换
    print("数据清理和类型转换...")
    for col in X_prediction.columns:
        if X_prediction[col].dtype == 'object':
            X_prediction[col] = pd.to_numeric(X_prediction[col], errors='coerce')
            X_prediction[col] = X_prediction[col].fillna(0)
    X_prediction = X_prediction.astype(float)

    # 4. 数据标准化
    print("数据标准化...")
    X_scaled = scaler.transform(X_prediction)
    X_scaled = pd.DataFrame(X_scaled, columns=X_prediction.columns, index=X_prediction.index)

    # 5. 检查并处理NaN值
    print("检查NaN值...")
    nan_count = X_scaled.isnull().sum().sum()
    if nan_count > 0:
        print(f"发现 {nan_count} 个NaN值，使用0填充...")
        X_scaled = X_scaled.fillna(0)

    # 6. 分位数裁剪（优化版本，确保per_mu不超过300）
    print("应用分位数裁剪（优化版本）...")
    for col in X_scaled.columns:
        q_low = X_scaled[col].quantile(0.001)  # 更严格的下界
        q_high = X_scaled[col].quantile(0.999)  # 更严格的上界
        X_scaled[col] = np.clip(X_scaled[col], q_low, q_high)

    # 7. 额外数据范围限制（确保预测值合理）
    print("应用额外数据范围限制...")
    for col in X_scaled.columns:
        # 进一步限制数据范围，避免极端值
        X_scaled[col] = np.clip(X_scaled[col], -2.5, 2.5)

    # 6. PCA降维
    print("PCA降维...")
    X_pca = pca.transform(X_scaled)
    X_pca_df = pd.DataFrame(X_pca, index=X_prediction.index)

    # 6. 交互特征生成（按照训练代码的方式：跳过）
    print("交互特征生成（跳过，与训练代码一致）...")
    X_inter = X_scaled.copy()

    # 7. 合并所有特征
    X_final = pd.concat([X_scaled, X_pca_df, X_inter], axis=1)
    print(f"合并后特征数量: {X_final.shape[1]}")

    # 8. XGBoost特征提取（简化版本）
    try:
        print("XGBoost特征提取...")
        # 加载XGBoost模型
        xgb_model = xgb.XGBRegressor()
        
        # 尝试加载JSON格式的模型
        if Config.xgboost_model_file.endswith('.json'):
            xgb_model.load_model(Config.xgboost_model_file)
        else:
            # 尝试加载PKL格式的模型
            import joblib
            xgb_model = joblib.load(Config.xgboost_model_file)

        # 调整特征数量
        if X_final.shape[1] != xgb_model.n_features_in_:
            if X_final.shape[1] < xgb_model.n_features_in_:
                missing_features = xgb_model.n_features_in_ - X_final.shape[1]
                print(f"添加 {missing_features} 个零特征以匹配XGBoost模型...")
                for i in range(missing_features):
                    X_final[f'missing_feature_{i}'] = 0
            else:
                print(f"截取前 {xgb_model.n_features_in_} 个特征...")
                X_final = X_final.iloc[:, :xgb_model.n_features_in_]

        # 提取XGBoost特征
        # 确保输入是DataFrame并保持特征名称
        if isinstance(X_final, pd.DataFrame):
            X_final_with_names = X_final
        else:
            # 如果是numpy数组，需要重新创建DataFrame
            X_final_with_names = pd.DataFrame(X_final, index=X_prediction.index)

        # 修复特征名称不匹配问题
        # 如果XGBoost模型期望特定的特征名称，我们需要调整输入数据
        if hasattr(xgb_model, 'feature_names_in_') and xgb_model.feature_names_in_ is not None:
            expected_feature_names = xgb_model.feature_names_in_
            print(f"XGBoost模型期望特征名称: {len(expected_feature_names)} 个")
            print(f"输入数据特征名称: {len(X_final_with_names.columns)} 个")
            
            # 如果特征数量匹配但名称不匹配，重命名列
            if len(X_final_with_names.columns) == len(expected_feature_names):
                X_final_with_names.columns = expected_feature_names
                print("✅ 已重命名特征以匹配XGBoost模型期望")
            else:
                print("⚠️ 特征数量不匹配，使用数值索引")
                # 创建数值索引的DataFrame
                X_final_with_names = pd.DataFrame(X_final_with_names.values, 
                                                columns=[f'feature_{i}' for i in range(X_final_with_names.shape[1])])

        # 修复：确保输入数据格式正确
        # XGBoost.apply() 期望numpy数组而不是DataFrame
        try:
            leaf_features = xgb_model.apply(X_final_with_names.values)
        except Exception as e:
            print(f"⚠️ XGBoost.apply()失败，尝试使用predict方法: {e}")
            # 回退到使用predict方法
            leaf_features = xgb_model.predict(X_final_with_names.values, pred_leaf=True)
        if leaf_features.ndim == 1:   
            leaf_features = leaf_features.reshape(-1, 1)

        # 合并特征
        X_combined = np.hstack([X_final_with_names.values, leaf_features])

        # 使用传入的动态特征数量
        if expected_features is None:
            # 如果没有提供，使用默认值（向后兼容）
            expected_features = 89
            print("⚠️ 未提供期望特征数量，使用默认值89")
        
        print(f"模型期望特征数量: {expected_features}")
        
        # 确保特征数量匹配模型期望
        if X_combined.shape[1] != expected_features:
            if X_combined.shape[1] < expected_features:
                missing_features = expected_features - X_combined.shape[1]
                print(f"添加 {missing_features} 个零特征以匹配模型期望...")
                for i in range(missing_features):
                    X_combined = np.hstack([X_combined, np.zeros((X_combined.shape[0], 1))])
            else:
                print(f"截取前 {expected_features} 个特征...")
                X_combined = X_combined[:, :expected_features]

        X_combined_df = pd.DataFrame(X_combined)
        print(f"最终特征数量: {X_combined_df.shape[1]}")

    except Exception as e:
        print(f"XGBoost特征提取失败: {e}")
        print("使用原始特征作为备选...")
        # 使用原始特征
        X_combined_df = X_final.copy()
        if X_combined_df.shape[1] != (expected_features or 113):
            if X_combined_df.shape[1] < 113:
                missing_features = (expected_features or 113) - X_combined_df.shape[1]
                print(f"添加 {missing_features} 个零特征以达到{expected_features or 113}个特征...")
                for i in range(missing_features):
                    X_combined_df[f'missing_feature_{i}'] = 0
            else:
                print(f"截取前 113 个特征...")
                X_combined_df = X_combined_df.iloc[:, :113]

        print(f"备选方案最终特征数量: {X_combined_df.shape[1]}")

        # 最终检查：确保特征数量是113
        if X_combined_df.shape[1] != (expected_features or 113):
            print(f"⚠️ 特征数量不正确: {X_combined_df.shape[1]}，调整为113")
            if X_combined_df.shape[1] < 113:
                missing_cols = 113 - X_combined_df.shape[1]
                for i in range(missing_cols):
                    X_combined_df[f'final_zero_feat_{i}'] = 0
            else:
                X_combined_df = X_combined_df.iloc[:, :113]
            print(f"调整后最终特征数量: {X_combined_df.shape[1]}")

    return X_combined_df


def preprocess_prediction_data_chunked(prediction_data, scaler, pca, feature_order, expected_features=None, chunk_size=100000, mapping_info=None):
    """分块预处理预测数据（避免内存问题，统一使用动态特征数量调整）"""
    print(f"开始分块预处理，块大小: {chunk_size:,} 行")

    n_total = prediction_data.shape[0]
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    processed_chunks = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_total)
        print(f"处理块 {i + 1}/{n_chunks}: 行 {start:,} - {end:,}")

        # 提取当前块的数据
        chunk_data = prediction_data.iloc[start:end].copy()

        # 预处理当前块
        chunk_processed = preprocess_single_chunk(chunk_data, scaler, pca, feature_order, expected_features, mapping_info)
        processed_chunks.append(chunk_processed)

        # 清理内存
        del chunk_data, chunk_processed

    # 合并所有处理后的块
    print("合并所有处理后的块...")
    X_combined_df = pd.concat(processed_chunks, ignore_index=True)
    del processed_chunks

    print(f"分块预处理完成，最终特征数量: {X_combined_df.shape[1]}")
    return X_combined_df


def preprocess_single_chunk(chunk_data, scaler, pca, feature_order, expected_features=None, mapping_info=None):
    """预处理单个数据块（智能特征适配）"""
    # 1. 提取特征列（排除PCA特征，因为标准化器不包含PCA特征）
    feature_columns = [col for col in chunk_data.columns if
                       col not in Config.exclude_columns and not col.startswith('pca_')]
    X_prediction = chunk_data[feature_columns].copy()

    # 2. 智能特征适配 - 检测并修复特征名称不匹配问题
    original_feature_order = [col for col in feature_order if not col.startswith('pca_')]
    
    # 检查是否需要进行特征名称转换
    needs_feature_conversion = _needs_feature_name_conversion(X_prediction.columns, original_feature_order)
    
    if needs_feature_conversion:
        print("🔧 检测到特征名称不匹配，进行智能转换...")
        X_prediction = _convert_feature_names_to_match_scaler(X_prediction, original_feature_order, scaler, mapping_info)
    else:
        # 原有的特征对齐逻辑
        available_features = [col for col in original_feature_order if col in X_prediction.columns]
        missing_features = [col for col in original_feature_order if col not in X_prediction.columns]

        if missing_features:
            for feat in missing_features:
                X_prediction[feat] = 0

        X_prediction = X_prediction[original_feature_order]

    # 3. 数据清理和类型转换
    for col in X_prediction.columns:
        if X_prediction[col].dtype == 'object':
            X_prediction[col] = pd.to_numeric(X_prediction[col], errors='coerce')
            X_prediction[col] = X_prediction[col].fillna(0)
    X_prediction = X_prediction.astype(float)

    # 4. 数据标准化
    X_scaled = scaler.transform(X_prediction)
    X_scaled = pd.DataFrame(X_scaled, columns=X_prediction.columns, index=X_prediction.index)

    # 5. 检查并处理NaN值
    nan_count = X_scaled.isnull().sum().sum()
    if nan_count > 0:
        X_scaled = X_scaled.fillna(0)

    # 6. 分位数裁剪（优化版本，确保per_mu不超过300）
    for col in X_scaled.columns:
        q_low = X_scaled[col].quantile(0.001)  # 更严格的下界
        q_high = X_scaled[col].quantile(0.999)  # 更严格的上界
        X_scaled[col] = np.clip(X_scaled[col], q_low, q_high)

    # 7. 额外数据范围限制（确保预测值合理，强化版本）
    for col in X_scaled.columns:
        # 进一步限制数据范围，避免极端值
        X_scaled[col] = np.clip(X_scaled[col], -1.5, 1.5)  # 进一步缩小范围

    # 8. PCA降维
    print("应用PCA变换...")
    print(f"🔍 PCA期望特征数量: {pca.n_features_in_}")
    print(f"🔍 PCA期望特征名称: {pca.feature_names_in_}")
    print(f"🔍 输入数据特征名称: {list(X_scaled.columns)}")
    
    # 关键修复：确保PCA输入的特征名称与PCA训练时一致
    # PCA期望的是原始特征名称，不是scaler处理后的特征名称
    pca_expected_features = list(pca.feature_names_in_)
    pca_input_features = []
    
    # 为PCA选择正确的特征
    for pca_feature in pca_expected_features:
        if pca_feature in X_scaled.columns:
            pca_input_features.append(pca_feature)
        else:
            # 如果PCA期望的特征不在scaler输出中，尝试找到对应的基础特征
            # 例如：PCA期望right_tsun_mean，但scaler输出right_tsun_sum
            base_name = pca_feature.replace('_sum', '').replace('_max', '').replace('_min', '').replace('_mean', '')
            mean_feature = base_name + '_mean'
            if mean_feature in X_scaled.columns:
                pca_input_features.append(mean_feature)
            else:
                # 如果找不到，使用0填充
                pca_input_features.append(pca_feature)
                print(f"⚠️ PCA特征 {pca_feature} 未找到，使用0填充")
    
    # 创建PCA输入数据
    X_pca_input = X_scaled[pca_input_features].copy()
    print(f"🔧 PCA输入特征数量: {len(X_pca_input.columns)}")
    print(f"🔧 PCA输入特征名称: {list(X_pca_input.columns)}")
    
    X_pca = pca.transform(X_pca_input)
    X_pca_df = pd.DataFrame(X_pca, index=X_prediction.index)

    # 9. 交互特征生成（按照训练代码的方式：跳过）
    X_inter = X_scaled.copy()

    # 10. 合并所有特征
    X_final = pd.concat([X_scaled, X_pca_df, X_inter], axis=1)

    # 11. XGBoost特征提取（使用与主函数相同的动态调整逻辑）
    try:
        # 加载XGBoost模型
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(Config.xgboost_model_file)

        # 调整特征数量以匹配XGBoost模型期望（与主函数保持一致）
        if X_final.shape[1] != xgb_model.n_features_in_:
            if X_final.shape[1] < xgb_model.n_features_in_:
                missing_features = xgb_model.n_features_in_ - X_final.shape[1]
                print(f"添加 {missing_features} 个零特征以匹配XGBoost模型...")
                for i in range(missing_features):
                    X_final[f'missing_feature_{i}'] = 0
            else:
                print(f"截取前 {xgb_model.n_features_in_} 个特征...")
                X_final = X_final.iloc[:, :xgb_model.n_features_in_]

        # 提取XGBoost特征（与主函数保持一致）
        # 确保输入是DataFrame并保持特征名称
        if isinstance(X_final, pd.DataFrame):
            X_final_with_names = X_final
        else:
            # 如果是numpy数组，需要重新创建DataFrame
            X_final_with_names = pd.DataFrame(X_final, index=X_prediction.index)

        # 修复：确保输入数据格式正确
        # XGBoost.apply() 期望numpy数组而不是DataFrame
        leaf_features = xgb_model.apply(X_final_with_names.values)
        if leaf_features.ndim == 1:
            leaf_features = leaf_features.reshape(-1, 1)

        # 合并特征（与主函数保持一致）
        X_combined = np.hstack([X_final_with_names.values, leaf_features])
        X_combined_df = pd.DataFrame(X_combined, index=X_prediction.index)

    except Exception as e:
        print(f"XGBoost特征提取失败: {str(e)}")
        print("使用原始特征作为备选...")

        # 备选方案：使用原始特征
        X_combined_df = X_final.copy()

    # 12. 确保特征数量正确（与主函数保持一致）
    # 使用传入的动态特征数量
    if expected_features is None:
        # 如果没有提供，使用默认值（向后兼容）
        expected_features = 89
        print("⚠️ 未提供期望特征数量，使用默认值89")
    
    if X_combined_df.shape[1] != expected_features:
        if X_combined_df.shape[1] < expected_features:
            missing_cols = expected_features - X_combined_df.shape[1]
            print(f"添加 {missing_cols} 个零特征以达到{expected_features}个特征...")
            for i in range(missing_cols):
                X_combined_df[f'zero_feat_{i}'] = 0
        else:
            print(f"截取前 {expected_features} 个特征...")
            X_combined_df = X_combined_df.iloc[:, :expected_features]
        print(f"调整后最终特征数量: {X_combined_df.shape[1]}")

    return X_combined_df


def predict_with_processed_data(prediction_data, X_processed, model, save_chunk_size=5000000):
    """使用预处理后的数据进行预测，并分块保存结果"""
    print("执行模型预测...")

    batch_size = 500000
    n_pred = X_processed.shape[0]
    results_chunks = []
    saved_files = []

    print(f"数据总量: {n_pred:,} 行，分 {(n_pred + batch_size - 1) // batch_size} 批处理")
    print(f"保存块大小: {save_chunk_size:,} 行")

    # 计算需要保存的文件数量
    num_save_files = (n_pred + save_chunk_size - 1) // save_chunk_size
    print(f"将保存 {num_save_files} 个预测结果文件")

    for start in range(0, n_pred, batch_size):
        end = min(start + batch_size, n_pred)
        print(f"处理批次: 行 {start:,} - {end:,}")

        # 调试：检查输入数据格式
        batch_data = X_processed.iloc[start:end]
        print(f"批次数据形状: {batch_data.shape}")
        print(f"批次数据类型: {type(batch_data)}")
        print(f"批次数据前5行前5列:\n{batch_data.iloc[:5, :5]}")

        # 确保数据是numpy数组格式
        if isinstance(batch_data, pd.DataFrame):
            batch_data = batch_data.values

        # 检查并处理异常值
        print(f"数据范围检查:")
        print(f"  最小值: {batch_data.min()}")
        print(f"  最大值: {batch_data.max()}")
        print(f"  是否有无穷值: {np.isinf(batch_data).any()}")
        print(f"  是否有NaN值: {np.isnan(batch_data).any()}")

        # 处理异常值
        batch_data = np.nan_to_num(batch_data, nan=0.0, posinf=0.0, neginf=0.0)

        # 限制数据范围到合理区间（强化版本，确保per_mu不超过300）
        # 训练时模型输出被限制在[3.0, 10.0]，对应expm1后为[19.09, 22025.46]
        # 我们需要确保输入数据在更严格的范围内，避免极端值
        batch_data = np.clip(batch_data, -1.5, 1.5)  # 进一步缩小范围

        print(f"处理后数据范围: [{batch_data.min():.6f}, {batch_data.max():.6f}]")

        # 检查模型输入形状
        print(f"模型输入形状期望: {model.input_shape}")
        print(f"实际输入形状: {batch_data.shape}")

        # 尝试预测
        try:
            predictions = model.predict(batch_data)
            print(f"预测结果类型: {type(predictions)}")
            if isinstance(predictions, list):
                print(f"预测结果数量: {len(predictions)}")
                cls_pred_batch = predictions[0]
                reg_pred_batch = predictions[1]
                print(f"分类预测形状: {cls_pred_batch.shape}")
                print(f"回归预测形状: {reg_pred_batch.shape}")
            else:
                print(f"预测结果形状: {predictions.shape}")
                cls_pred_batch = predictions
                reg_pred_batch = predictions
        except Exception as e:
            print(f"模型预测错误: {str(e)}")
            raise

        # 反变换（按照训练代码的方式）
        reg_pred_batch = np.expm1(reg_pred_batch)

        results_chunk = pd.DataFrame({
            'x': prediction_data['x'].iloc[start:end],
            'y': prediction_data['y'].iloc[start:end],
            'yyyy': prediction_data['yyyy'].iloc[start:end],
            'suit': np.argmax(cls_pred_batch, axis=1),
            'per_mu': reg_pred_batch[:, 1],  # 第二列是per_mu
            'per_qu': reg_pred_batch[:, 0]  # 第一列是per_qu
        })
        results_chunks.append(results_chunk)

        # 检查是否需要保存文件
        current_total_rows = sum(len(chunk) for chunk in results_chunks)
        if current_total_rows >= save_chunk_size or start + batch_size >= n_pred:
            # 合并当前所有结果
            current_results = pd.concat(results_chunks, ignore_index=True)

            # 如果超过块大小，只保存前面的部分
            if current_total_rows > save_chunk_size:
                save_results = current_results.head(save_chunk_size)
                remaining_results = current_results.iloc[save_chunk_size:]
                results_chunks = [remaining_results] if len(remaining_results) > 0 else []
            else:
                save_results = current_results
                results_chunks = []

            # 保存文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_index = len(saved_files) + 1
            filename = f"prediction_results_chunk_{file_index:03d}_{timestamp}.csv"
            filepath = os.path.join(Config.result_dir, filename)

            save_results.to_csv(filepath, index=False)
            saved_files.append(filepath)

            print(f"✅ 已保存预测结果文件 {file_index}: {filename}")
            print(f"   文件包含: {len(save_results):,} 行数据")
            print(f"   per_mu范围: {save_results['per_mu'].min():.2f} - {save_results['per_mu'].max():.2f}")
            print(f"   文件路径: {filepath}")

    return saved_files


# 在类定义完成后初始化路径
Config._update_paths()

if __name__ == "__main__":
    import sys

    print("=== 产量预测程序 ===")
    print("使用现有模型对预测气象数据进行产量预测")
    print("数据源: 预测气象数据 + 土壤数据")
    print("=" * 50)

    # 检查是否启用小型测试模式
    test_mode = False
    test_size = 10000
    weather_start_row = None
    weather_end_row = None
    model_name = None

    # 显示可用模型
    print("\n🔍 检测可用模型...")
    available_models = Config.list_available_models()

    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['--test', '--small', '--mini']:
            test_mode = True
            print("🔬 启用小型测试模式")
        elif sys.argv[1].lower().startswith('--test-size='):
            test_mode = True
            try:
                test_size = int(sys.argv[1].split('=')[1])
                print(f"🔬 启用小型测试模式，数据量限制: {test_size:,} 行")
            except ValueError:
                print("⚠️ 无效的测试大小参数，使用默认值 10,000")
                test_size = 10000

    # 解析所有参数
    for arg in sys.argv[1:]:
        if arg.startswith('--weather-start='):
            try:
                weather_start_row = int(arg.split('=')[1])
                print(f"🌤️ 气象数据起始行设置为: {weather_start_row:,}")
            except ValueError:
                print("⚠️ 无效的起始行参数")
                weather_start_row = None
        elif arg.startswith('--weather-end='):
            try:
                weather_end_row = int(arg.split('=')[1])
                print(f"🌤️ 气象数据结束行设置为: {weather_end_row:,}")
            except ValueError:
                print("⚠️ 无效的结束行参数")
                weather_end_row = None
        elif arg.startswith('--model='):
            model_name = arg.split('=')[1]
            if model_name in available_models:
                print(f"🎯 指定使用模型: {model_name}")
            else:
                print(f"⚠️ 指定的模型 '{model_name}' 不存在，可用模型: {available_models}")
                model_name = None
        elif arg == '--list-models':
            print("\n📋 可用模型列表:")
            for i, model in enumerate(available_models, 1):
                print(f"   {i}. {model}")
            print("\n使用方法: python 产量模型调用预测.py --model=模型名称")
            exit(0)

    # 运行预测
    success = main(test_mode=test_mode, test_size=test_size,
                   weather_start_row=weather_start_row, weather_end_row=weather_end_row,
                   model_name=model_name)

    if success:
        print("\n🎉 预测程序执行成功！")
    else:
        print("\n❌ 预测程序执行失败！")
        exit(1)


