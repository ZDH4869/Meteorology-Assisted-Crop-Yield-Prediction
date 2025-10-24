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

# ================= 深度学习库 =================
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, Layer, Add
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
    product_data_csv = os.path.join(input_data_dir, "maize_test_1013.csv")

    # 气象数据配置 - 文件夹模式
    weather_data_folder = r"2-产量预测代码/train+text_csv数据/训练气象"  # 训练气象数据文件夹路径
    use_weather_folder = True  # 使用文件夹模式读取训练气象数据

    soil_data_csv = os.path.join(input_data_dir, "soil_test.csv")

    # 输出目录配置
    output_dir = os.path.join(base_dir, "model_test_06")
    model_dir = os.path.join(output_dir, "modelAA")
    final_model_dir = os.path.join(output_dir, "modelAA")
    label_encoder_dir = os.path.join(output_dir, "LabelEnconderAA")
    scaler_dir = os.path.join(output_dir, "ScalerAA")
    xgboost_dir = os.path.join(output_dir, "XGBoostAA")
    logs_dir = os.path.join(output_dir, "final_logsAA")
    result_dir = os.path.join(output_dir, "result_predited_csvAA")
    feature_importance_dir = os.path.join(output_dir, "feature_importanceAA")
    result_analysis_dir = os.path.join(output_dir, "analysis_result_mapAA")  # 应该输出2
    training_analysis_dir = os.path.join(output_dir, "analysis_training_mapAA")  # 应该输出4

    # 具体文件路径 max_rows_per_weather_file sample_size
    label_encoder_file = os.path.join(label_encoder_dir, "label_encoder_AA.pkl")
    scaler_file = os.path.join(scaler_dir, "scaler_AA.pkl")
    xgboost_model_file = os.path.join(xgboost_dir, "xgb_model_AA.json")
    final_model_file = os.path.join(final_model_dir, "final_model_r2_dynamic.h5")  # 动态文件名，将在训练时更新
    result_file = os.path.join(result_dir, "result_predited_AA.csv")
    feature_importance_file = os.path.join(feature_importance_dir, "feature_importance_AA.csv")
    key_features_list_file = os.path.join(feature_importance_dir, "key_features_AA.json")
    feature_importance_plot = os.path.join(feature_importance_dir, "feature_importance_plot_AA.png")
    suitability_map_file = os.path.join(result_analysis_dir, "result_suitability_map_AA.png")  #
    feature_importance_map_file = os.path.join(result_analysis_dir, "feature_importance.png")  #
    evaluation_file = os.path.join(training_analysis_dir, "evaluation_report.json")  #
    confusion_matrix_file = os.path.join(training_analysis_dir, "confusion_matrix.png")  #
    accuracy_loss_map_file = os.path.join(training_analysis_dir, "Accuracy+Loss_map_AA.png")  # 只有这一个输出
    regression_scatter_file = os.path.join(training_analysis_dir, "regression_scatter.png")  #

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
    
    # ============= 训练数据量控制 =============
    # 一键设置最终训练和验证集大小（预处理后）
    enable_data_sampling = True  # 是否启用数据采样
    max_train_samples = 5000    # 大幅增加训练集样本数，提高模型性能
    max_val_samples = 1000      # 增加验证集样本数，提高评估准确性
    sampling_strategy ='first_n' # 改为随机采样，提高数据多样性 'stratified' 'first_n'
    
    # ============= 快速预设配置 =============
    # 取消注释以下任一配置来快速设置数据量
    # 配置1: 快速测试（1万训练+2千验证）
    # max_train_samples = 10000
    # max_val_samples = 2000
    
    # 配置2: 中等规模（5万训练+1万验证）
    # max_train_samples = 50000
    # max_val_samples = 10000
    
    # 配置3: 大规模（10万训练+2万验证）
    # max_train_samples = 100000
    # max_val_samples = 20000
    
    # 配置4: 使用全部数据（禁用采样）
    # enable_data_sampling = False

    # XGBoost参数（稳定性优化配置）
    xgb_params = {
        'max_depth': 2,  # 增加深度，提高学习能力
        'learning_rate': 0.1,  # 降低学习率，提高稳定性
        'n_estimators': 15,  # 增加树数量，提高性能
        'subsample': 0.5,  # 降低子采样率，防止过拟合
        'colsample_bytree': 0.5,  # 降低特征采样率，防止过拟合
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

    # 深度学习参数（稳定性优化配置） 代码设置了R²>0.7时，保存训练过程的模型
    dl_params = {
        'epochs': 20,  # 大幅增加训练轮次，确保充分学习
        'batch_size': 16,  # 进一步减小batch size，提高训练稳定性
        'learning_rate': 0.005,  # 提高学习率，加快收敛
        'dropout_rate': 0.2,  # 进一步降低dropout，提高模型学习能力
        'early_stop_patience': 10,  # 大幅增加早停耐心，给模型更多学习时间
        'l2_reg': 1e-5,  # 大幅降低L2正则化强度
        'l1_reg': 1e-6,  # 大幅降低L1正则化强度
        'min_delta': 0.00001,  # 大幅降低最小改善阈值，更敏感
        'reduce_lr_patience': 8,  # 增加学习率衰减耐心
        'reduce_lr_factor': 0.5,  # 更激进的学习率衰减
        'min_lr': 1e-8,  # 进一步降低最小学习率
        'min_r2': 0.7,  # R²阈值设置为0.7
        'validation_split': 0.2,  # 验证集比例
        'verbose': 1  # 详细输出
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
        'n_trials': 3,  # 进一步增加试验次数，找到更好的参数
        'timeout': 600,  # 大幅增加超时时间，允许充分训练
        'param_ranges': {
            'lr': (0.001, 0.02),  # 进一步提高学习率范围，加快收敛
            'neurons1': (64, 256),  # 增加神经元数量，提高模型容量
            'neurons2': (32, 128),  # 增加神经元数量
            'dropout_rate': (0.05, 0.2),  # 进一步降低dropout范围，提高学习能力
            'batch_size': [8, 16, 32],  # 增加更多batch size选项
            'attention_units': (16, 64),  # 增加attention units
            'l1_lambda': (1e-8, 1e-6),  # 进一步降低L1正则化范围
            'l2_lambda': (1e-8, 1e-6),  # 进一步降低L2正则化范围
            'optimizer_type': ['adam'],  # TensorFlow 2.10.1只支持Adam
            'activation': ['relu', 'gelu', 'swish'],  # 添加更多激活函数
            'loss_type': ['mse', 'huber', 'mae']  # 添加更多损失函数
        }
    }

    # ============= 特征重要性配置（防过拟合优化） =============
    feature_importance = {
        'threshold': 0.1,  # 进一步提高阈值，减少特征数量
        'sample_size': 5000,  # 减少采样数量，适合小数据集
        'save_plots': True,
        'min_features': 10,  # 进一步减少最小特征数
        'max_features': 15  # 进一步减少最大特征数，防止过拟合
    }

    # 集成学习配置（速度优化）
    ensemble_params = {
        'n_splits': 2,  # 保持较少的交叉验证折数，加快训练
        'n_models': 1,  # 保持单模型，减少训练时间
        'voting': 'soft',
        'weights': None,
        'bootstrap': True,  # 禁用bootstrap采样，加快训练
        'bootstrap_ratio': 0.5  # 降低bootstrap采样比例
    }

    # 数据增强配置（速度优化）``
    augmentation_params = {
        'augmentation_factor': 0.1,  # 减少数据增强比例，加快训练
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
        ]

        # 验证训练气象数据文件夹
        if not os.path.exists(Config.weather_data_folder):
            raise FileNotFoundError(f"训练气象数据文件夹不存在: {Config.weather_data_folder}")
        # 检查文件夹中是否有CSV文件
        csv_files = [f for f in os.listdir(Config.weather_data_folder) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"训练气象数据文件夹中没有CSV文件: {Config.weather_data_folder}")
        print(f"找到 {len(csv_files)} 个训练气象数据CSV文件")

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
    # 限制每个气象文件读取的最大行数（基于V3优化结果）
    max_rows_per_weather_file = 5000 # 减少到500万行，避免过度采样

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
        batch_size = 50000  # 增大批次大小，提高处理效率
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


def load_weather_data_from_folder(folder_path):
    """
    从指定文件夹中读取所有气象数据CSV文件并合并
    使用Config.max_rows_per_weather_file参数限制每个文件读取的行数

    Args:
        folder_path (str): 气象数据文件夹路径

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
    print(f"每个文件最多读取 {Config.max_rows_per_weather_file:,} 行数据")

    # 读取并合并所有CSV文件
    weather_data_list = []

    for csv_file in tqdm(csv_files, desc="读取气象数据文件"):
        file_path = os.path.join(folder_path, csv_file)
        try:
            # 获取文件总行数（快速统计）
            print(f"正在读取文件: {csv_file}")
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

            except Exception as e:
                print(f"  警告: 无法统计文件行数，使用默认限制: {str(e)}")
                rows_to_read = Config.max_rows_per_weather_file

            # 读取指定行数的数据
            try:
                df = pd.read_csv(file_path, encoding='utf-8', nrows=rows_to_read)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='gbk', nrows=rows_to_read)
                except UnicodeDecodeError:
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
        def spatial_merge(left, right, on=None, tolerance=50000):
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

                # 强制垃圾回收
                import gc
                gc.collect()

            if merged_list:
                print(f"合并完成，共 {len(common_years)} 个年份的数据")
                return pd.concat(merged_list, ignore_index=True)
            else:
                print("没有成功合并任何数据")
                return pd.DataFrame()

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

        # ============= 关键修复：在数据分割前进行log1p变换 =============
        print("\n🔧 关键修复：在数据分割前进行log1p变换，确保训练集和验证集使用相同的数据预处理...")
        
        # 对目标变量进行log1p变换（在数据分割之前）
        print("对目标变量进行log1p变换...")
        data_train[['per_qu', 'per_mu']] = np.log1p(data_train[['per_qu', 'per_mu']])
        
        # 检查log1p变换后的数据质量
        print(f"log1p变换后目标变量统计信息:")
        print(f"per_mu: min={data_train['per_mu'].min():.4f}, max={data_train['per_mu'].max():.4f}, mean={data_train['per_mu'].mean():.4f}")
        print(f"per_qu: min={data_train['per_qu'].min():.4f}, max={data_train['per_qu'].max():.4f}, mean={data_train['per_qu'].mean():.4f}")
        
        # 对log1p变换后的数据进行轻微裁剪，避免极值
        for col in ['per_qu', 'per_mu']:
            q_low = data_train[col].quantile(0.0005)
            q_high = data_train[col].quantile(0.9995)
            print(f"{col} log1p后裁剪: [{q_low:.4f}, {q_high:.4f}]")
            data_train[col] = np.clip(data_train[col], q_low, q_high)

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
        
        # ============= 一键数据量控制 =============
        if Config.enable_data_sampling:
            print(f"\n🔧 启用数据采样控制...")
            print(f"配置: 最大训练集={Config.max_train_samples}, 最大验证集={Config.max_val_samples}")
            print(f"采样策略: {Config.sampling_strategy}")
            
            # 采样训练集
            if Config.max_train_samples is not None and len(data_train_final) > Config.max_train_samples:
                print(f"训练集采样: {len(data_train_final):,} -> {Config.max_train_samples:,}")
                
                if Config.sampling_strategy == 'random':
                    # 随机采样
                    data_train_final = data_train_final.sample(
                        n=Config.max_train_samples, 
                        random_state=Config.random_seed
                    ).reset_index(drop=True)
                elif Config.sampling_strategy == 'stratified' and 'suit' in data_train_final.columns:
                    # 分层采样
                    from sklearn.model_selection import train_test_split
                    data_train_final, _ = train_test_split(
                        data_train_final,
                        train_size=Config.max_train_samples,
                        random_state=Config.random_seed,
                        stratify=data_train_final['suit']
                    )
                else:
                    # 取前N个样本
                    data_train_final = data_train_final.head(Config.max_train_samples).reset_index(drop=True)
                
                print(f"✅ 训练集采样完成: {data_train_final.shape}")
            
            # 采样验证集
            if Config.max_val_samples is not None and len(data_val) > Config.max_val_samples:
                print(f"验证集采样: {len(data_val):,} -> {Config.max_val_samples:,}")
                
                if Config.sampling_strategy == 'random':
                    # 随机采样
                    data_val = data_val.sample(
                        n=Config.max_val_samples, 
                        random_state=Config.random_seed
                    ).reset_index(drop=True)
                elif Config.sampling_strategy == 'stratified' and 'suit' in data_val.columns:
                    # 分层采样
                    from sklearn.model_selection import train_test_split
                    data_val, _ = train_test_split(
                        data_val,
                        train_size=Config.max_val_samples,
                        random_state=Config.random_seed,
                        stratify=data_val['suit']
                    )
                else:
                    # 取前N个样本
                    data_val = data_val.head(Config.max_val_samples).reset_index(drop=True)
                
                print(f"✅ 验证集采样完成: {data_val.shape}")
            
            print(f"🎯 最终数据量: 训练集={data_train_final.shape[0]:,}, 验证集={data_val.shape[0]:,}")
        else:
            print(f"📊 使用全部数据: 训练集={data_train_final.shape[0]:,}, 验证集={data_val.shape[0]:,}")

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

        # === 检查目标变量数据质量（拆分后，已经是log1p变换后的） ===
        print(f"拆分后目标变量统计信息（已经是log1p变换后的）:")
        print(
            f"per_mu: min={y_reg_train['per_mu'].min():.4f}, max={y_reg_train['per_mu'].max():.4f}, mean={y_reg_train['per_mu'].mean():.4f}")
        print(
            f"per_qu: min={y_reg_train['per_qu'].min():.4f}, max={y_reg_train['per_qu'].max():.4f}, mean={y_reg_train['per_qu'].mean():.4f}")
        print(f"suit: unique values={y_cls_train.unique()}")

        # 注意：log1p变换已经在数据分割前完成，这里不需要再次变换

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
        # 提取验证集标签（已经是log1p变换后的）
        y_reg_val = data_val[['per_qu', 'per_mu']].iloc[:len(X_val_scaled)]
        
        # 返回标准格式：(X_train, X_val, y_reg_train, y_reg_val, data_val)
        return X_train_scaled, X_val_scaled, y_reg_train, y_reg_val, data_val

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
        return cls(**config)

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
        # log1p变换后的数据范围: per_mu [4.64, 5.08], per_qu [7.13, 7.57]
        regression_output = Dense(2, activation='linear',
                                  kernel_initializer='glorot_uniform',
                                  bias_initializer=tf.keras.initializers.Constant([7.28, 4.79]))(
            regression_branch)  # 调整初始值，更接近log1p变换后的数据 [per_qu, per_mu]

        # 添加输出约束：限制预测值在log1p变换后的合理范围内
        # log1p变换后的真实值范围: per_mu [4.64, 5.08], per_qu [7.13, 7.57]
        # 放宽约束范围，避免过度限制模型学习
        regression_output = tf.keras.layers.Lambda(
            lambda x: tf.clip_by_value(x, 4.0, 8.0),  # 扩大范围，覆盖log1p变换后的数据
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


class DynamicModelCheckpoint(tf.keras.callbacks.Callback):
    """动态保存模型，文件名包含R²值"""
    
    def __init__(self, trial_number, model_dir):
        super().__init__()
        self.trial_number = trial_number
        self.model_dir = model_dir
        self.saved_models = []
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # 获取R²值
        r2 = logs.get('val_r2', 0.0)
        
        # 创建包含R²的文件名
        filename = f'model_trial_{self.trial_number}_epoch_{epoch+1}_r2_{r2:.4f}.h5'
        filepath = os.path.join(self.model_dir, filename)
        
        # 保存模型
        try:
            self.model.save(filepath)
            self.saved_models.append({
                'epoch': epoch + 1,
                'r2': r2,
                'filepath': filepath
            })
            print(f"💾 模型已保存: {filename} (R²={r2:.4f})")
        except Exception as e:
            print(f"❌ 保存模型失败: {e}")


class DynamicFinalModelCheckpoint(tf.keras.callbacks.Callback):
    """动态保存最终模型，文件名包含R²值"""
    
    def __init__(self, model_dir, monitor='val_r2', mode='max'):
        super().__init__()
        self.model_dir = model_dir
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_model_path = None
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # 获取监控指标
        current_score = logs.get(self.monitor, float('inf'))
        
        # 获取R²值
        r2 = logs.get('val_r2', 0.0)
        
        # 检查是否是最佳模型
        is_best = False
        if self.mode == 'min':
            if current_score < self.best_score:
                self.best_score = current_score
                is_best = True
        else:  # mode == 'max'
            if current_score > self.best_score:
                self.best_score = current_score
                is_best = True
        
        # 如果是最佳模型，保存它
        if is_best:
            # 删除之前的模型文件
            if self.best_model_path and os.path.exists(self.best_model_path):
                try:
                    os.remove(self.best_model_path)
                except:
                    pass
            
            # 创建新的模型文件名
            filename = f'final_model_r2_{r2:.4f}.h5'
            self.best_model_path = os.path.join(self.model_dir, filename)
            
            # 保存模型
            try:
                self.model.save(self.best_model_path)
                print(f"💾 最佳模型已保存: {filename} (R²={r2:.4f}, {self.monitor}={current_score:.6f})")
            except Exception as e:
                print(f"❌ 保存最佳模型失败: {e}")


class OptunaR2EarlyStopping(tf.keras.callbacks.Callback):
    """Optuna R²早停回调，当R²达到0.7时停止调参训练"""

    def __init__(self, X_val, y_val, min_r2=0.7, trial=None):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.min_r2 = min_r2
        self.trial = trial
        self.best_r2 = 0.0
        self.stopped_epoch = 0
        self.saved_models = []  # 记录所有保存的模型

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
        
        # 将R²值写入logs，供ModelCheckpoint使用
        logs['val_r2'] = r2
        
        # 调试信息：确认val_r2已写入logs
        if epoch < 3:  # 只在前几个epoch打印调试信息
            print(f"🔍 调试: logs['val_r2'] = {logs.get('val_r2', 'NOT_FOUND')}")

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
                print(f"🎯 Trial {self.trial.number} 已标记为R²达到0.7")

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
                'regression': y_reg_train_final
            },
            validation_data=(
                X_val_combined,
                {
                    'classification': y_cls_val,
                    'regression': y_reg_val
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
    try:
        # 创建子图
        plt.figure(figsize=(15, 10))
        
        # 检查可用的键
        available_keys = list(history.history.keys())
        print(f"🔍 可用的训练历史键: {available_keys}")

        # 子图1：分类损失 vs 回归损失
        plt.subplot(2, 2, 1)
        if 'classification_loss' in available_keys:
            plt.plot(history.history['classification_loss'], label='Train Cls Loss')
        if 'val_classification_loss' in available_keys:
            plt.plot(history.history['val_classification_loss'], label='Val Cls Loss')
        if 'regression_loss' in available_keys:
            plt.plot(history.history['regression_loss'], label='Train Yield Loss')
        if 'val_regression_loss' in available_keys:
            plt.plot(history.history['val_regression_loss'], label='Val Yield Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Classification & Regression Loss')
        plt.legend()
        plt.grid(True)

        # 子图2：分类准确度 vs 回归MAE
        plt.subplot(2, 2, 2)
        if 'classification_accuracy' in available_keys:
            plt.plot(history.history['classification_accuracy'], label='Train Cls Acc')
        if 'val_classification_accuracy' in available_keys:
            plt.plot(history.history['val_classification_accuracy'], label='Val Cls Acc')
        if 'regression_mae' in available_keys:
            plt.plot(history.history['regression_mae'], label='Train Yield MAE')
        if 'val_regression_mae' in available_keys:
            plt.plot(history.history['val_regression_mae'], label='Val Yield MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy / MAE')
        plt.title('Classification Accuracy & Yield MAE')
        plt.legend()
        plt.grid(True)

        # 子图3：总损失
        plt.subplot(2, 2, 3)
        if 'loss' in available_keys:
            plt.plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in available_keys:
            plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Total Loss')
        plt.title('Total Loss')
        plt.grid(True)
        plt.legend()

        # 子图4：MSE
        plt.subplot(2, 2, 4)
        if 'regression_mse' in available_keys:
            plt.plot(history.history['regression_mse'], label='Train MSE')
        if 'val_regression_mse' in available_keys:
            plt.plot(history.history['val_regression_mse'], label='Val MSE')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('Mean Squared Error')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        
        # 保存图表
        plot_file = os.path.join(Config.logs_dir, "training_history_plot.png")
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 训练历史图表已保存至: {plot_file}")
        
    except Exception as e:
        print(f"⚠️ 绘制训练历史图表时出错: {str(e)}")
        print("跳过训练历史图表生成")
        plt.close()


def plot_prediction_maps(
        X_val: Union[pd.DataFrame, np.ndarray],
        cls_pred: np.ndarray,
        reg_pred: np.ndarray,
        val_ids: np.ndarray,
        data_val: Union[pd.DataFrame, np.ndarray]
) -> None:
    try:
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val)
        
        # 自动适配列名
        x_col = next((c for c in ['x', 'x_product', 'right_x', 'X'] if c in X_val.columns), None)
        y_col = next((c for c in ['y', 'y_product', 'right_y', 'Y'] if c in X_val.columns), None)
        
        if not (x_col and y_col):
            print(f"⚠️ X_val中未找到有效的坐标列，可用列: {X_val.columns.tolist()}")
            print("尝试从data_val中查找坐标信息...")
            
            if isinstance(data_val, pd.DataFrame):
                x_col = next((c for c in ['x', 'x_product', 'right_x', 'X'] if c in data_val.columns), None)
                y_col = next((c for c in ['y', 'y_product', 'right_y', 'Y'] if c in data_val.columns), None)
                
                if x_col and y_col:
                    print(f"✅ 在data_val中找到坐标列: {x_col}, {y_col}")
                    # 使用data_val中的坐标信息
                    X_val[x_col] = data_val[x_col].values
                    X_val[y_col] = data_val[y_col].values
                else:
                    print(f"❌ data_val中也没有找到有效的坐标列: {data_val.columns.tolist()}")
                    print("跳过地图绘制")
                    return
        
        if not (x_col and y_col):
            raise ValueError("无法找到有效的坐标列进行地图绘制")
        
        print(f"🔍 使用坐标列: x={x_col}, y={y_col}")
        
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
        print(f"✅ 预测地图已保存至: {Config.suitability_map_file}")
        
    except Exception as e:
        print(f"❌ 绘制预测地图时出错: {str(e)}")
        print("地图绘制失败，但不会影响其他功能")
        import traceback
        traceback.print_exc()


def ensure_log1p_consistency(predictions, stage="预测"):
    """确保预测结果与训练时的log1p变换保持一致"""
    if isinstance(predictions, (list, tuple)) and len(predictions) == 2:
        cls_pred, reg_pred = predictions
    else:
        cls_pred = predictions
        reg_pred = None
    
    if reg_pred is not None:
        print(f"🔍 {stage}数据一致性检查:")
        print(f"回归预测值范围: [{np.min(reg_pred):.4f}, {np.max(reg_pred):.4f}]")
        print(f"回归预测值均值: {np.mean(reg_pred):.4f}")
        
        # 检查是否在log1p变换后的合理范围内
        log1p_range_per_mu = (4.6, 5.1)
        log1p_range_per_qu = (7.1, 7.6)
        
        if reg_pred.shape[1] >= 2:
            per_mu_pred = reg_pred[:, 0]
            per_qu_pred = reg_pred[:, 1]
            
            if (np.min(per_mu_pred) < log1p_range_per_mu[0] or np.max(per_mu_pred) > log1p_range_per_mu[1]):
                print(f"⚠️ per_mu预测值超出log1p变换后的合理范围 {log1p_range_per_mu}")
            if (np.min(per_qu_pred) < log1p_range_per_qu[0] or np.max(per_qu_pred) > log1p_range_per_qu[1]):
                print(f"⚠️ per_qu预测值超出log1p变换后的合理范围 {log1p_range_per_qu}")
    
    return cls_pred, reg_pred


def debug_data_transformation(y_true, y_pred, stage="训练"):
    """调试数据变换，确保训练和预测使用相同的数据格式"""
    print(f"\n🔍 {stage}数据变换调试:")
    print(f"真实值范围: [{np.min(y_true):.4f}, {np.max(y_true):.4f}]")
    print(f"预测值范围: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
    print(f"真实值均值: {np.mean(y_true):.4f}, 标准差: {np.std(y_true):.4f}")
    print(f"预测值均值: {np.mean(y_pred):.4f}, 标准差: {np.std(y_pred):.4f}")
    
    # 检查数据是否在log1p变换后的合理范围内
    log1p_range_per_mu = (4.6, 5.1)  # log1p变换后的per_mu范围
    log1p_range_per_qu = (7.1, 7.6)  # log1p变换后的per_qu范围
    
    if len(y_true.shape) > 1 and y_true.shape[1] >= 2:
        per_qu_true = y_true[:, 0]  # per_qu是第一列
        per_mu_true = y_true[:, 1]   # per_mu是第二列
        per_qu_pred = y_pred[:, 0]   # per_qu是第一列
        per_mu_pred = y_pred[:, 1]   # per_mu是第二列
        
        print(f"per_mu真实值范围: [{np.min(per_mu_true):.4f}, {np.max(per_mu_true):.4f}]")
        print(f"per_mu预测值范围: [{np.min(per_mu_pred):.4f}, {np.max(per_mu_pred):.4f}]")
        print(f"per_qu真实值范围: [{np.min(per_qu_true):.4f}, {np.max(per_qu_true):.4f}]")
        print(f"per_qu预测值范围: [{np.min(per_qu_pred):.4f}, {np.max(per_qu_pred):.4f}]")
        
        # 检查是否在合理范围内
        if (np.min(per_mu_true) < log1p_range_per_mu[0] or np.max(per_mu_true) > log1p_range_per_mu[1]):
            print(f"⚠️ per_mu真实值超出log1p变换后的合理范围 {log1p_range_per_mu}")
        if (np.min(per_qu_true) < log1p_range_per_qu[0] or np.max(per_qu_true) > log1p_range_per_qu[1]):
            print(f"⚠️ per_qu真实值超出log1p变换后的合理范围 {log1p_range_per_qu}")


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
    # === 注意：训练和验证都使用log1p变换后的数据，不需要反变换 ===
    # 直接使用log1p变换后的数据进行评估
    y_reg_val_log1p = y_reg_val  # 已经是log1p变换后的
    reg_pred_log1p = reg_pred    # 模型输出也是log1p变换后的
    
    # 添加调试信息
    debug_data_transformation(y_reg_val_log1p, reg_pred_log1p, "模型评估")
    
    # 分类指标
    cls_accuracy = accuracy_score(y_cls_val, cls_pred_labels)
    cls_report = classification_report(y_cls_val, cls_pred_labels)
    
    # 回归指标 - 只关注per_mu（第一个输出），使用log1p变换后的数据
    reg_r2 = [r2_score(y_reg_val_log1p[:, 1], reg_pred_log1p[:, 1])]  # 只计算per_mu的R²（第二列）
    reg_mae = [mean_absolute_error(y_reg_val_log1p[:, i], reg_pred_log1p[:, i]) for i in range(y_reg_val_log1p.shape[1])]
    reg_rmse = [np.sqrt(mean_squared_error(y_reg_val_log1p[:, i], reg_pred_log1p[:, i])) for i in range(y_reg_val_log1p.shape[1])]
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

        # 添加R²值 - 只关注per_mu，使用log1p变换后的数据
        if i == 1:  # 只对per_mu计算R²（第二列）
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
            shap = None
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

            # 只计算per_mu的R²值（第二列，因为标签顺序是['per_qu', 'per_mu']）
            if y_true.ndim > 1 and y_true.shape[1] > 1:
                y_true_per_mu = y_true[:, 1]  # per_mu是第二列
            else:
                y_true_per_mu = y_true.flatten()

            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                y_pred_per_mu = y_pred[:, 1]  # per_mu是第二列
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
        if r2 > self.min_r2 and r2 > self.best_r2:
            self.best_r2 = r2
            if self.save_path:
                # 生成动态文件名，包含R²值
                base_path = os.path.dirname(self.save_path)
                base_name = os.path.basename(self.save_path).replace('_dynamic', '')
                dynamic_filename = base_name.replace('.h5', f'_r2_{r2:.4f}.h5')
                dynamic_save_path = os.path.join(base_path, dynamic_filename)
                
                # 删除之前的模型文件
                if hasattr(self, 'last_saved_path') and self.last_saved_path and os.path.exists(self.last_saved_path):
                    try:
                        os.remove(self.last_saved_path)
                    except:
                        pass
                
                save_model_with_custom_objects(self.model, dynamic_save_path)
                self.last_saved_path = dynamic_save_path
                print(f"✅ Epoch {epoch + 1}: R²={r2:.4f} >= {self.min_r2}, 模型已保存到: {dynamic_filename}")
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
        print(f"\n🎯 跳过Optuna优化，直接使用R²达到0.7时的最佳参数...")
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
            # 先添加R²计算回调，确保val_r2在logs中可用
            OptunaR2EarlyStopping(X_val, y_reg_val, min_r2=0.0, trial=None),
            EarlyStopping(
                monitor='val_loss',
                patience=Config.dl_params['early_stop_patience'],
                restore_best_weights=True,
                verbose=1
            ),
        # 使用自定义回调函数来动态保存模型，避免ModelCheckpoint的格式化问题
        DynamicFinalModelCheckpoint(Config.final_model_dir),
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
                    factor=0.7,  # 温和的学习率衰减
                    patience=3,  # 增加耐心，避免过早衰减
                    min_lr=1e-6,  # 提高最小学习率
                    verbose=1,
                    mode='min'
                ),
                # 先执行R²计算，确保logs中有val_r2值
                OptunaR2EarlyStopping(
                    X_val, y_reg_val, min_r2=0.7, trial=trial
                ),
                # 然后执行模型保存，此时logs中已有val_r2值
                DynamicModelCheckpoint(trial.number, Config.model_dir),
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: monitor.update_metrics(epoch, logs)
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
                # 检查是否是因为R²达到0.7而停止的
                if hasattr(trial, 'user_attrs') and trial.user_attrs.get('r2_achieved', 0) >= 0.7:
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

    # 添加R²达到0.7时的早期停止机制
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

                # 检查trial是否因为R²达到0.7而停止
                if hasattr(trial, 'user_attrs') and trial.user_attrs.get('r2_achieved', 0) >= 0.7:
                    r2_achieved = True
                    best_r2 = trial.user_attrs.get('r2_achieved', 0)
                    print(f"🎯 试验 {trial.number}: R²达到{best_r2:.4f}，停止调参训练！")
                    break

            except optuna.TrialPruned:
                # 检查是否是因为R²达到0.7而剪枝
                if hasattr(trial, 'user_attrs') and trial.user_attrs.get('r2_achieved', 0) >= 0.7:
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

    # 检查是否有试验达到R²=0.7
    for trial in study.trials:
        if hasattr(trial, 'user_attrs') and trial.user_attrs.get('r2_achieved', 0) >= 0.7:
            print(f"🎯 发现R²达到0.7的试验: 试验{trial.number}, R²={trial.user_attrs.get('r2_achieved', 0):.4f}")
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
    # 使用全局最佳参数（如果R²达到0.7）或Optuna最佳参数
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
        # 使用自定义回调函数来动态保存模型，避免ModelCheckpoint的格式化问题
        DynamicFinalModelCheckpoint(Config.final_model_dir),
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
        
        # ============= 交叉验证中的数据量控制 =============
        if Config.enable_data_sampling:
            print(f"🔧 交叉验证第{fold}折: 应用数据量控制...")
            
            # 控制训练集大小
            if Config.max_train_samples is not None and len(X_train_fold) > Config.max_train_samples:
                print(f"  训练集采样: {len(X_train_fold):,} -> {Config.max_train_samples:,}")
                
                if Config.sampling_strategy == 'random':
                    # 随机采样
                    sample_idx = np.random.choice(len(X_train_fold), Config.max_train_samples, replace=False)
                elif Config.sampling_strategy == 'stratified':
                    # 分层采样
                    from sklearn.model_selection import train_test_split
                    _, sample_idx = train_test_split(
                        range(len(X_train_fold)),
                        train_size=Config.max_train_samples,
                        random_state=Config.random_seed,
                        stratify=y_cls_train_fold
                    )
                else:
                    # 取前N个样本
                    sample_idx = range(Config.max_train_samples)
                
                X_train_fold = X_train_fold.iloc[sample_idx].reset_index(drop=True)
                y_cls_train_fold = y_cls_train_fold[sample_idx]
                y_reg_train_fold = y_reg_train_fold.iloc[sample_idx].reset_index(drop=True)
                print(f"  ✅ 训练集采样完成: {X_train_fold.shape}")
            
            # 控制验证集大小
            if Config.max_val_samples is not None and len(X_val_fold) > Config.max_val_samples:
                print(f"  验证集采样: {len(X_val_fold):,} -> {Config.max_val_samples:,}")
                
                if Config.sampling_strategy == 'random':
                    # 随机采样
                    sample_idx = np.random.choice(len(X_val_fold), Config.max_val_samples, replace=False)
                elif Config.sampling_strategy == 'stratified':
                    # 分层采样
                    from sklearn.model_selection import train_test_split
                    _, sample_idx = train_test_split(
                        range(len(X_val_fold)),
                        train_size=Config.max_val_samples,
                        random_state=Config.random_seed,
                        stratify=y_cls_val_fold
                    )
                else:
                    # 取前N个样本
                    sample_idx = range(Config.max_val_samples)
                
                X_val_fold = X_val_fold.iloc[sample_idx].reset_index(drop=True)
                y_cls_val_fold = y_cls_val_fold[sample_idx]
                y_reg_val_fold = y_reg_val_fold.iloc[sample_idx].reset_index(drop=True)
                print(f"  ✅ 验证集采样完成: {X_val_fold.shape}")
            
            print(f"  🎯 第{fold}折最终数据量: 训练集={X_train_fold.shape[0]:,}, 验证集={X_val_fold.shape[0]:,}")
        fold_models = []
        for model_idx in range(n_models):
            print(f"\n训练模型 {model_idx + 1}/{n_models}...")
            # XGBoost特征提取（使用全部数据）
            X_train_combined, X_val_combined, xgb_model, y_cls_train_sample, y_cls_val_sample, y_reg_train_sample, y_reg_val_sample = extract_xgboost_features(
                X_train_fold, X_val_fold, y_cls_train_fold, y_cls_val_fold, y_reg_train_fold, y_reg_val_fold
            )
            # 训练深度学习模型（只用采样数据）
            # 如果R²已达到0.7，跳过Optuna优化，直接使用最佳参数
            if r2_achieved and global_best_params is not None:
                print(f"🎯 交叉验证第{fold}折: 使用R²达到0.7时的最佳参数，跳过调参")
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
        # 只计算per_mu的R²值，使用log1p变换后的数据
        # 确保y_reg_val_sample是numpy数组
        if hasattr(y_reg_val_sample, 'values'):
            y_reg_val_sample = y_reg_val_sample.values
        reg_r2 = r2_score(y_reg_val_sample[:, 1], ensemble_reg[:, 1])  # 只使用per_mu（第二列），log1p变换后的数据
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
        X_train, X_val, y_reg_train, y_reg_val, data_val = load_data()
        
        # 创建虚拟的分类标签（因为当前模型只做回归）
        y_cls_train = np.zeros(len(X_train))  # 虚拟分类标签
        y_cls_val = np.zeros(len(X_val))      # 虚拟分类标签
        
        # 创建虚拟的验证集ID和数据
        val_ids = np.arange(len(X_val))
        data_val = pd.DataFrame({
            'per_qu': y_reg_val['per_qu'], 
            'per_mu': y_reg_val['per_mu'],
            'suit': np.ones(len(y_reg_val))  # 添加suit列，设置为1
        })
        
        check_data(X_train, y_reg_train, y_cls_train, "train")

        # 4.1 验证数据质量
        print("\n4.1 验证数据质量...")
        # 注意：这里验证集还没有进行log1p变换，所以先不验证标签范围
        validate_training_data(X_train, y_reg_train, y_cls_train, X_val, y_reg_val, y_cls_val)

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

        # 从data_val中提取对应的标签（已经是log1p变换后的）
        y_cls_val = data_val['suit'].iloc[:len(X_val)]
        y_reg_val = data_val[['per_qu', 'per_mu']].iloc[:len(X_val)]

        # === 验证集目标变量已经是log1p变换后的，无需再次变换 ===
        print("验证集目标变量已经是log1p变换后的，无需再次变换...")

        # 保存验证集标签，供后续采样使用
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

        print(f"验证集统计信息（已经是log1p变换后的）:")
        print(
            f"per_mu: min={y_reg_val['per_mu'].min():.4f}, max={y_reg_val['per_mu'].max():.4f}, mean={y_reg_val['per_mu'].mean():.4f}")
        print(
            f"per_qu: min={y_reg_val['per_qu'].min():.4f}, max={y_reg_val['per_qu'].max():.4f}, mean={y_reg_val['per_qu'].mean():.4f}")

        print(f"训练集标签形状: y_cls={y_cls_train_final.shape}, y_reg={y_reg_train_final.shape}")
        print(f"验证集标签形状: y_cls={y_cls_val.shape}, y_reg={y_reg_val.shape}")

        # 8. 交叉验证和模型集成
        print("\n8. 执行交叉验证和模型集成...")

        # 使用全局变量检查R²是否已达到0.7
        global GLOBAL_R2_ACHIEVED, GLOBAL_BEST_PARAMS, GLOBAL_BEST_R2

        print(f"🔍 全局R²状态检查: GLOBAL_R2_ACHIEVED={GLOBAL_R2_ACHIEVED}, GLOBAL_BEST_R2={GLOBAL_BEST_R2}")
        print(f"🔍 全局最佳参数: {GLOBAL_BEST_PARAMS}")

        if GLOBAL_R2_ACHIEVED:
            print(f"🎯 检测到R²已达到{GLOBAL_BEST_R2:.4f}，交叉验证将使用最佳参数，跳过调参")
            print(f"使用全局最佳参数: {GLOBAL_BEST_PARAMS}")
        else:
            print("ℹ️ 未检测到R²达到0.7，交叉验证将进行正常调参")

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

        # 检查是否已有训练好的模型（优先使用所有模型中的最佳模型）
        final_model_dir = Config.final_model_dir
        
        # 调试信息：显示目录中的所有文件
        all_files = os.listdir(final_model_dir)
        print(f"🔍 目录 {final_model_dir} 中的所有文件:")
        for file in all_files:
            print(f"  - {file}")
        
        # 支持所有包含 r2_数字 格式的文件名（如 model_trial_1_epoch_15_r2_0.4785.h5）
        model_files = [f for f in all_files if 'r2_' in f and f.endswith('.h5')]
        print(f"🔍 找到 {len(model_files)} 个包含 r2_ 格式的模型文件")
        
        if model_files:
            # 第一优先级：使用所有模型中的最佳模型
            best_model_file = None
            best_r2 = 0.0
            
            print(f"🔍 扫描 {len(model_files)} 个模型文件，寻找最佳模型...")
            for model_file in model_files:
                try:
                    # 使用正则表达式提取 r2_ 后面的数字
                    import re
                    r2_match = re.search(r'r2_([+-]?\d+\.?\d*)', model_file)
                    if r2_match:
                        r2_str = r2_match.group(1)
                        r2_value = float(r2_str)
                        if r2_value > best_r2:
                            best_r2 = r2_value
                            best_model_file = model_file
                            print(f"  📊 找到更好的模型: {model_file} (R²={r2_value:.4f})")
                except (ValueError, AttributeError):
                    print(f"  ⚠️ 无法解析文件名: {model_file}")
                    continue
            
            if best_model_file:
                final_model_path = os.path.join(final_model_dir, best_model_file)
                
                # 根据R²值显示不同的提示信息
                if best_r2 >= 0.7:
                    print(f"🎯 使用最佳模型: {best_model_file} (R²={best_r2:.4f}) - 性能优秀!")
                elif best_r2 >= 0.3:
                    print(f"✅ 使用最佳模型: {best_model_file} (R²={best_r2:.4f}) - 性能良好")
                else:
                    print(f"⚠️ 使用最佳模型: {best_model_file} (R²={best_r2:.4f}) - 性能一般")
                
                try:
                    best_model = load_model_with_custom_objects(final_model_path)
                    print("✅ 成功加载最佳模型")
                    # 尝试从final_logsAA读取训练历史
                    final_logs_dir = os.path.join(Config.output_dir, "final_logsAA")
                    final_history_file = os.path.join(final_logs_dir, "training_history.json")
                    
                    if os.path.exists(final_history_file):
                        try:
                            with open(final_history_file, 'r', encoding='utf-8') as f:
                                history_data = json.load(f)
                            # 创建MockHistory对象
                            class MockHistory:
                                def __init__(self, history_data):
                                    self.history = history_data
                            history = MockHistory(history_data)
                            print("✅ 成功从final_logsAA读取训练历史")
                        except Exception as e:
                            print(f"⚠️ 读取训练历史失败: {e}")
                            history = None
                    else:
                        history = None  # 没有历史记录
                except Exception as e:
                    print(f"❌ 加载模型失败: {e}")
                    print("将重新训练模型...")
                    best_model, history = None, None
            else:
                print("❌ 无法解析任何模型文件，将重新训练...")
                best_model, history = None, None
        else:
            print("📝 未发现已存在的模型，将重新训练...")
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

        # 14. 保存训练历史
        print("\n14. 保存训练历史...")
        history_file = os.path.join(Config.logs_dir, "training_history.json")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        # 尝试从final_logsAA目录读取训练历史
        final_logs_dir = os.path.join(Config.output_dir, "final_logsAA")
        final_history_file = os.path.join(final_logs_dir, "training_history.json")
        
        if os.path.exists(final_history_file):
            print(f"📁 从final_logsAA目录读取训练历史: {final_history_file}")
            try:
                with open(final_history_file, 'r', encoding='utf-8') as f:
                    final_history_data = json.load(f)
                
                # 保存到当前logs目录
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(final_history_data, f, ensure_ascii=False, indent=2)
                print(f"✅ 训练历史已从final_logsAA复制到: {history_file}")
                
                # 创建history对象用于后续图表生成
                class MockHistory:
                    def __init__(self, history_data):
                        self.history = history_data
                
                history = MockHistory(final_history_data)
                
            except Exception as e:
                print(f"⚠️ 读取final_logsAA训练历史失败: {e}")
                if history is not None:
                    with open(history_file, 'w', encoding='utf-8') as f:
                        json.dump(history.history, f, default=json_fallback, ensure_ascii=False, indent=2)
                    print(f"训练历史已保存至: {history_file}")
                else:
                    print("⚠️ 没有训练历史记录（使用了已存在的模型）")
                    with open(history_file, 'w', encoding='utf-8') as f:
                        json.dump({"message": "使用已存在的模型，无训练历史记录"}, f, ensure_ascii=False, indent=2)
        else:
            print(f"⚠️ final_logsAA目录中未找到训练历史文件: {final_history_file}")
            if history is not None:
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(history.history, f, default=json_fallback, ensure_ascii=False, indent=2)
                print(f"训练历史已保存至: {history_file}")
            else:
                print("⚠️ 没有训练历史记录（使用了已存在的模型）")
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

        # 15. 生成预测地图
        print("\n15. 生成预测地图...")
        try:
            # 修复：使用最后一次batch的预测结果，或者重新预测一小部分数据用于可视化
            sample_size = min(1000, X_val_combined.shape[0])  # 取1000个样本用于可视化
            
            # 检查data_val是否有坐标信息
            coord_cols = ['x', 'y', 'x_product', 'y_product', 'right_x', 'right_y']
            available_coords = [col for col in coord_cols if col in data_val.columns]
            
            if not available_coords or len(available_coords) < 2:
                print("⚠️ 无法找到有效的坐标列，跳过预测地图生成")
                print(f"可用的列: {data_val.columns.tolist()}")
                print("需要的坐标列应该包含: ['x', 'y'] 或 ['x_product', 'y_product'] 或 ['right_x', 'right_y']")
            else:
                # 创建包含坐标信息的X_val用于地图绘制
                sample_preds = best_model.predict(X_val_combined[:sample_size])
                
                # 确保预测结果与训练时的log1p变换保持一致
                sample_cls_pred, sample_reg_pred = ensure_log1p_consistency(sample_preds, "地图生成")
                
                # 使用data_val中对应的样本数据（包含坐标信息）
                # 创建val_ids作为索引
                val_ids_sample = np.arange(sample_size)
                plot_prediction_maps(data_val.iloc[:sample_size], sample_cls_pred, sample_reg_pred, val_ids_sample,
                                     data_val.iloc[:sample_size])
                print("✅ 预测地图生成完成")
        except Exception as e:
            print(f"⚠️ 生成预测地图时出错: {str(e)}")
            print("地图生成失败，但不会影响训练和模型的正常使用")
        # 自动保存交互特征名和最终特征顺序
        with open(os.path.join(Config.output_dir, "selected_inter_feats.json"), "w", encoding="utf-8") as f:
            json.dump(selected_inter_feats, f, ensure_ascii=False)
        with open(os.path.join(Config.output_dir, "feature_order.json"), "w", encoding="utf-8") as f:
            # 安全获取特征列名
            if hasattr(X_train_final, 'columns'):
                feature_columns = list(X_train_final.columns)
            else:
                # 如果X_train_final是numpy数组，创建默认特征名
                feature_columns = [f"feature_{i}" for i in range(X_train_final.shape[1])]
            json.dump(feature_columns, f, ensure_ascii=False)
        # 16. 生成分析图表
        print("\n16. 生成分析图表...")
        
        # 尝试从final_logsAA目录读取训练历史用于图表生成
        final_logs_dir = os.path.join(Config.output_dir, "final_logsAA")
        final_history_file = os.path.join(final_logs_dir, "training_history.json")
        
        # 如果之前没有从final_logsAA读取到history，现在尝试读取
        if history is None and os.path.exists(final_history_file):
            print(f"📁 从final_logsAA目录读取训练历史用于图表生成: {final_history_file}")
            try:
                with open(final_history_file, 'r', encoding='utf-8') as f:
                    final_history_data = json.load(f)
                
                # 创建history对象用于图表生成
                class MockHistory:
                    def __init__(self, history_data):
                        self.history = history_data
                
                history = MockHistory(final_history_data)
                print("✅ 成功从final_logsAA读取训练历史用于图表生成")
                
            except Exception as e:
                print(f"⚠️ 读取final_logsAA训练历史失败: {e}")
        
        try:
            # 16.1 训练历史图表
            if history is not None:
                print("📊 使用final_logsAA中的训练历史生成图表...")
                plot_training_history(history)
            else:
                print("⚠️ 没有训练历史记录，跳过训练历史图表生成")

            # 16.2 混淆矩阵
            # 修复：多输出模型predict返回(cls_pred, reg_pred)，不能直接argmax
            print("🔍 生成混淆矩阵...")
            preds = best_model.predict(X_val_combined)
            
            # 确保预测结果与训练时的log1p变换保持一致
            cls_pred, reg_pred = ensure_log1p_consistency(preds, "混淆矩阵生成")
            
            y_cls_pred = np.argmax(cls_pred, axis=1)
            # 使用完整的验证集标签，而不是未定义的sample版本
            plot_confusion_matrix(y_cls_val, y_cls_pred, classes=['不适合', '适合'])

            # 16.3 回归散点图
            print("🔍 生成回归散点图...")
            if reg_pred is not None:
                y_reg_pred = reg_pred
            else:
                # 兼容性兜底 - 避免重复predict调用
                y_reg_pred = preds[1] if isinstance(preds, (list, tuple)) and len(preds) > 1 else preds
            # 使用完整的验证集标签，而不是未定义的sample版本
            plot_regression_scatter(y_reg_val, y_reg_pred, ['per_qu', 'per_mu'])

            # 16.4 特征重要性图
            print("🔍 生成特征重要性图...")
            # 尝试从XGBoost特征提取中获取模型
            try:
                # 重新加载XGBoost模型
                xgb_model_path = os.path.join(Config.xgboost_dir, "xgb_model_AA.json")
                if os.path.exists(xgb_model_path):
                    # 使用全局导入的xgb，而不是局部导入
                    xgb_model = xgb.Booster()
                    xgb_model.load_model(xgb_model_path)
                    plot_feature_importance(xgb_model, feature_names)
                else:
                    print("⚠️ XGBoost模型文件不存在，跳过特征重要性图")
            except Exception as e:
                print(f"⚠️ 加载XGBoost模型失败: {str(e)}")
                print("跳过特征重要性图")

            # 16.5 生成evaluation_report.json
            print("🔍 生成模型评估报告...")
            # 使用完整的验证集标签，而不是未定义的sample版本
            evaluate_model(best_model, X_val_combined, y_cls_val, y_reg_val)
            
            print("✅ 所有分析图表生成完成")
            
        except Exception as e:
            print(f"⚠️ 生成分析图表时出错: {str(e)}")
            print("图表生成失败，但不会影响训练和模型的正常使用")
            import traceback
            traceback.print_exc()

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
            os.path.join(Config.final_model_dir, 'final_model_r2_{val_r2:.4f}.h5'),
            monitor='val_r2',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    # 训练模型
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
        epochs=1,  # 进一步减少epochs，加快训练
        batch_size=conservative_params['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    return model, history


print('=== 脚本已启动 ===')
# ... existing code ...
if __name__ == "__main__":
    print('=== 进入主流程 ===')
    main()


