"""
Copyright (c) 2025 张德海
MIT Licensed - 详见项目根目录 LICENSE 文件

项目: Meteorology-Assisted-Crop-Yield-Prediction
仓库: https://github.com/ZDH4869/Meteorology-Assisted-Crop-Yield-Prediction.git
联系: zhangdehai1412@163.com
"""
# -*- coding: utf-8 -*-
"""
气象数据空间插值系统 
"""
import gc
import os
import threading
import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pykrige.uk import UniversalKriging
from dask.distributed import Client, progress
import matplotlib.pyplot as plt
import optuna
from joblib import dump, load
import warnings
from rasterio.transform import Affine
from torch.cuda.amp import autocast, GradScaler
from sklearn.neighbors import NearestNeighbors
import psutil
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
from scipy.interpolate import griddata

warnings.filterwarnings('ignore')
from tqdm.auto import tqdm

# 可选：用于获取 GPU 详细信息
try:
    import GPUtil
except ImportError:
    GPUtil = None

# 配置全局进度条样式
tqdm._instances.clear()  # 防止多个实例冲突
tqdm.pandas(
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

# ================= GPU 检测与监控 =================
# 全局状态
gpu_monitor_active = False


def detect_device():
    """自动检测 GPU 可用性"""
    if not Config.force_cpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def gpu_monitor(interval=5):
    """后台线程：定时打印 GPU 使用情况"""
    global gpu_monitor_active
    while gpu_monitor_active:
        if GPUtil:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                print(f"[GPU Monitor] GPU Usage: {gpu.load * 100:.1f}% | Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
        else:
            # 备用：使用 torch API
            print(f"[GPU Monitor] Allocated: {torch.cuda.memory_allocated() // 1024 ** 2} MiB | "
                  f"Cached: {torch.cuda.memory_reserved() // 1024 ** 2} MiB")
        time.sleep(interval)


# 打印 GPU 信息
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"检测到的GPU数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")


def memory_safe(func):
    """内存安全装饰器"""

    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem = process.memory_info().rss / 1024 ** 3  # GB
        if mem > 10:  # 当内存超过10GB时报警
            print(f"Memory warning: {mem:.2f}GB used")
        return func(*args, **kwargs)

    return wrapper


def meters_to_degrees(meters):
    """将米转换为度（近似值，在赤道处1度约等于111.32公里）"""
    return meters / 111320.0  # 1度约等于111.32公里


# ================= 参数配置 =================
class Config:
    # 输入输出路径 气象站点中的lon,lat对应的是x,y坐标
    input_csv_dir = r"1-气象插值代码/测试数据/气象数据"
    input_terrain_tif = r"1-气象插值代码/测试数据/test_tif.tif"
    output_raster_dir = r"1-气象插值代码/输出展示raster"
    output_csv_path = r"1-气象插值代码/输出展示\csv/final.csv"
    model_save_dir = r"1-气象插值代码/输出展示\model"
    plot_dir = r"1-气象插值代码/输出展示\plot"
    final_output_csv = r"1-气象插值代码/输出展示\csv/final_interpolated_data.csv"  # 新增：最终插值结果CSV路径
    confidence_uncertainty_csv = r"1-气象插值代码/输出展示\confidence_uncertainty_analysis.csv"  # 置信分析和不确定性分析结果CSV路径

    # 日期范围设置
    start_date = "2005-01"  # 格式：YYYY-MM
    end_date = "2015-12"  # 格式：YYYY-MM

    # 像元大小设置（单位：米）
    pixel_size = 90  # 设置像元大小为10000米
    # 分辨率与像元大小的换算方式：分辨率 = 像元大小 / 111320.0（1度约等于111.32公里）
    
    # CSV输出坐标间隔设置（单位：米）
    csv_pixel_size = 500  # None时自动采用地形栅格分辨率，否则使用指定值

    # 数据处理参数 Rain_min 没有训练价值，恒定为0

    target_vars = ['Tsun_sum', 'Tsun_max', 'Tsun_min', 'Tsun_mean', 'TAVE_max', 'TAVE_min', 'TAVE_mean',
        'Tmax_max', 'Tmax_min', 'Tmax_mean', 'Tmin_max', 'Tmin_min', 'Tmin_mean',
        'Rain_sum', 'Rain_max',
        'Rain_mean', 'GTAVE_max', 'GTAVE_min', 'GTAVE_mean',
        'GTmax_max', 'GTmax_min', 'GTmax_mean', 'GTmin_max', 'GTmin_min', 'GTmin_mean',
        'Sevp_sum', 'Sevp_max', 'Sevp_min', 'Sevp_mean',
             ]  # 示例变量
    validation_ratio = 0.2
    random_seed = 42
    pos_enc_dim = 64  # 位置编码维度

    # 深度学习参数
    dl_params = {
        'hidden_layers':[512, 256, 128],  # 增加模型复杂度
        'activation': 'ReLU',
        'dropout': 0.2,
        'learning_rate': 1e-3,
        'batch_size': 256,
        'epochs':50,
        'early_stop_patience': 10,
        'use_layer_norm': True,
        'weight_decay': 1e-3,
        'grad_clip': 10,
        'use_swa': False
    }

    # 协同克里金参数（实际为：指数衰减权重空间插值）
    kriging_params = {
        'variogram_model': 'spherical',
        'nlags': 6,
        'anisotropy_angle': 30.0,
        'anisotropy_scaling': 2.0
    }

    # 系统参数
    use_gpu = True  # 启用 GPU 支持
    chunk_size = 50000
    memory_limit = '16GB'  # 限制Dask工作进程内存
    parallel_workers = 2  # 减少并行数
    force_cpu = False  # 是否强制使用 CPU
    use_amp = True  # 启用自动混合精度训练
    gpu_batch_size = 1024  # GPU批处理大小
    gpu_memory_fraction = 0.8  # GPU内存使用比例

    # 确保输出目录存在
    @staticmethod
    def create_output_dirs():
        """确保所有输出目录存在"""
        dirs_to_create = [
            Config.output_raster_dir,
            Config.model_save_dir,
            Config.plot_dir,
            os.path.dirname(Config.final_output_csv),
            os.path.dirname(Config.output_csv_path),
            os.path.dirname(Config.confidence_uncertainty_csv)
        ]

        for dir_path in dirs_to_create:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"确保目录存在: {dir_path}")
            except Exception as e:
                print(f"创建目录失败 {dir_path}: {str(e)}")

    @staticmethod
    def parse_date(date_str):
        """解析日期字符串为年月"""
        year, month = map(int, date_str.split('-'))
        return year, month

    @staticmethod
    def get_date_range():
        """获取日期范围"""
        start_year, start_month = Config.parse_date(Config.start_date)
        end_year, end_month = Config.parse_date(Config.end_date)
        return start_year, start_month, end_year, end_month

    @staticmethod
    def setup_gpu():
        """配置GPU设置"""
        if torch.cuda.is_available() and Config.use_gpu and not Config.force_cpu:
            # 设置GPU内存使用比例
            torch.cuda.set_per_process_memory_fraction(Config.gpu_memory_fraction)
            # 设置GPU设备
            torch.cuda.set_device(0)
            # 启用cuDNN自动调优
            torch.backends.cudnn.benchmark = True
            # 启用自动混合精度
            torch.backends.cudnn.enabled = True
            return True
        return False


# ================= 硬件配置 =================
device = torch.device('cuda' if torch.cuda.is_available() and Config.use_gpu and not Config.force_cpu else 'cpu')
print(f"Using device: {device}")


# ================= 数据加载模块 =================
class GeoDataLoader:
    @staticmethod
    def load_meteo_data():
        """加载并合并气象站数据"""
        try:
            dfs = []
            for f in tqdm(os.listdir(Config.input_csv_dir), desc="Loading CSVs"):
                if f.endswith('.csv'):
                    try:
                        df = dd.read_csv(os.path.join(Config.input_csv_dir, f),
                                         blocksize=Config.chunk_size)
                        dfs.append(df)
                    except Exception as e:
                        print(f"Error loading {f}: {str(e)}")
                        continue

            if not dfs:
                raise ValueError("No valid CSV files found in input directory")

            return dd.concat(dfs).compute()
        except Exception as e:
            print(f"Error in load_meteo_data: {str(e)}")
            raise

    @staticmethod
    def load_terrain():
        """加载地形数据"""
        try:
            with rasterio.open(Config.input_terrain_tif) as src:
                terrain = src.read()
                height, width = src.shape
                x, y = np.meshgrid(np.arange(width), np.arange(height))
                lons, lats = rasterio.transform.xy(src.transform, y, x)
                # rasterio.transform.xy返回的是两个1D数组，需要转换为2D数组
                lons = np.array(lons).reshape(height, width)
                lats = np.array(lats).reshape(height, width)

                # 获取nodata值并创建有效数据掩码
                nodata_value = src.nodata
                if nodata_value is None:
                    nodata_value = -9999
                
                elevation_data = terrain[0].astype(np.float32)
                # 创建有效数据掩码，排除nodata值、异常值和高程为0的区域
                valid_mask = (elevation_data != nodata_value) & (~np.isnan(elevation_data)) & (elevation_data > 0) & (elevation_data < 10000)
                
                print(f"训练数据 - 总像素数: {elevation_data.size}")
                print(f"训练数据 - 有效像素数: {np.sum(valid_mask)}")
                print(f"训练数据 - 无效像素数: {np.sum(~valid_mask)}")
                print(f"训练数据 - 有效数据比例: {np.sum(valid_mask) / elevation_data.size * 100:.2f}%")

                # 使用float32减少内存使用
                return {
                    'data': terrain.astype(np.float32),
                    'profile': src.profile,
                    'transform': src.transform,
                    'lons': lons.astype(np.float32),
                    'lats': lats.astype(np.float32),
                    'elevation': elevation_data,
                    'valid_mask': valid_mask,
                    'nodata_value': nodata_value
                }
        except Exception as e:
            print(f"Error in load_terrain: {str(e)}")
            raise

# ================= 位置编码模块 =================
class PositionalEncoder:
    @staticmethod
    def geo_encoding(lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        """内存优化的位置编码"""
        d = Config.pos_enc_dim
        enc = torch.empty((len(lon), d),
                          dtype=torch.float32,
                          device=lon.device)

        # 分频段逐步计算
        for i in range(d // 4):
            freq = 2 ** i
            enc[:, 4 * i] = torch.sin(lon * freq)
            enc[:, 4 * i + 1] = torch.cos(lon * freq)
            enc[:, 4 * i + 2] = torch.sin(lat * freq)
            enc[:, 4 * i + 3] = torch.cos(lat * freq)
            # 及时释放中间变量
            del freq
            if i % 4 == 0:
                torch.cuda.empty_cache() if lon.device.type == 'cuda' else None

        return enc


# ================= 深度学习模型 =================
class GeoNetWithPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe_dim = Config.pos_enc_dim

        # 位置编码分支
        self.pe_net = nn.Sequential(
            nn.Linear(self.pe_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        # 主特征分支（输入维度应为：128[位置编码输出] + 3[其他特征] = 131）
        self.main_net = nn.Sequential(
            nn.Linear(128 + 3, 256),  # 修正输入维度
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(Config.dl_params['dropout']),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        assert x.shape[1] == 5, f"输入维度应为5，实际为{x.shape[1]}"
        # 分离坐标特征（输入应为原始5维特征）
        lon_lat = x[:, :2]  # 经度/纬度
        other_features = x[:, 2:]  # elevation/YYYY/MM (3维)

        # 生成位置编码
        pe = PositionalEncoder.geo_encoding(lon_lat[:, 0], lon_lat[:, 1])
        pe = self.pe_net(pe)  # 输出128维

        # 合并特征
        combined = torch.cat([pe, other_features], dim=1)  # 128+3=131维
        return self.main_net(combined)


# ================= 训练模块 =================
class GeoTrainer:
    def __init__(self, model):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and Config.use_gpu and not Config.force_cpu else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(),
                                     lr=Config.dl_params['learning_rate'],
                                     weight_decay=Config.dl_params['weight_decay'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
        self.scaler = GradScaler(enabled=Config.use_amp and self.device.type == 'cuda')
        self.criterion = nn.HuberLoss()
        self.best_model_state = None
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': [],
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': []
        }

    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return r2, mae, rmse

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        try:
            for X, y in tqdm(train_loader, desc="Training", leave=False):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                with autocast(enabled=Config.use_amp and self.device.type == 'cuda'):
                    outputs = self.model(X).squeeze()
                    loss = self.criterion(outputs, y)

                if self.device.type == 'cuda':
                    self.scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), Config.dl_params['grad_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), Config.dl_params['grad_clip'])
                    self.optimizer.step()

                total_loss += loss.item()
                all_preds.extend(outputs.detach().cpu().numpy())
                all_targets.extend(y.cpu().numpy())

                # 清理不需要的张量
                del X, y, outputs, loss
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # 计算训练指标
            r2, mae, rmse = self.calculate_metrics(all_targets, all_preds)
            self.history['train_r2'].append(r2)
            self.history['train_mae'].append(mae)
            self.history['train_rmse'].append(rmse)
            avg_loss = total_loss / len(train_loader)
            self.history['train_loss'].append(avg_loss)
            return avg_loss
        except Exception as e:
            print(f"Error in train_epoch: {str(e)}")
            raise

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        try:
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    with autocast(enabled=Config.use_amp and self.device.type == 'cuda'):
                        outputs = self.model(X).squeeze()
                        loss = self.criterion(outputs, y)
                    total_loss += loss.item()
                    all_preds.extend(outputs.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())

                    # 清理不需要的张量
                    del X, y, outputs, loss
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

            # 计算验证指标
            r2, mae, rmse = self.calculate_metrics(all_targets, all_preds)
            self.history['val_r2'].append(r2)
            self.history['val_mae'].append(mae)
            self.history['val_rmse'].append(rmse)
            avg_loss = total_loss / len(val_loader)
            self.history['val_loss'].append(avg_loss)
            return avg_loss
        except Exception as e:
            print(f"Error in validate: {str(e)}")
            raise

    def plot_training_history(self, var_name):
        """绘制训练历史"""
        plt.figure(figsize=(15, 10))

        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History for {var_name}', fontsize=16)

        # 损失曲线
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # R2曲线
        ax2.plot(self.history['train_r2'], label='Train R²')
        ax2.plot(self.history['val_r2'], label='Validation R²')
        ax2.set_title('R² Score Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R² Score')
        ax2.legend()
        ax2.grid(True)

        # MAE曲线
        ax3.plot(self.history['train_mae'], label='Train MAE')
        ax3.plot(self.history['val_mae'], label='Validation MAE')
        ax3.set_title('MAE Curves')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MAE')
        ax3.legend()
        ax3.grid(True)

        # RMSE曲线
        ax4.plot(self.history['train_rmse'], label='Train RMSE')
        ax4.plot(self.history['val_rmse'], label='Validation RMSE')
        ax4.set_title('RMSE Curves')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('RMSE')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(f"{Config.plot_dir}/{var_name}_training_history.png")
        plt.close()

    def print_final_metrics(self, var_name):
        """打印最终指标"""
        print(f"\n=== Final Metrics for {var_name} ===")
        print(f"Training Loss: {self.history['train_loss'][-1]:.4f}")
        print(f"Validation Loss: {self.history['val_loss'][-1]:.4f}")
        print(f"Training R²: {self.history['train_r2'][-1]:.4f}")
        print(f"Validation R²: {self.history['val_r2'][-1]:.4f}")
        print(f"Training MAE: {self.history['train_mae'][-1]:.4f}")
        print(f"Validation MAE: {self.history['val_mae'][-1]:.4f}")
        print(f"Training RMSE: {self.history['train_rmse'][-1]:.4f}")
        print(f"Validation RMSE: {self.history['val_rmse'][-1]:.4f}")
        print(f"Total Epochs: {len(self.history['train_loss'])}")

    def save_best_model(self, var_name):
        """保存最佳模型状态"""
        if self.best_model_state is not None:
            try:
                save_path = f"{Config.model_save_dir}/{var_name}_best.pth"
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict() if self.device.type == 'cuda' else None,
                    'device': self.device.type
                }, save_path)
            except Exception as e:
                print(f"Error saving model for {var_name}: {str(e)}")
                raise
    
    def analyze_confidence_uncertainty(self, X_val, y_val, var_name, scaler):
        """进行置信分析和不确定性分析"""
        print(f"\n开始{var_name}的置信分析和不确定性分析...")
        
        # 创建分析器
        analyzer = ConfidenceUncertaintyAnalyzer(self.model, scaler, self.device)
        
        # 获取坐标信息（用于空间分析）
        coords = X_val[:, :2]  # 经度和纬度
        
        # 进行Bootstrap分析
        bootstrap_results = analyzer.comprehensive_analysis(
            X_val, coords=coords, method='bootstrap', n_samples=50
        )
        
        # 进行Monte Carlo Dropout分析
        mc_results = analyzer.comprehensive_analysis(
            X_val, coords=coords, method='monte_carlo', n_samples=50
        )
        
        # 计算验证集上的预测性能
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val).to(self.device)
            val_predictions = self.model(X_tensor).cpu().numpy().squeeze()
        
        # 计算验证指标
        val_r2, val_mae, val_rmse = self.calculate_metrics(y_val, val_predictions)
        
        # 汇总分析结果
        analysis_results = {
            'variable': var_name,
            'training_metrics': {
                'training_loss': self.history['train_loss'][-1] if self.history['train_loss'] else np.nan,
                'training_r2': self.history['train_r2'][-1] if self.history['train_r2'] else np.nan,
                'training_mae': self.history['train_mae'][-1] if self.history['train_mae'] else np.nan,
                'training_rmse': self.history['train_rmse'][-1] if self.history['train_rmse'] else np.nan,
                'total_epochs': len(self.history['train_loss']) if self.history['train_loss'] else 0
            },
            'validation_metrics': {
                'r2': val_r2,
                'mae': val_mae,
                'rmse': val_rmse
            },
            'bootstrap_analysis': {
                'mean_prediction': np.mean(bootstrap_results['prediction_intervals']['mean']),
                'mean_std': np.mean(bootstrap_results['uncertainty_metrics']['std']),
                'mean_cv': np.mean(bootstrap_results['uncertainty_metrics']['coefficient_of_variation']),
                'mean_relative_uncertainty': np.mean(bootstrap_results['uncertainty_metrics']['relative_uncertainty']),
                'mean_prediction_interval_width': np.mean(bootstrap_results['prediction_intervals']['prediction_interval_width'])
            },
            'monte_carlo_analysis': {
                'mean_prediction': np.mean(mc_results['prediction_intervals']['mean']),
                'mean_std': np.mean(mc_results['uncertainty_metrics']['std']),
                'mean_cv': np.mean(mc_results['uncertainty_metrics']['coefficient_of_variation']),
                'mean_relative_uncertainty': np.mean(mc_results['uncertainty_metrics']['relative_uncertainty']),
                'mean_prediction_interval_width': np.mean(mc_results['prediction_intervals']['prediction_interval_width'])
            }
        }
        
        # 分别进行Bootstrap和Monte Carlo的空间分析
        bootstrap_spatial = analyzer.analyze_spatial_uncertainty(coords, bootstrap_results['predictions'])
        mc_spatial = analyzer.analyze_spatial_uncertainty(coords, mc_results['predictions'])
        
        # 综合空间分析结果
        analysis_results['spatial_uncertainty'] = {
            'bootstrap_mean_uncertainty': bootstrap_spatial.get('mean_uncertainty', np.nan),
            'bootstrap_max_uncertainty': bootstrap_spatial.get('max_uncertainty', np.nan),
            'bootstrap_min_uncertainty': bootstrap_spatial.get('min_uncertainty', np.nan),
            'bootstrap_std_uncertainty': bootstrap_spatial.get('std_uncertainty', np.nan),
            'bootstrap_median_uncertainty': bootstrap_spatial.get('median_uncertainty', np.nan),
            'bootstrap_q25_uncertainty': bootstrap_spatial.get('q25_uncertainty', np.nan),
            'bootstrap_q75_uncertainty': bootstrap_spatial.get('q75_uncertainty', np.nan),
            'monte_carlo_mean_uncertainty': mc_spatial.get('mean_uncertainty', np.nan),
            'monte_carlo_max_uncertainty': mc_spatial.get('max_uncertainty', np.nan),
            'monte_carlo_min_uncertainty': mc_spatial.get('min_uncertainty', np.nan),
            'monte_carlo_std_uncertainty': mc_spatial.get('std_uncertainty', np.nan),
            'monte_carlo_median_uncertainty': mc_spatial.get('median_uncertainty', np.nan),
            'monte_carlo_q25_uncertainty': mc_spatial.get('q25_uncertainty', np.nan),
            'monte_carlo_q75_uncertainty': mc_spatial.get('q75_uncertainty', np.nan),
            'combined_mean_uncertainty': np.mean([bootstrap_spatial.get('mean_uncertainty', np.nan), 
                                                mc_spatial.get('mean_uncertainty', np.nan)]),
            'combined_max_uncertainty': np.max([bootstrap_spatial.get('max_uncertainty', np.nan), 
                                               mc_spatial.get('max_uncertainty', np.nan)]),
            'combined_min_uncertainty': np.min([bootstrap_spatial.get('min_uncertainty', np.nan), 
                                               mc_spatial.get('min_uncertainty', np.nan)]),
            'combined_std_uncertainty': np.mean([bootstrap_spatial.get('std_uncertainty', np.nan), 
                                                mc_spatial.get('std_uncertainty', np.nan)]),
            'uncertainty_method_difference': abs(bootstrap_spatial.get('mean_uncertainty', np.nan) - 
                                               mc_spatial.get('mean_uncertainty', np.nan))
        }
        
        print(f"{var_name}置信分析完成:")
        print(f"  验证R²: {val_r2:.4f}")
        print(f"  Bootstrap平均不确定性: {analysis_results['bootstrap_analysis']['mean_std']:.4f}")
        print(f"  Monte Carlo平均不确定性: {analysis_results['monte_carlo_analysis']['mean_std']:.4f}")
        
        # 显示空间不确定性分析结果
        spatial = analysis_results['spatial_uncertainty']
        print(f"  空间不确定性分析:")
        print(f"    Bootstrap空间不确定性: {spatial['bootstrap_mean_uncertainty']:.4f}")
        print(f"    Monte Carlo空间不确定性: {spatial['monte_carlo_mean_uncertainty']:.4f}")
        print(f"    综合空间不确定性: {spatial['combined_mean_uncertainty']:.4f}")
        print(f"    方法差异: {spatial['uncertainty_method_difference']:.4f}")
        
        return analysis_results


# ================= 置信分析和不确定性分析模块 =================
class ConfidenceUncertaintyAnalyzer:
    """置信分析和不确定性分析器"""
    
    def __init__(self, model, scaler, device):
        self.model = model
        self.scaler = scaler
        self.device = device
        
    def bootstrap_prediction(self, X, n_bootstrap=100):
        """使用Bootstrap方法进行不确定性分析"""
        predictions = []
        
        # 创建Bootstrap样本
        n_samples = len(X)
        bootstrap_indices = np.random.choice(n_samples, size=(n_bootstrap, n_samples), replace=True)
        
        for i in range(n_bootstrap):
            # 获取Bootstrap样本
            X_bootstrap = X[bootstrap_indices[i]]
            
            # 预测
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_bootstrap).to(self.device)
                pred = self.model(X_tensor).cpu().numpy().squeeze()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        return predictions
    
    def monte_carlo_dropout(self, X, n_samples=100):
        """使用Monte Carlo Dropout进行不确定性分析"""
        self.model.train()  # 启用dropout
        predictions = []
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            for _ in range(n_samples):
                pred = self.model(X_tensor).cpu().numpy().squeeze()
                predictions.append(pred)
        
        self.model.eval()  # 恢复评估模式
        predictions = np.array(predictions)
        return predictions
    
    def calculate_prediction_intervals(self, predictions, confidence_level=0.95):
        """计算预测区间"""
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        mean_prediction = np.mean(predictions, axis=0)
        
        return {
            'mean': mean_prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level,
            'prediction_interval_width': upper_bound - lower_bound
        }
    
    def calculate_uncertainty_metrics(self, predictions):
        """计算不确定性指标"""
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        var_pred = np.var(predictions, axis=0)
        
        # 变异系数
        cv = np.where(mean_pred != 0, std_pred / np.abs(mean_pred), 0)
        
        # 相对不确定性
        relative_uncertainty = np.where(mean_pred != 0, std_pred / np.abs(mean_pred), 0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'variance': var_pred,
            'coefficient_of_variation': cv,
            'relative_uncertainty': relative_uncertainty
        }
    
    def analyze_spatial_uncertainty(self, coords, predictions):
        """增强的空间不确定性分析"""
        try:
            # 确保坐标和预测结果维度匹配
            if len(coords) != predictions.shape[1]:
                print(f"警告: 坐标数量({len(coords)})与预测结果数量({predictions.shape[1]})不匹配，跳过空间分析")
                uncertainties = np.std(predictions, axis=0)
                return self._basic_uncertainty_stats(uncertainties)
            
            # 计算每个点的不确定性
            uncertainties = np.std(predictions, axis=0)
            
            # 基本统计信息
            basic_stats = self._basic_uncertainty_stats(uncertainties)
            
            # 空间分析
            spatial_stats = self._spatial_pattern_analysis(coords, uncertainties)
            
            # 合并结果
            return {**basic_stats, **spatial_stats}
            
        except Exception as e:
            print(f"空间不确定性分析出错: {str(e)}")
            uncertainties = np.std(predictions, axis=0)
            return self._basic_uncertainty_stats(uncertainties)
    
    def _basic_uncertainty_stats(self, uncertainties):
        """基本不确定性统计"""
        return {
            'mean_uncertainty': np.mean(uncertainties),
            'max_uncertainty': np.max(uncertainties),
            'min_uncertainty': np.min(uncertainties),
            'std_uncertainty': np.std(uncertainties),
            'median_uncertainty': np.median(uncertainties),
            'q25_uncertainty': np.percentile(uncertainties, 25),
            'q75_uncertainty': np.percentile(uncertainties, 75)
        }
    
    def _spatial_pattern_analysis(self, coords, uncertainties):
        """空间模式分析"""
        try:
            # 计算空间范围
            coord_ranges = {
                'lon_range': np.max(coords[:, 0]) - np.min(coords[:, 0]),
                'lat_range': np.max(coords[:, 1]) - np.min(coords[:, 1])
            }
            
            # 计算不确定性在空间上的分布
            uncertainty_spatial_stats = {
                'coord_ranges': coord_ranges,
                'uncertainty_spatial_variance': np.var(uncertainties),
                'uncertainty_spatial_cv': np.std(uncertainties) / np.mean(uncertainties) if np.mean(uncertainties) > 0 else 0,
                'uncertainty_range_ratio': (np.max(uncertainties) - np.min(uncertainties)) / np.mean(uncertainties) if np.mean(uncertainties) > 0 else 0
            }
            
            return uncertainty_spatial_stats
            
        except Exception as e:
            print(f"空间模式分析出错: {str(e)}")
            return {
                'coord_ranges': {}, 
                'uncertainty_spatial_variance': np.nan, 
                'uncertainty_spatial_cv': np.nan,
                'uncertainty_range_ratio': np.nan
            }
    
    def comprehensive_analysis(self, X, coords=None, method='bootstrap', n_samples=100):
        """综合置信分析和不确定性分析"""
        print(f"开始{method}不确定性分析，样本数: {n_samples}")
        
        if method == 'bootstrap':
            predictions = self.bootstrap_prediction(X, n_samples)
        elif method == 'monte_carlo':
            predictions = self.monte_carlo_dropout(X, n_samples)
        else:
            raise ValueError("method must be 'bootstrap' or 'monte_carlo'")
        
        # 计算预测区间
        prediction_intervals = self.calculate_prediction_intervals(predictions)
        
        # 计算不确定性指标
        uncertainty_metrics = self.calculate_uncertainty_metrics(predictions)
        
        # 空间不确定性分析（如果提供了坐标）
        spatial_analysis = None
        if coords is not None:
            spatial_analysis = self.analyze_spatial_uncertainty(coords, predictions)
        
        return {
            'predictions': predictions,
            'prediction_intervals': prediction_intervals,
            'uncertainty_metrics': uncertainty_metrics,
            'spatial_analysis': spatial_analysis,
            'method': method,
            'n_samples': n_samples
        }


# ================= 协同克里金模块 =================
class CokrigingProcessor:
    def __init__(self, terrain):
        self.terrain = terrain
        self.max_points = 500  # 进一步减少最大点数
        self.step = 10  # 增加步长
        self.grid_size = 0.2  # 增加网格大小

    def process_residuals(self, var, coords, residuals):
        try:
            # 使用更少的采样点
            if len(coords) > self.max_points:
                indices = np.random.choice(len(coords), self.max_points, replace=False)
                coords = coords[indices]
                residuals = residuals[indices]

            # 获取地形数据的形状和范围
            grid_shape = self.terrain['elevation'].shape
            x_min, x_max = self.terrain['lons'].min(), self.terrain['lons'].max()
            y_min, y_max = self.terrain['lats'].min(), self.terrain['lats'].max()

            # 创建与地形数据相同形状的网格
            x_grid = np.linspace(x_min, x_max, grid_shape[1])
            y_grid = np.linspace(y_min, y_max, grid_shape[0])
            grid_x, grid_y = np.meshgrid(x_grid, y_grid)

            # 使用简化的变差函数模型
            variogram = {
                'model': 'spherical',
                'range': self.grid_size,
                'sill': np.var(residuals),
                'nugget': 0.1 * np.var(residuals)
            }

            # 分块处理以减少内存使用
            grid = np.zeros_like(grid_x, dtype=np.float32)
            block_size = 100  # 每次处理100x100的块

            for i in range(0, grid_x.shape[0], block_size):
                for j in range(0, grid_x.shape[1], block_size):
                    i_end = min(i + block_size, grid_x.shape[0])
                    j_end = min(j + block_size, grid_x.shape[1])

                    block_x = grid_x[i:i_end, j:j_end]
                    block_y = grid_y[i:i_end, j:j_end]

                    # 处理当前块
                    block_grid = self._process_block(
                        coords, residuals, block_x, block_y, variogram)
                    grid[i:i_end, j:j_end] = block_grid

                    # 清理内存
                    del block_x, block_y, block_grid
                    gc.collect()

            return grid

        except Exception as e:
            print(f"Error in process_residuals: {str(e)}")
            raise

    def _process_block(self, coords, residuals, block_x, block_y, variogram):
        """处理单个网格块"""
        block_grid = np.zeros_like(block_x, dtype=np.float32)

        # 使用向量化操作加速计算
        for i in range(block_x.shape[0]):
            for j in range(block_x.shape[1]):
                point = np.array([block_x[i, j], block_y[i, j]])
                distances = np.sqrt(np.sum((coords - point) ** 2, axis=1))
                weights = self._calculate_weights(distances, variogram)
                block_grid[i, j] = np.sum(weights * residuals)

        return block_grid

    def _calculate_weights(self, distances, variogram):
        """计算权重"""
        weights = np.exp(-distances / variogram['range'])
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            return weights / sum_weights
        return weights


# ================= 主处理系统 =================
class SpatialInterpolationSystem:
    def __init__(self):
        self.data = None
        self.terrain = None
        self.models = {}
        self.client = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and Config.use_gpu and not Config.force_cpu else 'cpu')

    def load_data(self):
        print("1/4.Loading data...")
        self.data = GeoDataLoader.load_meteo_data()
        self.terrain = GeoDataLoader.load_terrain()

        # 获取日期范围
        start_year, start_month, end_year, end_month = Config.get_date_range()

        # 过滤数据到指定日期范围
        date_mask = (
                            (self.data['YYYY'] > start_year) |
                            ((self.data['YYYY'] == start_year) & (self.data['MM'] >= start_month))
                    ) & (
                            (self.data['YYYY'] < end_year) |
                            ((self.data['YYYY'] == end_year) & (self.data['MM'] <= end_month))
                    )
        self.data = self.data[date_mask].copy()

        print(f"数据范围: {start_year}-{start_month:02d} 到 {end_year}-{end_month:02d}")
        print(f"过滤后的数据量: {len(self.data)}")
        
        # 添加坐标范围验证
        print(f"气象站点经度范围: {self.data['Lon'].min():.6f} 到 {self.data['Lon'].max():.6f}")
        print(f"气象站点纬度范围: {self.data['Lat'].min():.6f} 到 {self.data['Lat'].max():.6f}")
        print(f"气象站点数量: {len(self.data)}")

        # 添加地形数据坐标范围验证
        print(f"地形栅格经度范围: {self.terrain['lons'].min():.6f} 到 {self.terrain['lons'].max():.6f}")
        print(f"地形栅格纬度范围: {self.terrain['lats'].min():.6f} 到 {self.terrain['lats'].max():.6f}")
        print(f"地形栅格高程范围: {self.terrain['elevation'].min():.2f} 到 {self.terrain['elevation'].max():.2f}")
        
        # 检查坐标系统
        terrain_crs = self.terrain['profile']['crs']
        is_projected = terrain_crs.is_projected
        
        print(f"地形数据坐标系: {'投影坐标系' if is_projected else '地理坐标系'}")
        print(f"地形数据CRS: {terrain_crs}")
        print(f"气象站点数据坐标系: Albers_Conic_Equal_Area投影坐标系")
        
        # 检查两个坐标系统是否匹配
        if is_projected:
            print("地形数据和气象站点数据都是投影坐标系")
            print(f"气象站点坐标范围:")
            print(f"  X范围: {self.data['Lon'].min():.2f} 到 {self.data['Lon'].max():.2f}")
            print(f"  Y范围: {self.data['Lat'].min():.2f} 到 {self.data['Lat'].max():.2f}")
            
            # 检查CRS是否匹配
            terrain_crs_str = str(terrain_crs).lower()
            if 'albers' in terrain_crs_str or 'conic' in terrain_crs_str:
                print("✓ 地形数据CRS与气象站点数据CRS匹配（Albers投影）")
            else:
                print(f"⚠ 地形数据CRS可能与气象站点数据CRS不匹配")
                print(f"  地形数据CRS: {terrain_crs}")
                print(f"  气象站点数据CRS: Albers_Conic_Equal_Area")
        else:
            print("错误：地形数据是地理坐标系，但气象站点数据是投影坐标系！")
            print("这会导致坐标不匹配问题")
        
        # 直接使用气象站点坐标（假设坐标系匹配）
        coords = self.data[['Lon', 'Lat']].values
        
        # 合并地形高程
        terrain_points = np.column_stack([
            self.terrain['lons'].ravel(),
            self.terrain['lats'].ravel()
        ])

        print(f"开始最近邻搜索匹配...")
        print(f"气象站点坐标示例: {coords[:3]}")
        print(f"地形栅格坐标示例: {terrain_points[:3]}")
        
        nn = NearestNeighbors(n_neighbors=1).fit(terrain_points)
        distances, indices = nn.kneighbors(coords)
        
        print(f"最近邻搜索完成，平均距离: {distances.mean():.6f}")
        print(f"最大距离: {distances.max():.6f}")
        print(f"距离单位: {'米' if is_projected else '度'}")
        
        # 显示每个气象站点的匹配情况
        print(f"\n各气象站点匹配详情:")
        for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
            station_coord = coords[i]
            terrain_coord = terrain_points[idx]
            print(f"  站点{i+1}: 气象站({station_coord[0]:.2f}, {station_coord[1]:.2f}) -> 地形({terrain_coord[0]:.2f}, {terrain_coord[1]:.2f}) 距离={dist:.2f}{'米' if is_projected else '度'}")
        
        if is_projected:
            print(f"距离超过1000米的点数量: {np.sum(distances > 1000)}")
            print(f"距离超过5000米的点数量: {np.sum(distances > 5000)}")
            print(f"距离超过10000米的点数量: {np.sum(distances > 10000)}")
        else:
            print(f"距离超过0.01度的点数量: {np.sum(distances > 0.01)}")
        
        # 检查匹配结果的质量
        if distances.max() > 10000:  # 如果最大距离超过10公里
            print("警告：最大匹配距离过大，可能存在坐标系统不匹配问题！")
            print("建议检查气象站点数据和地形数据的坐标系统是否一致")
        
        # 检查坐标范围重叠情况
        terrain_x_min, terrain_x_max = self.terrain['lons'].min(), self.terrain['lons'].max()
        terrain_y_min, terrain_y_max = self.terrain['lats'].min(), self.terrain['lats'].max()
        station_x_min, station_x_max = self.data['Lon'].min(), self.data['Lon'].max()
        station_y_min, station_y_max = self.data['Lat'].min(), self.data['Lat'].max()
        
        print(f"\n坐标范围比较:")
        print(f"地形栅格 X范围: {terrain_x_min:.2f} 到 {terrain_x_max:.2f}")
        print(f"气象站点 X范围: {station_x_min:.2f} 到 {station_x_max:.2f}")
        print(f"地形栅格 Y范围: {terrain_y_min:.2f} 到 {terrain_y_max:.2f}")
        print(f"气象站点 Y范围: {station_y_min:.2f} 到 {station_y_max:.2f}")
        
        # 检查是否有重叠
        x_overlap = not (station_x_max < terrain_x_min or station_x_min > terrain_x_max)
        y_overlap = not (station_y_max < terrain_y_min or station_y_min > terrain_y_max)
        
        if x_overlap and y_overlap:
            print("✓ 坐标范围有重叠，匹配应该正常")
        else:
            print("⚠ 坐标范围没有重叠！这可能是插值结果位置错误的根本原因")
            if not x_overlap:
                print(f"  X方向无重叠：气象站点X范围({station_x_min:.2f}, {station_x_max:.2f})与地形X范围({terrain_x_min:.2f}, {terrain_x_max:.2f})不重叠")
            if not y_overlap:
                print(f"  Y方向无重叠：气象站点Y范围({station_y_min:.2f}, {station_y_max:.2f})与地形Y范围({terrain_y_min:.2f}, {terrain_y_max:.2f})不重叠")

        # 确保地形高程数据维度匹配（只使用有效区域）
        valid_mask = self.terrain['valid_mask'].ravel()
        valid_terrain_points = terrain_points[valid_mask]
        valid_elevation = self.terrain['elevation'].ravel()[valid_mask]
        
        print(f"有效地形点数: {len(valid_terrain_points)}")
        print(f"气象站点数: {len(coords)}")
        
        # 重新进行最近邻搜索，只使用有效地形点
        nn_valid = NearestNeighbors(n_neighbors=1).fit(valid_terrain_points)
        distances_valid, indices_valid = nn_valid.kneighbors(coords)
        
        print(f"重新匹配后的最大距离: {distances_valid.max():.6f}")
        print(f"距离超过1000米的点数量: {np.sum(distances_valid > 1000)}")
        
        # 检查是否有气象站匹配到无效区域
        invalid_stations = np.sum(distances_valid > 1000)
        if invalid_stations > 0:
            print(f"警告: {invalid_stations}个气象站距离最近有效地形点超过1000米")
            print("这些气象站可能位于水体或无效区域附近")
        
        # 只使用有效区域的高程数据
        self.data['elevation'] = valid_elevation[indices_valid.flatten()]

    def train_all_models(self):
        print("\n2/4.Training models...")
        confidence_uncertainty_results = []  # 存储所有变量的置信分析结果
        
        for var in Config.target_vars:
            print(f"\n=== Training {var} ===")

            # 准备基础数据（仅使用原始5维特征）
            raw_features = self.data[['Lon', 'Lat', 'elevation', 'YYYY', 'MM']].values
            y = self.data[var].values

            # 拆分数据集
            X_train_raw, X_val_raw, y_train, y_val = train_test_split(
                raw_features, y,
                test_size=Config.validation_ratio,
                random_state=Config.random_seed
            )

            # 标准化处理
            scaler = StandardScaler().fit(X_train_raw)
            X_train_scaled = scaler.transform(X_train_raw)
            X_val_scaled = scaler.transform(X_val_raw)

            print(f"训练集特征维度: {X_train_scaled.shape[1]} (应=5)")
            print(f"验证集特征维度: {X_val_scaled.shape[1]} (应=5)")

            # 转换为Tensor
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)

            # 创建数据集
            train_set = TensorDataset(X_train_tensor, y_train_tensor)
            val_set = TensorDataset(X_val_tensor, y_val_tensor)

            # 初始化模型
            model = GeoNetWithPE()
            trainer = GeoTrainer(model)

            # 训练循环
            best_loss = float('inf')
            early_stop = 0
            history = {'train': [], 'val': []}

            for epoch in range(Config.dl_params['epochs']):
                train_loss = trainer.train_epoch(DataLoader(
                    train_set, batch_size=Config.dl_params['batch_size'], shuffle=True))
                val_loss = trainer.validate(DataLoader(
                    val_set, batch_size=Config.dl_params['batch_size'] * 2))

                trainer.scheduler.step(val_loss)
                history['train'].append(train_loss)
                history['val'].append(val_loss)

                # 早停机制
                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stop = 0
                    trainer.save_best_model(var)
                else:
                    early_stop += 1

                print(f"Epoch {epoch + 1}: Train_loss={train_loss:.4f}, Val_loss={val_loss:.4f}")

                if early_stop >= Config.dl_params['early_stop_patience']:
                    print("Early stopping triggered")
                    break

            # 绘制训练历史
            trainer.plot_training_history(var)
            # 打印最终指标
            trainer.print_final_metrics(var)

            # 进行置信分析和不确定性分析
            analysis_results = trainer.analyze_confidence_uncertainty(
                X_val_scaled, y_val, var, scaler
            )
            confidence_uncertainty_results.append(analysis_results)

            # 保存模型
            torch.save(model.state_dict(), os.path.join(Config.model_save_dir, f"{var}_best.pth"))
            # 保存scaler，确保key与Config.target_vars一致
            scaler_dict = {var: scaler for var in Config.target_vars if var in self.models}
            dump(scaler_dict, os.path.join(Config.model_save_dir, "scalers.pkl"))
            # 新增：每个变量单独保存scaler，便于插值模型调用
            from joblib import dump as joblib_dump
            joblib_dump(scaler, os.path.join(Config.model_save_dir, f"{var}_scaler.joblib"))

            self.models[var] = {
                'model': model,
                'scaler': scaler,
                'history': history
            }
        
        # 保存置信分析和不确定性分析结果到CSV
        self.save_confidence_uncertainty_results(confidence_uncertainty_results)

    def save_confidence_uncertainty_results(self, results):
        """保存置信分析和不确定性分析结果到CSV文件"""
        print(f"\n保存置信分析和不确定性分析结果到: {Config.confidence_uncertainty_csv}")
        
        # 准备CSV数据
        csv_data = []
        
        for result in results:
            var_name = result['variable']
            
            # 基础信息
            row = {
                'variable': var_name,
                'training_loss': result['training_metrics']['training_loss'],
                'training_r2': result['training_metrics']['training_r2'],
                'training_mae': result['training_metrics']['training_mae'],
                'training_rmse': result['training_metrics']['training_rmse'],
                'total_epochs': result['training_metrics']['total_epochs'],
                'validation_r2': result['validation_metrics']['r2'],
                'validation_mae': result['validation_metrics']['mae'],
                'validation_rmse': result['validation_metrics']['rmse']
            }
            
            # Bootstrap分析结果
            bootstrap = result['bootstrap_analysis']
            row.update({
                'bootstrap_mean_prediction': bootstrap['mean_prediction'],
                'bootstrap_mean_std': bootstrap['mean_std'],
                'bootstrap_mean_cv': bootstrap['mean_cv'],
                'bootstrap_mean_relative_uncertainty': bootstrap['mean_relative_uncertainty'],
                'bootstrap_mean_prediction_interval_width': bootstrap['mean_prediction_interval_width']
            })
            
            # Monte Carlo分析结果
            mc = result['monte_carlo_analysis']
            row.update({
                'monte_carlo_mean_prediction': mc['mean_prediction'],
                'monte_carlo_mean_std': mc['mean_std'],
                'monte_carlo_mean_cv': mc['mean_cv'],
                'monte_carlo_mean_relative_uncertainty': mc['mean_relative_uncertainty'],
                'monte_carlo_mean_prediction_interval_width': mc['mean_prediction_interval_width']
            })
            
            # 空间不确定性分析结果（如果可用）
            if 'spatial_uncertainty' in result:
                spatial = result['spatial_uncertainty']
                row.update({
                    # Bootstrap空间不确定性
                    'bootstrap_spatial_mean_uncertainty': spatial.get('bootstrap_mean_uncertainty', np.nan),
                    'bootstrap_spatial_max_uncertainty': spatial.get('bootstrap_max_uncertainty', np.nan),
                    'bootstrap_spatial_min_uncertainty': spatial.get('bootstrap_min_uncertainty', np.nan),
                    'bootstrap_spatial_std_uncertainty': spatial.get('bootstrap_std_uncertainty', np.nan),
                    'bootstrap_spatial_median_uncertainty': spatial.get('bootstrap_median_uncertainty', np.nan),
                    'bootstrap_spatial_q25_uncertainty': spatial.get('bootstrap_q25_uncertainty', np.nan),
                    'bootstrap_spatial_q75_uncertainty': spatial.get('bootstrap_q75_uncertainty', np.nan),
                    # Monte Carlo空间不确定性
                    'monte_carlo_spatial_mean_uncertainty': spatial.get('monte_carlo_mean_uncertainty', np.nan),
                    'monte_carlo_spatial_max_uncertainty': spatial.get('monte_carlo_max_uncertainty', np.nan),
                    'monte_carlo_spatial_min_uncertainty': spatial.get('monte_carlo_min_uncertainty', np.nan),
                    'monte_carlo_spatial_std_uncertainty': spatial.get('monte_carlo_std_uncertainty', np.nan),
                    'monte_carlo_spatial_median_uncertainty': spatial.get('monte_carlo_median_uncertainty', np.nan),
                    'monte_carlo_spatial_q25_uncertainty': spatial.get('monte_carlo_q25_uncertainty', np.nan),
                    'monte_carlo_spatial_q75_uncertainty': spatial.get('monte_carlo_q75_uncertainty', np.nan),
                    # 综合空间不确定性
                    'combined_spatial_mean_uncertainty': spatial.get('combined_mean_uncertainty', np.nan),
                    'combined_spatial_max_uncertainty': spatial.get('combined_max_uncertainty', np.nan),
                    'combined_spatial_min_uncertainty': spatial.get('combined_min_uncertainty', np.nan),
                    'combined_spatial_std_uncertainty': spatial.get('combined_std_uncertainty', np.nan),
                    'uncertainty_method_difference': spatial.get('uncertainty_method_difference', np.nan)
                })
            else:
                # 如果没有空间不确定性分析结果，填充NaN值
                spatial_fields = [
                    'bootstrap_spatial_mean_uncertainty', 'bootstrap_spatial_max_uncertainty', 
                    'bootstrap_spatial_min_uncertainty', 'bootstrap_spatial_std_uncertainty',
                    'bootstrap_spatial_median_uncertainty', 'bootstrap_spatial_q25_uncertainty', 
                    'bootstrap_spatial_q75_uncertainty', 'monte_carlo_spatial_mean_uncertainty',
                    'monte_carlo_spatial_max_uncertainty', 'monte_carlo_spatial_min_uncertainty',
                    'monte_carlo_spatial_std_uncertainty', 'monte_carlo_spatial_median_uncertainty',
                    'monte_carlo_spatial_q25_uncertainty', 'monte_carlo_spatial_q75_uncertainty',
                    'combined_spatial_mean_uncertainty', 'combined_spatial_max_uncertainty',
                    'combined_spatial_min_uncertainty', 'combined_spatial_std_uncertainty',
                    'uncertainty_method_difference'
                ]
                for field in spatial_fields:
                    row[field] = np.nan
            
            csv_data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(csv_data)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(Config.confidence_uncertainty_csv), exist_ok=True)
        
        # 保存到CSV
        df.to_csv(Config.confidence_uncertainty_csv, index=False, float_format='%.6f')
        
        print(f"置信分析和不确定性分析结果已保存:")
        print(f"  文件路径: {Config.confidence_uncertainty_csv}")
        print(f"  变量数量: {len(results)}")
        print(f"  分析指标: {len(df.columns)}")
        
        # 打印汇总统计
        print(f"\n=== 置信分析和不确定性分析汇总 ===")
        for _, row in df.iterrows():
            print(f"{row['variable']}:")
            print(f"  训练轮数: {row['total_epochs']}")
            print(f"  训练损失: {row['training_loss']:.4f}")
            print(f"  训练R²: {row['training_r2']:.4f}")
            print(f"  训练MAE: {row['training_mae']:.4f}")
            print(f"  训练RMSE: {row['training_rmse']:.4f}")
            print(f"  验证R²: {row['validation_r2']:.4f}")
            print(f"  验证MAE: {row['validation_mae']:.4f}")
            print(f"  验证RMSE: {row['validation_rmse']:.4f}")
            print(f"  Bootstrap不确定性: {row['bootstrap_mean_std']:.4f}")
            print(f"  Monte Carlo不确定性: {row['monte_carlo_mean_std']:.4f}")
            
            # 显示新的空间不确定性分析结果
            if not np.isnan(row['bootstrap_spatial_mean_uncertainty']):
                print(f"  Bootstrap空间不确定性: {row['bootstrap_spatial_mean_uncertainty']:.4f}")
                print(f"    Bootstrap空间范围: [{row['bootstrap_spatial_min_uncertainty']:.4f}, {row['bootstrap_spatial_max_uncertainty']:.4f}]")
                print(f"    Bootstrap空间中位数: {row['bootstrap_spatial_median_uncertainty']:.4f}")
            
            if not np.isnan(row['monte_carlo_spatial_mean_uncertainty']):
                print(f"  Monte Carlo空间不确定性: {row['monte_carlo_spatial_mean_uncertainty']:.4f}")
                print(f"    Monte Carlo空间范围: [{row['monte_carlo_spatial_min_uncertainty']:.4f}, {row['monte_carlo_spatial_max_uncertainty']:.4f}]")
                print(f"    Monte Carlo空间中位数: {row['monte_carlo_spatial_median_uncertainty']:.4f}")
            
            if not np.isnan(row['combined_spatial_mean_uncertainty']):
                print(f"  综合空间不确定性: {row['combined_spatial_mean_uncertainty']:.4f}")
                print(f"    方法差异: {row['uncertainty_method_difference']:.4f}")
            print()

    def interpolate_all(self):
        print("\n3/4.Performing interpolation...")
        # 配置更保守的Dask客户端
        self.client = Client(
            n_workers=1,
            memory_limit='2GB',
            threads_per_worker=1,
            dashboard_address=None,
            silence_logs=50
        )

        # 获取日期范围
        start_year, start_month, end_year, end_month = Config.get_date_range()

        # 获取需要处理的年月组合（在日期范围内）
        ym_list = self.data[['YYYY', 'MM']].drop_duplicates().values
        ym_list = ym_list[
            ((ym_list[:, 0] > start_year) |
             ((ym_list[:, 0] == start_year) & (ym_list[:, 1] >= start_month))) &
            ((ym_list[:, 0] < end_year) |
             ((ym_list[:, 0] == end_year) & (ym_list[:, 1] <= end_month)))
            ]
        total_ym = len(ym_list)

        print(f"处理时间范围: {start_year}-{start_month:02d} 到 {end_year}-{end_month:02d}")
        print(f"总月数: {total_ym}")
        print(f"使用设备: {self.device}")
        print(f"最终输出CSV路径: {Config.final_output_csv}")

        # 强制使用单线程numpy
        os.environ.update({
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1"
        })

        cokrig = CokrigingProcessor(self.terrain)
        results = []

        try:
            # 主进度条：按年月处理
            with tqdm(total=total_ym, desc="Processing Months") as pbar_main:
                for ym in ym_list:
                    year, month = ym
                    mask = (self.data['YYYY'] == year) & (self.data['MM'] == month)
                    subset = self.data[mask].copy()

                    # 提交每个变量的任务
                    futures = []
                    for var in Config.target_vars:
                        # 提取特征数据
                        X_raw = subset[['Lon', 'Lat', 'elevation', 'YYYY', 'MM']].values.astype(np.float32)

                        # 标准化处理
                        scaler = self.models[var]['scaler']
                        X_scaled = scaler.transform(X_raw)

                        # 预测趋势项
                        model = self.models[var]['model'].to(self.device)
                        with torch.no_grad(), autocast(enabled=Config.use_amp and self.device.type == 'cuda'):
                            trend_tensor = model(torch.FloatTensor(X_scaled).to(self.device))
                            trend = trend_tensor.cpu().numpy().squeeze()

                        # 计算残差
                        residuals = subset[var].values.astype(np.float32) - trend.astype(np.float32)

                        # 提交协同克里金任务
                        future = self.client.submit(
                            cokrig.process_residuals,
                            var,
                            X_raw[:, :2],  # 使用原始坐标
                            residuals
                        )
                        futures.append((var, year, month, future))

                        # 及时清理中间变量
                        del X_raw, X_scaled, trend_tensor, trend
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        gc.collect()

                    # 子进度条：跟踪变量处理
                    completed = 0
                    with tqdm(total=len(futures),
                              desc=f"{year}-{month:02d} Variables",
                              leave=False,
                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [time:{elapsed}]") as pbar_var:

                        # 使用as_completed实时跟踪完成情况
                        from dask.distributed import as_completed
                        for future in as_completed([f for _, _, _, f in futures]):
                            # 获取结果并处理
                            var, y, m, _ = next((item for item in futures if item[3] == future),
                                                (None, year, month, None))
                            try:
                                grid = future.result()
                                final_grid = self._generate_full_grid(var, y, m, grid)
                                self._save_geotiff(var, y, m, final_grid)
                                results.append(self._create_output_row(var, y, m, final_grid))
                                pbar_var.update(1)
                                completed += 1
                            except Exception as e:
                                print(f"\nError processing {var} {y}-{m}: {str(e)}")
                            finally:
                                del future
                                gc.collect()

                            # 实时更新进度描述
                            pbar_var.set_description(
                                f"{year}-{month:02d} Vars (OK:{completed}/Fail:{len(futures) - completed})")

                    # 更新主进度条
                    pbar_main.update(1)
                    pbar_main.set_postfix_str(f"Latest: {year}-{month:02d}")

                    # 每个月份处理完强制清理内存
                    del subset, futures
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()

        finally:
            # 确保结果保存
            if results:
                print(f"\n保存最终插值结果到: {Config.final_output_csv}")
                # 合并结果并保存
                final_df = pd.concat(results)
                # 按经纬度和高程合并，保留年份和月份信息
                # 只聚合实际存在的列
                agg_dict = {}
                for var in Config.target_vars:
                    if var in final_df.columns:
                        agg_dict[var] = 'first'
                
                final_df = final_df.groupby(['Lon', 'Lat', 'altitude', 'YYYY', 'MM']).agg(agg_dict).reset_index()
                # 按经纬度、高程、年份、月份排序
                final_df = final_df.sort_values(['Lon', 'Lat', 'altitude', 'YYYY', 'MM'])
                final_df.to_csv(Config.final_output_csv, index=False)
            self.cleanup()

    @memory_safe
    def _generate_full_grid(self, var, year, month, residuals):
        """生成完整预测网格（分块处理）"""
        try:
            # 确保残差数组形状正确
            if residuals.shape != self.terrain['elevation'].shape:
                print(f"Warning: Reshaping residuals from {residuals.shape} to {self.terrain['elevation'].shape}")
                residuals = residuals.reshape(self.terrain['elevation'].shape)

            chunk_size = Config.gpu_batch_size if self.device.type == 'cuda' else 50000
            lons = self.terrain['lons'].ravel()
            lats = self.terrain['lats'].ravel()
            elevation = self.terrain['elevation'].ravel()
            valid_mask = self.terrain['valid_mask'].ravel()
            
            # 只处理有效数据点
            valid_lons = lons[valid_mask]
            valid_lats = lats[valid_mask]
            valid_elevation = elevation[valid_mask]
            total_points = len(valid_lons)

            print(f"训练插值 - 只处理有效数据点: {total_points}/{len(lons)}")

            final_grid = np.full(len(lons), np.nan, dtype=np.float32)
            valid_predictions = np.empty(total_points, dtype=np.float32)
            model = self.models[var]['model'].to(self.device)

            for i in tqdm(range(0, total_points, chunk_size), desc="Processing grid chunks"):
                chunk = slice(i, min(i + chunk_size, total_points))

                # 构建分块特征（只使用有效数据点）
                grid_features = np.column_stack([
                    valid_lons[chunk],
                    valid_lats[chunk],
                    valid_elevation[chunk],
                    np.full(chunk.stop - chunk.start, year),
                    np.full(chunk.stop - chunk.start, month)
                ])

                # 标准化处理
                X_scaled = self.models[var]['scaler'].transform(grid_features)

                # 使用GPU进行预测
                with torch.no_grad(), autocast(enabled=Config.use_amp and self.device.type == 'cuda'):
                    X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                    trend = model(X_tensor).cpu().numpy().squeeze()

                # 合并趋势与残差（只对有效点）
                valid_predictions[chunk] = trend + residuals.ravel()[valid_mask][chunk]

                # 清理内存
                del grid_features, X_scaled, X_tensor, trend
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

            # 将有效预测结果放回原始网格
            final_grid[valid_mask] = valid_predictions
            return final_grid.reshape(self.terrain['elevation'].shape)

        except Exception as e:
            print(f"Error in _generate_full_grid: {str(e)}")
            raise

    def _save_geotiff(self, var, year, month, data):
        """保存指定分辨率的GeoTIFF文件"""
        try:
            # 确保数据有效
            if data.size == 0 or np.all(np.isnan(data)):
                print(f"Empty data for {var} {year}-{month}, skipping...")
                return

            # 确保输出目录存在
            os.makedirs(Config.output_raster_dir, exist_ok=True)

            # 获取原始地理参考信息
            src_transform = self.terrain['transform']
            src_crs = self.terrain['profile']['crs']
            src_width = self.terrain['elevation'].shape[1]
            src_height = self.terrain['elevation'].shape[0]

            # 检测坐标系
            is_projected = src_crs.is_projected
            print(f"输入栅格坐标系: {'投影坐标系' if is_projected else '地理坐标系'}")

            # 计算输入栅格的空间范围
            x_min = src_transform.c
            y_max = src_transform.f
            x_max = x_min + src_transform.a * src_width
            y_min = y_max + src_transform.e * src_height

            # 计算输出栅格需要的行列数，以保持500m像元大小
            if is_projected:
                # 投影坐标系下直接使用500m
                pixel_size = Config.pixel_size
            else:
                # 地理坐标系下转换为度
                pixel_size = Config.pixel_size / 111320.0

            # 计算新的行列数
            new_width = max(1, int(abs((x_max - x_min) / pixel_size)))
            new_height = max(1, int(abs((y_max - y_min) / pixel_size)))

            print(f"输入栅格大小: {src_width}x{src_height}")
            print(f"输出栅格大小: {new_width}x{new_height}")
            print(f"像元大小: {pixel_size} {'米' if is_projected else '度'}")
            print(f"空间范围: X({x_min:.6f}, {x_max:.6f}), Y({y_min:.6f}, {y_max:.6f})")

            # 创建新的变换矩阵
            new_transform = Affine(pixel_size, 0, x_min,
                                   0, -pixel_size, y_max)

            # 创建目标数组
            resampled_data = np.zeros((new_height, new_width), dtype=np.float32)

            # 执行重采样
            reproject(
                source=data,
                destination=resampled_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=new_transform,
                dst_crs=src_crs,
                resampling=Resampling.bilinear
            )

            # 保存新栅格
            profile = self.terrain['profile'].copy()
            profile.update(
                dtype=rasterio.float32,
                count=1,
                transform=new_transform,
                width=new_width,
                height=new_height,
                nodata=np.nan
            )

            output_path = f"{Config.output_raster_dir}/{var}_{year}_{month}.tif"
            print(f"正在保存栅格到: {output_path}")

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(resampled_data.reshape(1, new_height, new_width).astype(np.float32))
                dst.update_tags(
                    VARIABLE=var,
                    YEAR=year,
                    MONTH=month,
                    UNITS="N/A"
                )

            print(f"成功保存栅格: {output_path}")
            print(f"栅格大小: {new_width}x{new_height}")
            print(f"像元大小: {pixel_size} {'米' if is_projected else '度'}")

        except Exception as e:
            print(f"Error in _save_geotiff: {str(e)}")
            import traceback
            print(traceback.format_exc())
            pass

    def _generate_csv_coordinate_grid(self):
        """根据csv_pixel_size生成CSV输出的坐标网格"""
        if Config.csv_pixel_size is None:
            # 使用原始地形栅格坐标
            return self.terrain['lons'], self.terrain['lats'], self.terrain['elevation']
        
        # 使用自定义分辨率生成坐标网格
        src_transform = self.terrain['transform']
        src_crs = self.terrain['profile']['crs']
        is_projected = src_crs.is_projected
        
        # 计算空间范围（只使用有效区域）
        valid_mask = self.terrain['valid_mask']
        valid_lons = self.terrain['lons'][valid_mask]
        valid_lats = self.terrain['lats'][valid_mask]
        
        x_min = valid_lons.min()
        x_max = valid_lons.max()
        y_min = valid_lats.min()
        y_max = valid_lats.max()
        
        print(f"有效区域空间范围: X({x_min:.2f}, {x_max:.2f}), Y({y_min:.2f}, {y_max:.2f})")
        
        # 计算CSV网格的分辨率
        if is_projected:
            csv_resolution = Config.csv_pixel_size
        else:
            csv_resolution = Config.csv_pixel_size / 111320.0
        
        # 计算新的网格尺寸
        csv_width = max(1, int(abs((x_max - x_min) / csv_resolution)))
        csv_height = max(1, int(abs((y_max - y_min) / csv_resolution)))
        
        print(f"CSV坐标网格: {csv_width}x{csv_height}, 分辨率: {csv_resolution:.2f} {'米' if is_projected else '度'}")
        
        # 生成新的坐标网格（确保覆盖有效区域）
        x_coords = np.linspace(x_min, x_max, csv_width)
        y_coords = np.linspace(y_min, y_max, csv_height)
        csv_lons, csv_lats = np.meshgrid(x_coords, y_coords)
        
        # 从原始地形数据中插值获取对应的高程值（只使用有效数据点）
        original_lons = self.terrain['lons'].ravel()
        original_lats = self.terrain['lats'].ravel()
        original_elevation = self.terrain['elevation'].ravel()
        valid_mask = self.terrain['valid_mask'].ravel()
        
        # 只使用有效数据点进行高程插值
        valid_original_lons = original_lons[valid_mask]
        valid_original_lats = original_lats[valid_mask]
        valid_original_elevation = original_elevation[valid_mask]
        
        # 插值到新的网格
        csv_elevation = griddata(
            (valid_original_lons, valid_original_lats), 
            valid_original_elevation, 
            (csv_lons, csv_lats), 
            method='linear',
            fill_value=np.nan
        )
        
        return csv_lons, csv_lats, csv_elevation

    def _create_output_row(self, var, year, month, grid):
        """创建CSV输出行"""
        # 根据csv_pixel_size生成坐标网格
        csv_lons, csv_lats, csv_elevation = self._generate_csv_coordinate_grid()
        
        if Config.csv_pixel_size is None:
            # 使用原始网格数据，但只输出有效数据点
            valid_mask = self.terrain['valid_mask'].ravel()
            return pd.DataFrame({
                'Lon': csv_lons.ravel()[valid_mask],
                'Lat': csv_lats.ravel()[valid_mask],
                'altitude': csv_elevation.ravel()[valid_mask],
                'YYYY': year,
                'MM': month,
                var: grid.ravel()[valid_mask]
            })
        else:
            # 需要将插值结果重采样到CSV网格
            
            # 原始插值网格的坐标（只使用有效数据点）
            original_lons = self.terrain['lons'].ravel()
            original_lats = self.terrain['lats'].ravel()
            original_values = grid.ravel()
            valid_mask = self.terrain['valid_mask'].ravel()
            
            # 只使用有效数据点进行插值
            valid_original_lons = original_lons[valid_mask]
            valid_original_lats = original_lats[valid_mask]
            valid_original_values = original_values[valid_mask]
            
            # 插值到CSV网格
            csv_values = griddata(
                (valid_original_lons, valid_original_lats), 
                valid_original_values, 
                (csv_lons, csv_lats), 
                method='linear',
                fill_value=np.nan
            )
            
            # 创建有效掩码（非NaN值）
            csv_valid_mask = ~np.isnan(csv_values) & ~np.isnan(csv_elevation)
            
            return pd.DataFrame({
                'Lon': csv_lons.ravel()[csv_valid_mask.ravel()],
                'Lat': csv_lats.ravel()[csv_valid_mask.ravel()],
                'altitude': csv_elevation.ravel()[csv_valid_mask.ravel()],
                'YYYY': year,
                'MM': month,
                var: csv_values.ravel()[csv_valid_mask.ravel()]
            })

    def cleanup(self):
        if self.client:
            self.client.close()


# ================= 主程序 =================
if __name__ == "__main__":
    try:
        # 确保输出目录存在
        Config.create_output_dirs()

        # 配置GPU
        use_gpu = Config.setup_gpu()
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"检测到的GPU数量: {torch.cuda.device_count()}")
            print(f"当前设备: {torch.cuda.current_device()}")
            print(f"设备名称: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存使用比例: {Config.gpu_memory_fraction}")

        # 启动 GPU 监控
        if use_gpu:
            gpu_monitor_active = True
            monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
            monitor_thread.start()

        # 初始化系统
        system = SpatialInterpolationSystem()

        # 设置异常处理
        try:
            print("Starting data loading...")
            system.load_data()
        except Exception as e:
            print(f"Error during data loading: {str(e)}")
            raise

        try:
            print("Starting model training...")
            system.train_all_models()
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise

        try:
            print("Starting interpolation...")
            system.interpolate_all()
        except Exception as e:
            print(f"Error during interpolation: {str(e)}")
            raise

    except Exception as e:
        print(f"Critical error in main program: {str(e)}")
        raise
    finally:
        # 确保资源清理
        try:
            if 'system' in locals():
                system.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

        # 停止 GPU 监控
        gpu_monitor_active = False
        if use_gpu and 'monitor_thread' in locals():
            monitor_thread.join()

        # 强制清理内存
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

    print("Process completed successfully!")