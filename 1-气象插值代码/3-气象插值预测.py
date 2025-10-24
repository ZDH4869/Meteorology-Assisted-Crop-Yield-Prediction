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
用于调用训练好的模型进行气象数据插值
"""
import os
import gc
import time
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import warnings
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import load
from scipy.interpolate import griddata
import psutil  # 添加内存监控
import importlib.util
spec = importlib.util.spec_from_file_location("气象插值模型训练", "3-气象插值模型训练.py")
气象插值模型训练 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(气象插值模型训练)
GeoNetWithPE = 气象插值模型训练.GeoNetWithPE
PositionalEncoder = 气象插值模型训练.PositionalEncoder
import subprocess
import sys
import json
import tempfile
warnings.filterwarnings('ignore')

# 可选：用于获取 GPU 详细信息
try:
    import GPUtil
except ImportError:
    GPUtil = None

# 配置全局进度条样式
tqdm._instances.clear()  # 防止多个实例冲突
tqdm.pandas(
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

# ========== 自动选择pandas或modin.pandas ==========
try:
    import modin.pandas as pd
    print("[INFO] Using Modin (并行pandas) 加速DataFrame操作")
except ImportError:
    import pandas as pd
    print("[INFO] Using standard pandas (如需更快合并可安装modin[ray])")

# ================= 参数配置 =================
class Config:
    """
    配置类：包含所有可调整的参数和路径设置
    """
    # ============= 输入输出路径配置 =============
    # 模型文件路径（训练好的模型存放位置）
    trained_model_dir = r"1-气象插值代码/输出展示/model"
    
    # 地形数据路径（DEM数据）
    input_terrain_tif = r"1-气象插值代码/测试数据/test_tif.tif"
    
    # 输出路径配置
    base_output_dir = r"1-气象插值代码/输出展示/csv"
    model_dir = os.path.join(base_output_dir, "models")  # 用于存放模型副本
    scaler_dir = r"1-气象插值代码/输出展示/model"
    output_raster_dir = os.path.join(base_output_dir, "rasters")
    output_csv_path = os.path.join(base_output_dir, "final.csv")
    final_output_csv = os.path.join(base_output_dir, "csv/test_weather_999.csv")
    model_metrics_csv = os.path.join(base_output_dir, "csv/model_metrics.csv")
    plot_dir = os.path.join(base_output_dir, "plots")

    # ============= 插值参数配置 =============
    # 目标变量列表（需要插值的变量）
    target_vars = ['Tsun_mean',
                   ]
    # 日期范围设置（格式：YYYY-MM）2002 2003
    start_date = "2014-01"  # 开始日期
    end_date = "2014-01"    # 结束日期
    
    # 输出栅格像元大小设置（单位：米）
    pixel_size = 500 # 设置像元大小为500米
    
    # CSV输出坐标间隔设置（单位：米）
    csv_pixel_size = 10000  # None时自动采用地形栅格分辨率，否则使用指定值
    
    # GPU相关参数
    use_gpu = True  # 是否使用GPU
    force_cpu = False  # 是否强制使用CPU
    gpu_memory_fraction = 0.8 # 降低GPU内存使用比例，为系统留出更多内存
    use_amp = True  # 是否使用自动混合精度训练
    
    # 批处理参数（性能优化版本）
    gpu_batch_size_per_device = 32768   # GPU批处理大小
    cpu_batch_size = 50000             # CPU批处理大小（大幅增加）
    chunk_size = 500000                 # 分块大小
    parallel_workers = 2               # 减少进程数，降低开销
    procs_per_gpu = 1                  # 每块GPU分1进程，避免内存竞争
    
    # 控制流程参数
    only_interpolate_to_csv = False  # True 只输出插值CSV后退出；False: 继续后续栅格输出等
    
    @staticmethod
    def create_output_dirs():
        """创建所有必要的输出目录"""
        for dir_path in [Config.output_raster_dir, Config.plot_dir, Config.model_dir]:
            os.makedirs(dir_path, exist_ok=True)
        # 确保CSV输出目录存在
        os.makedirs(os.path.dirname(Config.final_output_csv), exist_ok=True)
        os.makedirs(os.path.dirname(Config.model_metrics_csv), exist_ok=True)

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
        """
        配置GPU设置，自动检测多卡
        """
        if torch.cuda.is_available() and Config.use_gpu and not Config.force_cpu:
            num_gpus = torch.cuda.device_count()
            for gpu_id in range(num_gpus):
                torch.cuda.set_per_process_memory_fraction(Config.gpu_memory_fraction, device=gpu_id)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print(f"检测到 {num_gpus} 块GPU, 已为每块分配 {Config.gpu_memory_fraction*100:.0f}% 显存")
            return True, num_gpus
        print("未检测到可用GPU，使用CPU模式")
        return False, 0

# ================= GPU 监控 =================
gpu_monitor_active = False

def print_memory_usage():
    """打印当前内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    print(f"[内存监控] 进程内存使用: {memory_info.rss / 1024**3:.2f} GB ({memory_percent:.1f}%)")
    
    # 系统总内存
    system_memory = psutil.virtual_memory()
    print(f"[内存监控] 系统总内存: {system_memory.total / 1024**3:.2f} GB")
    print(f"[内存监控] 系统可用内存: {system_memory.available / 1024**3:.2f} GB ({system_memory.percent:.1f}% 已使用)")

def gpu_monitor(interval=5):
    """
    后台线程：定时打印 GPU 使用情况
    """
    global gpu_monitor_active
    while gpu_monitor_active:
        if GPUtil:
            gpus = GPUtil.getGPUs()
            if gpus:
                for gpu in gpus:
                    print(f"[GPU Monitor] GPU{gpu.id} Usage: {gpu.load*100:.1f}% | Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
        else:
            for i in range(torch.cuda.device_count()):
                print(f"[GPU Monitor] GPU{i} Allocated: {torch.cuda.memory_allocated(i)//1024**2} MiB | "
                      f"Cached: {torch.cuda.memory_reserved(i)//1024**2} MiB")
        time.sleep(interval)

# ================= 模型评估类 =================
class ModelEvaluator:
    def __init__(self):
        self.metrics = {
            'model_name': [],
            'max_accuracy': [],
            'min_accuracy': [],
            'mean_accuracy': [],
            'processing_time': [],
            'r2_score': [],
            'rmse': [],
            'mae': []
        }
    
    def add_metrics(self, model_name, predictions, true_values, processing_time):
        """添加模型评估指标"""
        if not model_name:
            print("Warning: Model name is empty, skipping metrics addition.")
            return
        # 计算评估指标
        r2 = r2_score(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        mae = mean_absolute_error(true_values, predictions)
        
        # 更新指标
        self.metrics['model_name'].append(model_name)
        self.metrics['r2_score'].append(r2)
        self.metrics['rmse'].append(rmse)
        self.metrics['mae'].append(mae)
        self.metrics['max_accuracy'].append(1 - rmse/np.mean(true_values))  # 使用RMSE计算相对准确度
        self.metrics['min_accuracy'].append(1 - mae/np.mean(true_values))   # 使用MAE计算相对准确度
        self.metrics['mean_accuracy'].append(1 - np.mean([rmse, mae])/np.mean(true_values))
        self.metrics['processing_time'].append(processing_time)
    
    def save_metrics(self):
        """保存评估指标到CSV文件"""
        df = pd.DataFrame(self.metrics)
        # 添加时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{Config.model_metrics_csv[:-4]}_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n模型评估指标已保存到: {output_path}")
        
        # 绘制评估指标图表
        self._plot_metrics()
    
    def _plot_metrics(self):
        """绘制评估指标图表"""
        if not self.metrics['model_name']:
            print("Warning: No metrics data available for plotting.")
            return
        plt.figure(figsize=(15, 10))
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Metrics', fontsize=16)
        
        # R2分数
        ax1.bar(self.metrics['model_name'], self.metrics['r2_score'])
        ax1.set_title('R² Score')
        ax1.set_xticklabels(self.metrics['model_name'], rotation=45)
        
        # RMSE
        ax2.bar(self.metrics['model_name'], self.metrics['rmse'])
        ax2.set_title('RMSE')
        ax2.set_xticklabels(self.metrics['model_name'], rotation=45)
        
        # MAE
        ax3.bar(self.metrics['model_name'], self.metrics['mae'])
        ax3.set_title('MAE')
        ax3.set_xticklabels(self.metrics['model_name'], rotation=45)
        
        # 处理时间
        ax4.bar(self.metrics['model_name'], self.metrics['processing_time'])
        ax4.set_title('Processing Time (s)')
        ax4.set_xticklabels(self.metrics['model_name'], rotation=45)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{Config.plot_dir}/model_metrics_{timestamp}.png")
        plt.close()

# ========== 多进程变量组+月份批量推理函数 ===========
def process_var_group_all_months(var_group, ym_list, terrain_data, trained_model_dir, scaler_dir, gpu_batch_size_per_device, use_gpu, force_cpu, tmp_dir):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    import torch
    import time
    import pandas as pd
    from joblib import load
    # GeoNetWithPE already imported at the top of the file
    import logging
    import gc
    device = torch.device('cuda:0' if torch.cuda.is_available() and use_gpu and not force_cpu else 'cpu')
    print(f'进程启动: 物理GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}, torch当前设备: {torch.cuda.current_device() if torch.cuda.is_available() else "cpu"}, 变量组: {var_group}')
    try:
        lons = terrain_data['lons'].ravel()
        lats = terrain_data['lats'].ravel()
        elevation = terrain_data['elevation'].ravel()
        valid_mask = terrain_data['valid_mask'].ravel()
        
        # 只处理有效数据点
        valid_lons = lons[valid_mask]
        valid_lats = lats[valid_mask]
        valid_elevation = elevation[valid_mask]
        total_points = len(valid_lons)
        
        print(f"[DEBUG] 总像素数: {len(lons)}, 有效像素数: {total_points}")
        chunk_size = gpu_batch_size_per_device if use_gpu and torch.cuda.is_available() else 100
        model_scaler_dict = {}
        for var in var_group:
            try:
                model_path = f"{trained_model_dir}/{var}_best.pth"
                scaler_path = f"{scaler_dir}/{var}_scaler.joblib"
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model_state = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        model_state = checkpoint['state_dict']
                    else:
                        model_state = checkpoint
                else:
                    model_state = checkpoint
                model = GeoNetWithPE().to(device)
                model.load_state_dict(model_state)
                model.eval()
                scaler = load(scaler_path)
                model_scaler_dict[var] = (model, scaler)
            except Exception as e:
                logging.error(f"[GPU{os.environ['CUDA_VISIBLE_DEVICES']}] Error loading model/scaler for {var}: {str(e)}")
                continue
        for year, month in ym_list:
            for var in var_group:
                if var not in model_scaler_dict:
                    continue
                model, scaler = model_scaler_dict[var]
                try:
                    for i in range(0, total_points, chunk_size):
                        chunk = slice(i, min(i + chunk_size, total_points))
                        grid_features = np.column_stack([
                            valid_lons[chunk],
                            valid_lats[chunk],
                            valid_elevation[chunk],
                            np.full(chunk.stop - chunk.start, year),
                            np.full(chunk.stop - chunk.start, month)
                        ])
                        X_scaled = scaler.transform(grid_features)
                        with torch.no_grad():
                            X_tensor = torch.FloatTensor(X_scaled).to(device)
                            predictions = model(X_tensor).cpu().numpy().squeeze()
                        df_chunk = pd.DataFrame({
                            'Lon': valid_lons[chunk],
                            'Lat': valid_lats[chunk],
                            'altitude': valid_elevation[chunk],
                            'YYYY': year,
                            'MM': month,
                            var: predictions
                        })
                        out_csv = os.path.join(tmp_dir, f'{var}_{year}_{month}.csv')
                        df_chunk.to_csv(out_csv, mode='a', header=(i==0), index=False)
                        del df_chunk, predictions, grid_features, X_scaled, X_tensor
                        gc.collect()
                except Exception as e:
                    logging.error(f"[GPU{os.environ['CUDA_VISIBLE_DEVICES']}] Error processing {var} for {year}-{month}: {str(e)}")
                    continue
    except Exception as e:
        logging.error(f"[GPU{os.environ['CUDA_VISIBLE_DEVICES']}] Error in process_var_group_all_months: {str(e)}")

# 在process_var_group_all_months_entry的推理循环前，定义自动批量调整函数：
def find_max_batch(model, X_scaled, device, init_batch, min_batch=1024):
    batch = init_batch
    while batch >= min_batch:
        try:
            with torch.no_grad():
                for i in range(0, X_scaled.shape[0], batch):
                    X_tensor = torch.FloatTensor(X_scaled[i:i+batch]).to(device, non_blocking=True)
                    _ = model(X_tensor)
                    del X_tensor
                    torch.cuda.empty_cache()
            return batch
        except RuntimeError as e:
            if 'out of memory' in str(e):
                batch //= 2
                torch.cuda.empty_cache()
            else:
                raise
    raise RuntimeError('No safe batch size found!')

# 在每个变量首次推理前，自动探测最大batch_size：
# 替换原有chunk_size = gpu_batch_size_per_device ...
# 在每个变量首次推理时：
def process_var_group_all_months_entry():
    import sys
    import json
    import os
    import torch
    import pandas as pd
    from joblib import load
    # GeoNetWithPE already imported at the top of the file
    import logging
    import gc
    # 读取参数
    param_path = sys.argv[2]  # 修正为sys.argv[2]
    with open(param_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    # 新增：每子进程内设置torch多线程
    torch_num_threads = params.get('torch_num_threads', 1)
    torch.set_num_threads(torch_num_threads)
    task_group = params['task_group']
    terrain_data_path = params['terrain_data_path']
    trained_model_dir = params['trained_model_dir']
    scaler_dir = params['scaler_dir']
    gpu_batch_size_per_device = params['gpu_batch_size_per_device']
    use_gpu = params['use_gpu']
    force_cpu = params['force_cpu']
    output_path = params['output_path']
    # 读取地形数据
    terrain_data = np.load(terrain_data_path, allow_pickle=True).item()
    device = torch.device('cuda:0' if torch.cuda.is_available() and use_gpu and not force_cpu else 'cpu')
    print(f'子进程PID={os.getpid()}  CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}  torch当前设备: {torch.cuda.current_device() if torch.cuda.is_available() else "cpu"}, 任务数: {len(task_group)}')
    try:
        # 获取CSV像素大小参数
        csv_pixel_size = params.get('csv_pixel_size', None)
        
        if csv_pixel_size is None:
            # 使用原始地形栅格坐标
            lons = terrain_data['lons'].ravel()
            lats = terrain_data['lats'].ravel()
            elevation = terrain_data['elevation'].ravel()
            valid_mask = terrain_data['valid_mask'].ravel()
            
            # 只处理有效数据点
            valid_lons = lons[valid_mask]
            valid_lats = lats[valid_mask]
            valid_elevation = elevation[valid_mask]
            total_points = len(valid_lons)
        else:
            # 使用自定义分辨率生成坐标网格
            print(f"[CSV网格] 使用自定义分辨率: {csv_pixel_size}米")
            
            # 从terrain_data重建坐标信息
            lons_data = terrain_data['lons']
            lats_data = terrain_data['lats']
            elevation_data = terrain_data['elevation']
            
            # 确保坐标数据是2D数组
            if lons_data.ndim == 1:
                # 如果是1D数组，需要重新reshape为2D
                # 从elevation的形状推断原始2D形状
                elevation_shape = elevation_data.shape
                lons_2d = lons_data.reshape(elevation_shape)
                lats_2d = lats_data.reshape(elevation_shape)
            else:
                lons_2d = lons_data
                lats_2d = lats_data
            
            # 计算空间范围（只使用有效区域）
            valid_mask_2d = terrain_data['valid_mask']
            valid_lons_2d = lons_2d[valid_mask_2d]
            valid_lats_2d = lats_2d[valid_mask_2d]
            
            x_min, x_max = valid_lons_2d.min(), valid_lons_2d.max()
            y_min, y_max = valid_lats_2d.min(), valid_lats_2d.max()
            
            print(f"[CSV网格] 有效区域空间范围: X({x_min:.2f}, {x_max:.2f}), Y({y_min:.2f}, {y_max:.2f})")
            
            # 计算CSV网格的分辨率（假设是投影坐标系）
            csv_resolution = csv_pixel_size
            
            # 计算新的网格尺寸
            csv_width = max(1, int(abs((x_max - x_min) / csv_resolution)))
            csv_height = max(1, int(abs((y_max - y_min) / csv_resolution)))
            
            print(f"[CSV网格] 大小: {csv_width}x{csv_height}, 分辨率: {csv_resolution:.2f}米")
            
            # 生成新的坐标网格（确保覆盖有效区域）
            x_coords = np.linspace(x_min, x_max, csv_width)
            y_coords = np.linspace(y_min, y_max, csv_height)
            csv_lons, csv_lats = np.meshgrid(x_coords, y_coords)
            
            # 从原始地形数据中插值获取对应的高程值（只使用有效数据点）
            original_lons = lons_2d.ravel()
            original_lats = lats_2d.ravel()
            original_elevation = elevation_data.ravel()
            original_valid_mask = terrain_data['valid_mask'].ravel()
            
            # 只使用有效数据点进行高程插值
            valid_original_lons = original_lons[original_valid_mask]
            valid_original_lats = original_lats[original_valid_mask]
            valid_original_elevation = original_elevation[original_valid_mask]
            
            # 插值到新的网格
            csv_elevation = griddata(
                (valid_original_lons, valid_original_lats), 
                valid_original_elevation, 
                (csv_lons, csv_lats), 
                method='linear',
                fill_value=np.nan
            )
            
            # 创建有效掩码（非NaN值）
            valid_mask = ~np.isnan(csv_elevation)
            
            # 将2D数组ravel为1D，然后使用1D的valid_mask进行索引
            valid_lons = csv_lons.ravel()[valid_mask.ravel()]
            valid_lats = csv_lats.ravel()[valid_mask.ravel()]
            valid_elevation = csv_elevation.ravel()[valid_mask.ravel()]
            total_points = len(valid_lons)
        
        print(f"[DEBUG] CSV像素大小: {csv_pixel_size if csv_pixel_size else '原始分辨率'}, 有效像素数: {total_points}")
        # 使用优化的批处理大小
        if use_gpu and torch.cuda.is_available():
            chunk_size = gpu_batch_size_per_device
        else:
            # CPU模式使用更大的批处理大小
            chunk_size = params.get('cpu_batch_size', 50000)
        model_scaler_dict = {}
        for task in task_group:
            var = task['var']
            year = task['year']
            month = task['month']
            print(f"[DEBUG] 处理变量: {var}, 年: {year}, 月: {month}")
            # 只加载一次模型和scaler
            if var not in model_scaler_dict:
                try:
                    model_path = f"{trained_model_dir}/{var}_best.pth"
                    scaler_path = f"{scaler_dir}/{var}_scaler.joblib"
                    print(f"[DEBUG] 加载模型: {model_path}")
                    print(f"[DEBUG] 加载scaler: {scaler_path}")
                    checkpoint = torch.load(model_path, map_location=device)
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            model_state = checkpoint['model_state_dict']
                        elif 'state_dict' in checkpoint:
                            model_state = checkpoint['state_dict']
                        else:
                            model_state = checkpoint
                    else:
                        model_state = checkpoint
                    model = GeoNetWithPE().to(device)
                    model.load_state_dict(model_state)
                    model.eval()
                    scaler = load(scaler_path)
                    # 自动探测最大batch_size
                    test_features = np.column_stack([
                        valid_lons[:min(10000, total_points)],
                        valid_lats[:min(10000, total_points)],
                        valid_elevation[:min(10000, total_points)],
                        np.full(min(10000, total_points), year),
                        np.full(min(10000, total_points), month)
                    ])
                    X_test = scaler.transform(test_features)
                    max_batch = find_max_batch(model, X_test, device, gpu_batch_size_per_device)
                    print(f"[AUTO-BATCH] {var} {year}-{month} max safe batch: {max_batch}")
                    model_scaler_dict[var] = (model, scaler, max_batch)
                except Exception as e:
                    print(f"[ERROR] 加载模型或scaler失败: {e}")
                    continue
            model, scaler, chunk_size = model_scaler_dict[var]
            try:
                # 优化的批处理循环
                tmpdir = os.path.dirname(param_path)
                parquet_files = []
                
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_gpu and torch.cuda.is_available()):
                    for i in range(0, total_points, chunk_size):
                        chunk = slice(i, min(i + chunk_size, total_points))
                        chunk_size_actual = chunk.stop - chunk.start
                        
                        # 预分配数组，避免重复创建
                        grid_features = np.empty((chunk_size_actual, 5), dtype=np.float32)
                        grid_features[:, 0] = valid_lons[chunk]
                        grid_features[:, 1] = valid_lats[chunk]
                        grid_features[:, 2] = valid_elevation[chunk]
                        grid_features[:, 3] = year
                        grid_features[:, 4] = month
                        
                        X_scaled = scaler.transform(grid_features)
                        X_tensor = torch.FloatTensor(X_scaled).to(device, non_blocking=True)
                        predictions = model(X_tensor).cpu().numpy().squeeze()
                        
                        # 直接创建DataFrame，减少中间步骤
                        df_chunk = pd.DataFrame({
                            'Lon': valid_lons[chunk],
                            'Lat': valid_lats[chunk],
                            'altitude': valid_elevation[chunk],
                            'YYYY': year,
                            'MM': month,
                            var: predictions
                        })
                        
                        # 保存为parquet文件
                        out_parquet = os.path.join(tmpdir, f'{var}_{year}_{month}_{i}.parquet')
                        df_chunk.to_parquet(out_parquet, engine='pyarrow', index=False)
                        parquet_files.append(out_parquet)
                        
                        # 清理内存
                        del df_chunk, predictions, grid_features, X_scaled, X_tensor
                        if i % (chunk_size * 10) == 0:  # 每10个chunk清理一次
                            gc.collect()
                # 每处理完一个任务，写状态文件
                done_flag = os.path.join(tmpdir, f'done_{var}_{year}_{month}.txt')
                with open(done_flag, 'w') as f:
                    f.write('done')
            except Exception as e:
                print(f"[ERROR] 推理失败: {e}")
                continue
        # 清理内存
        del model_scaler_dict, terrain_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"[GPU{os.environ['CUDA_VISIBLE_DEVICES']}] Error in process_var_group_all_months: {str(e)}")

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

            # 获取地形数据的形状和范围（只使用有效区域）
            valid_mask = self.terrain['valid_mask']
            valid_lons = self.terrain['lons'][valid_mask]
            valid_lats = self.terrain['lats'][valid_mask]
            
            grid_shape = self.terrain['elevation'].shape
            x_min, x_max = valid_lons.min(), valid_lons.max()
            y_min, y_max = valid_lats.min(), valid_lats.max()

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


# ================= 插值系统 =================
class InterpolationSystem:
    def __init__(self):
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() and Config.use_gpu and not Config.force_cpu else 1
        self.terrain = None
        self.models = {}
        self.evaluator = ModelEvaluator()
        
    def load_terrain(self):
        """加载地形数据"""
        print("加载地形数据...")
        with rasterio.open(Config.input_terrain_tif) as src:
            terrain = src.read()
            height, width = src.shape
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            lons, lats = rasterio.transform.xy(src.transform, y, x)
            # rasterio.transform.xy返回的是两个一维数组，需要转换为二维数组
            lons = np.array(lons).reshape(height, width)
            lats = np.array(lats).reshape(height, width)
            
            # 获取nodata值
            nodata_value = src.nodata
            if nodata_value is None:
                # 如果没有设置nodata，使用-9999作为默认值
                nodata_value = -9999
            
            # 创建有效数据掩码
            elevation_data = terrain[0].astype(np.float32)
            # 正确处理nodata值，排除异常大的负值和高程为0的区域
            valid_mask = (elevation_data != nodata_value) & (~np.isnan(elevation_data)) & (elevation_data > 0) & (elevation_data < 10000)
            
            print(f"总像素数: {elevation_data.size}")
            print(f"有效像素数: {np.sum(valid_mask)}")
            print(f"无效像素数: {np.sum(~valid_mask)}")
            print(f"有效数据比例: {np.sum(valid_mask) / elevation_data.size * 100:.2f}%")
            
            # 添加坐标范围调试信息
            valid_lons = lons[valid_mask]
            valid_lats = lats[valid_mask]
            print(f"有效区域经度范围: {valid_lons.min():.6f} 到 {valid_lons.max():.6f}")
            print(f"有效区域纬度范围: {valid_lats.min():.6f} 到 {valid_lats.max():.6f}")
            print(f"有效区域高程范围: {elevation_data[valid_mask].min():.2f} 到 {elevation_data[valid_mask].max():.2f}")
            
            # 检查插值点与原始栅格点的对应关系
            print(f"\n插值点诊断:")
            print(f"总栅格点数: {lons.size}")
            print(f"有效栅格点数: {np.sum(valid_mask)}")
            print(f"插值将使用 {np.sum(valid_mask)} 个有效点进行插值")
            
            # 检查栅格分辨率
            if hasattr(src, 'transform') and src.transform:
                pixel_size_x = abs(src.transform.a)
                pixel_size_y = abs(src.transform.e)
                print(f"栅格分辨率: X={pixel_size_x:.2f}m, Y={pixel_size_y:.2f}m")
                
                # 检查插值点密度
                if np.sum(valid_mask) > 0:
                    area_km2 = (valid_lons.max() - valid_lons.min()) * (valid_lats.max() - valid_lats.min()) / 1e6
                    density = np.sum(valid_mask) / area_km2 if area_km2 > 0 else 0
                    print(f"有效区域面积: {area_km2:.2f} km²")
                    print(f"插值点密度: {density:.2f} 点/km²")
                
                # 检查坐标对应关系
                print(f"\n坐标对应关系检查:")
                print(f"栅格左上角 (0,0) 对应地理坐标: ({lons[0,0]:.2f}, {lats[0,0]:.2f})")
                print(f"栅格右下角 ({height-1},{width-1}) 对应地理坐标: ({lons[-1,-1]:.2f}, {lats[-1,-1]:.2f})")
                print(f"栅格中心 ({height//2},{width//2}) 对应地理坐标: ({lons[height//2,width//2]:.2f}, {lats[height//2,width//2]:.2f})")
                
                # 检查是否存在系统性偏移
                if np.sum(valid_mask) > 0:
                    # 找到第一个有效点的位置
                    valid_indices = np.where(valid_mask)
                    first_valid_row, first_valid_col = valid_indices[0][0], valid_indices[1][0]
                    print(f"第一个有效点位置: 栅格({first_valid_row},{first_valid_col}) -> 地理({lons[first_valid_row,first_valid_col]:.2f}, {lats[first_valid_row,first_valid_col]:.2f})")
                    
                    # 检查有效点的分布模式
                    valid_rows = valid_indices[0]
                    valid_cols = valid_indices[1]
                    print(f"有效点行范围: {valid_rows.min()} 到 {valid_rows.max()}")
                    print(f"有效点列范围: {valid_cols.min()} 到 {valid_cols.max()}")
                    
                    # 检查是否存在"避开"模式
                    row_gaps = np.diff(np.sort(valid_rows))
                    col_gaps = np.diff(np.sort(valid_cols))
                    if len(row_gaps) > 0 and len(col_gaps) > 0:
                        print(f"有效点行间距统计: 最小={row_gaps.min()}, 最大={row_gaps.max()}, 平均={row_gaps.mean():.1f}")
                        print(f"有效点列间距统计: 最小={col_gaps.min()}, 最大={col_gaps.max()}, 平均={col_gaps.mean():.1f}")
                        
                        # 检查是否存在规律性间隔
                        if row_gaps.min() > 1:
                            print("⚠ 发现有效点之间存在行间隔，可能存在数据质量问题")
                        if col_gaps.min() > 1:
                            print("⚠ 发现有效点之间存在列间隔，可能存在数据质量问题")
                
                # 检查坐标对应关系
                print(f"\n坐标对应关系检查:")
                print(f"栅格左上角 (0,0) 对应地理坐标: ({lons[0,0]:.2f}, {lats[0,0]:.2f})")
                print(f"栅格右下角 ({height-1},{width-1}) 对应地理坐标: ({lons[-1,-1]:.2f}, {lats[-1,-1]:.2f})")
                print(f"栅格中心 ({height//2},{width//2}) 对应地理坐标: ({lons[height//2,width//2]:.2f}, {lats[height//2,width//2]:.2f})")
                
                # 检查是否存在系统性偏移
                if np.sum(valid_mask) > 0:
                    # 找到第一个有效点的位置
                    valid_indices = np.where(valid_mask)
                    first_valid_row, first_valid_col = valid_indices[0][0], valid_indices[1][0]
                    print(f"第一个有效点位置: 栅格({first_valid_row},{first_valid_col}) -> 地理({lons[first_valid_row,first_valid_col]:.2f}, {lats[first_valid_row,first_valid_col]:.2f})")
                    
                    # 检查有效点的分布模式
                    valid_rows = valid_indices[0]
                    valid_cols = valid_indices[1]
                    print(f"有效点行范围: {valid_rows.min()} 到 {valid_rows.max()}")
                    print(f"有效点列范围: {valid_cols.min()} 到 {valid_cols.max()}")
                    
                    # 检查是否存在"避开"模式
                    row_gaps = np.diff(np.sort(valid_rows))
                    col_gaps = np.diff(np.sort(valid_cols))
                    if len(row_gaps) > 0 and len(col_gaps) > 0:
                        print(f"有效点行间距统计: 最小={row_gaps.min()}, 最大={row_gaps.max()}, 平均={row_gaps.mean():.1f}")
                        print(f"有效点列间距统计: 最小={col_gaps.min()}, 最大={col_gaps.max()}, 平均={col_gaps.mean():.1f}")
                        
                        # 检查是否存在规律性间隔
                        if row_gaps.min() > 1:
                            print("⚠ 发现有效点之间存在行间隔，可能存在数据质量问题")
                        if col_gaps.min() > 1:
                            print("⚠ 发现有效点之间存在列间隔，可能存在数据质量问题")
            
            # 检查坐标系统
            terrain_crs = src.profile['crs']
            is_projected = terrain_crs.is_projected
            print(f"地形数据坐标系: {'投影坐标系' if is_projected else '地理坐标系'}")
            print(f"地形数据CRS: {terrain_crs}")
            print(f"模型学习的是Albers_Conic_Equal_Area投影坐标系")
            
            if is_projected:
                # 检查CRS是否匹配
                terrain_crs_str = str(terrain_crs).lower()
                if 'albers' in terrain_crs_str or 'conic' in terrain_crs_str:
                    print("✓ 地形数据CRS与模型CRS匹配（Albers投影）")
                else:
                    print(f"⚠ 地形数据CRS可能与模型CRS不匹配")
                    print(f"  地形数据CRS: {terrain_crs}")
                    print(f"  模型CRS: Albers_Conic_Equal_Area")
            else:
                print("错误：地形数据是地理坐标系，但模型学习的是投影坐标！")
                print("这会导致插值结果位置错误")
            
            self.terrain = {
                'data': terrain.astype(np.float32),
                'profile': src.profile,
                'transform': src.transform,
                'lons': np.array(lons, dtype=np.float32),
                'lats': np.array(lats, dtype=np.float32),
                'elevation': elevation_data,
                'valid_mask': valid_mask,
                'nodata_value': nodata_value
            }
    
    def load_models(self):
        """加载训练好的模型"""
        print("\n加载模型...")
        num_gpus = self.num_gpus
        for idx, var in enumerate(Config.target_vars):
            device = torch.device(f'cuda:{idx % num_gpus}' if torch.cuda.is_available() and Config.use_gpu and not Config.force_cpu else 'cpu')
            model_path = f"{Config.trained_model_dir}/{var}_best.pth"
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            model_state = checkpoint['model_state_dict']
                        elif 'state_dict' in checkpoint:
                            model_state = checkpoint['state_dict']
                        else:
                            print(f"Warning: Unknown model structure for {var}")
                            model_state = checkpoint
                    else:
                        print(f"Warning: Model file for {var} is not a dictionary")
                        model_state = checkpoint
                    model = GeoNetWithPE().to(device)
                    try:
                        model.load_state_dict(model_state)
                    except Exception as e:
                        print(f"Error loading state dict for {var}: {str(e)}")
                        model = checkpoint
                    model = model.to(device)
                    model.eval()
                    scaler_path = f"{Config.scaler_dir}/{var}_scaler.joblib"
                    if os.path.exists(scaler_path):
                        try:
                            scaler = load(scaler_path)
                        except Exception as e:
                            print(f"Error loading scaler for {var}: {str(e)}")
                            raise RuntimeError(f"Scaler文件损坏: {scaler_path}")
                    else:
                        raise FileNotFoundError(f"未找到scaler文件: {scaler_path}，请将训练时保存的scaler文件放到此目录下！")
                    self.models[var] = {
                        'model': model,
                        'scaler': scaler,
                        'checkpoint': checkpoint,
                        'device': device
                    }
                    print(f"Successfully loaded model for {var} on device {device}")
                except Exception as e:
                    print(f"Error loading model for {var}: {str(e)}")
                    raise
            else:
                print(f"Warning: Model not found for {var} at {model_path}")
                print(f"Available files in model directory:")
                for f in os.listdir(Config.trained_model_dir):
                    print(f"  - {f}")
                raise FileNotFoundError(f"未找到模型文件: {model_path}，请确保训练好的模型文件在此目录下！")
    
    def interpolate(self):
        global gpu_monitor_active
        print("\n开始插值... (多进程分配任务，主进程显示真实推理进度)")
        if not self.models:
            print("Error: No models loaded. 请检查模型和scaler文件是否齐全！")
            return
        
        # 优化：单变量时使用单进程模式，避免多进程开销
        total_vars = len(Config.target_vars)
        start_year, start_month, end_year, end_month = Config.get_date_range()
        total_months = (end_year - start_year) * 12 + (end_month - start_month) + 1
        
        if total_vars == 1 and total_months == 1:
            print("检测到单变量单月份，使用优化的单进程模式")
            self._interpolate_single_process()
            return
            
        # 1. 生成每个进程负责的变量组
        num_gpus = self.num_gpus
        vars_per_proc = (total_vars + num_gpus - 1) // num_gpus
        var_groups = [Config.target_vars[i*vars_per_proc:(i+1)*vars_per_proc] for i in range(num_gpus)]
        
        # 生成年月列表
        ym_list = []
        current_year = start_year
        current_month = start_month
        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            ym_list.append((current_year, current_month))
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        import tempfile, os, json, subprocess, glob
        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存地形数据
            terrain_data_path = os.path.join(tmpdir, 'terrain_data.npy')
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
                valid_mask = (elevation_data != nodata_value) & (~np.isnan(elevation_data)) & (elevation_data > 0)
                
                # 检查坐标系统
                terrain_crs = src.profile['crs']
                is_projected = terrain_crs.is_projected
                
                if is_projected:
                    # 检查CRS是否匹配
                    terrain_crs_str = str(terrain_crs).lower()
                    if 'albers' in terrain_crs_str or 'conic' in terrain_crs_str:
                        print("子进程：✓ 地形数据CRS与模型CRS匹配（Albers投影）")
                    else:
                        print("子进程：⚠ 地形数据CRS可能与模型CRS不匹配")
                else:
                    print("子进程：错误：地形数据是地理坐标系，但模型学习的是投影坐标！")
                
                # 只保存有效数据
                terrain_data = {
                    'lons': np.array(lons, dtype=np.float32),
                    'lats': np.array(lats, dtype=np.float32),
                    'elevation': elevation_data,
                    'valid_mask': valid_mask,
                    'nodata_value': nodata_value
                }
            np.save(terrain_data_path, terrain_data)
            processes = []
            param_files = []
            result_files = []
            for proc_idx, var_group in enumerate(var_groups):
                if not var_group:
                    continue
                # 该进程负责的所有(变量,年,月)任务
                task_group = []
                for var in var_group:
                    for year, month in ym_list:
                        task_group.append({'var': var, 'year': year, 'month': month})
                param = {
                    'task_group': task_group,
                    'terrain_data_path': terrain_data_path,
                    'trained_model_dir': Config.trained_model_dir,
                    'scaler_dir': Config.scaler_dir,
                    'gpu_batch_size_per_device': Config.gpu_batch_size_per_device,
                    'cpu_batch_size': Config.cpu_batch_size,  # 添加CPU批处理大小
                    'use_gpu': Config.use_gpu,
                    'force_cpu': Config.force_cpu,
                    'output_path': os.path.join(tmpdir, f'proc_{proc_idx}_result.pkl'),
                    'torch_num_threads': 2,  # 减少线程数，降低开销
                    'csv_pixel_size': Config.csv_pixel_size  # 添加CSV像素大小参数
                }
                param_path = os.path.join(tmpdir, f'proc_{proc_idx}_param.json')
                with open(param_path, 'w', encoding='utf-8') as f:
                    json.dump(param, f, ensure_ascii=False)
                param_files.append(param_path)
                result_files.append(param['output_path'])
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(proc_idx)
                p = subprocess.Popen([
                    sys.executable, __file__, '--subproc', param_path
                ], env=env)
                processes.append(p)
            # 3. 实时进度条：统计done_*.txt文件
            from tqdm.auto import tqdm
            total_tasks = total_vars * len(ym_list)
            with tqdm(total=total_tasks, desc="推理进度(月份)", ncols=80) as pbar_main:
                import time as _time
                while not all([p.poll() is not None for p in processes]):
                    done_files = glob.glob(os.path.join(tmpdir, 'done_*.txt'))
                    pbar_main.n = len(done_files)
                    pbar_main.refresh()
                    _time.sleep(1)
                # 最后确保进度条到100%
                done_files = glob.glob(os.path.join(tmpdir, 'done_*.txt'))
                pbar_main.n = len(done_files)
                pbar_main.refresh()
            # 4. 收集结果
            var_dfs = []
            import pandas as pd_origin
            for rf in result_files:
                if os.path.exists(rf):
                    obj = pd_origin.read_pickle(rf)
                    print(f"[CHECK] {rf} type: {type(obj)}, len: {len(obj) if hasattr(obj, '__len__') else 'N/A'}")
                    if isinstance(obj, list) and obj:
                        for i, df in enumerate(obj):
                            print(f"[CHECK]   DataFrame {i} shape: {df.shape}")
                else:
                    print(f"[CHECK] {rf} does not exist")
            # 优化：异步收集，减少主进程等待
            import concurrent.futures
            def load_pickle(rf):
                import pandas as pd_origin  # 强制用标准pandas读取pickle，兼容list
                if os.path.exists(rf):
                    obj = pd_origin.read_pickle(rf)
                    if isinstance(obj, pd_origin.DataFrame):
                        return [obj]
                    elif isinstance(obj, list):
                        return obj
                    else:
                        return []
                return []
            with concurrent.futures.ThreadPoolExecutor(max_workers=Config.parallel_workers) as executor:
                futures = [executor.submit(load_pickle, rf) for rf in result_files]
                for f in tqdm(concurrent.futures.as_completed(futures), total=len(result_files), desc="收集子进程结果", ncols=80):
                    group_dfs = f.result()
                    if group_dfs:
                        var_dfs.extend(group_dfs)
            print(f"[CHECK] var_dfs collected: {len(var_dfs)}")
            if var_dfs:
                for i, df in enumerate(var_dfs):
                    print(f"[CHECK]   DataFrame {i} shape: {df.shape}")
            print("所有子进程结果已收集，开始合并数据...")
            # ========== 内存友好分块合并写入 ==========
            import pandas as pd_origin
            import gc
            key_cols = ['Lon', 'Lat', 'altitude', 'YYYY', 'MM']
            # 替换合并和写出部分：
            years = sorted(set([y for y, m in ym_list]))
            key_cols = ['Lon', 'Lat', 'altitude', 'YYYY', 'MM']
            for year in years:
                output_dir = os.path.dirname(Config.final_output_csv)
                os.makedirs(output_dir, exist_ok=True)
                output_path_tpl = os.path.join(output_dir, f"final_interpolated_data66_{year}_{{month}}.parquet")
                csv_output_path = Config.final_output_csv
                with tqdm(total=12, desc=f"合并写出{year}年", ncols=80) as pbar_merge:
                    month_parquet_files = []
                    for month in range(1, 13):
                        dfs = []
                        # 内存优化的月度数据处理
                        month_data = None
                        for idx, var in enumerate(Config.target_vars):
                            files = glob.glob(f"{tmpdir}/{var}_{year}_{month}_*.parquet")
                            if not files:
                                continue
                            
                            # 分块读取和合并，避免内存溢出
                            var_chunks = []
                            for f in files:
                                try:
                                    chunk = pd_origin.read_parquet(f)
                                    var_chunks.append(chunk)
                                    del chunk
                                except Exception as e:
                                    print(f"读取文件 {f} 失败: {e}")
                                    continue
                            
                            if not var_chunks:
                                continue
                                
                            # 合并该变量的所有分块
                            df_var = pd_origin.concat(var_chunks, ignore_index=True)
                            del var_chunks
                            
                            # 只保留必要的列，减少内存使用
                            if month_data is None:
                                # 第一个变量保留所有key列
                                month_data = df_var.copy()
                            else:
                                # 后续变量只保留变量列
                                month_data[var] = df_var[var].values
                            
                            del df_var
                            gc.collect()
                        
                        if month_data is None or month_data.empty:
                            pbar_merge.update(1)
                            continue
                            
                        # 保存月度数据
                        out_parquet = output_path_tpl.format(month=month)
                        df_month = pd_origin.DataFrame(month_data)
                        df_month.to_parquet(out_parquet, engine='pyarrow', index=False)
                        month_parquet_files.append(out_parquet)
                        del df_month, month_data
                        gc.collect()
                        pbar_merge.update(1)
                print(f"已分块写入{year}年所有结果为parquet到: {output_dir}")
                print_memory_usage()  # 监控内存使用
                # ========== 年度合并输出csv（内存优化版本）==========
                # 避免一次性加载所有数据到内存，改用流式处理
                print(f"开始流式合并{year}年数据到CSV...")
                
                # 按指定顺序输出列
                out_cols = key_cols + list(Config.target_vars)
                
                # 分块处理每个月的parquet文件
                chunk_size = Config.chunk_size if hasattr(Config, 'chunk_size') else 500000  # 减小chunk_size
                
                # 先写入CSV头部
                with open(csv_output_path, 'w', newline='') as f:
                    f.write(','.join(out_cols) + '\n')
                
                # 逐月处理，避免内存溢出
                for month_idx, month_file in enumerate(tqdm(month_parquet_files, desc=f"流式处理{year}年数据")):
                    try:
                        # 每处理3个月检查一次内存
                        if month_idx % 3 == 0:
                            print_memory_usage()
                        
                        # 读取单个月的parquet文件
                        df_month = pd_origin.read_parquet(month_file)
                        
                        # 确保列顺序正确
                        available_cols = [col for col in out_cols if col in df_month.columns]
                        df_month = df_month[available_cols]
                        
                        # 分块写入CSV
                        for i in range(0, len(df_month), chunk_size):
                            chunk = df_month.iloc[i:i+chunk_size]
                            chunk.to_csv(csv_output_path, mode='a', header=False, index=False)
                            del chunk
                        
                        del df_month
                        gc.collect()
                        
                    except Exception as e:
                        print(f"处理文件 {month_file} 时出错: {e}")
                        continue
                
                print(f"已输出年度csv: {csv_output_path}")
                gc.collect()

            # 清理内存
            del var_dfs
            gc.collect()

            # 控制流程：如只需插值到CSV则直接return
            if getattr(Config, 'only_interpolate_to_csv', False):
                # 主动关闭GPU监控线程，防止无关输出
                gpu_monitor_active = False
                if 'monitor_thread' in globals():
                    monitor_thread.join()
                print("已按配置跳过后续步骤，仅输出插值CSV。")
                return
            
            # ========== 生成栅格文件 ==========
            print("开始生成栅格文件...")
            self._generate_raster_files(ym_list)
            self.evaluator.save_metrics()

    def _interpolate_single_process(self):
        """优化的单进程插值模式，用于单变量单月份的情况"""
        try:
            import tempfile
            import os
            import glob
            from scipy.interpolate import griddata
            
            var = Config.target_vars[0]
            start_year, start_month, end_year, end_month = Config.get_date_range()
            
            print(f"单进程模式处理: {var}, {start_year}-{start_month}")
            print(f"CSV像素大小: {Config.csv_pixel_size if Config.csv_pixel_size else '原始分辨率'}")
            print(f"只输出CSV: {Config.only_interpolate_to_csv}")
            
            # 获取模型和scaler
            model_info = self.models[var]
            model = model_info['model']
            scaler = model_info['scaler']
            device = model_info['device']
            
            # 确定批处理大小
            if device.type == 'cuda':
                chunk_size = Config.gpu_batch_size_per_device
            else:
                chunk_size = Config.cpu_batch_size
            
            print(f"使用批处理大小: {chunk_size}")
            
            # 准备地形数据 - 支持csv_pixel_size
            if Config.csv_pixel_size is None:
                # 使用原始地形栅格坐标
                lons = self.terrain['lons'].ravel()
                lats = self.terrain['lats'].ravel()
                elevation = self.terrain['elevation'].ravel()
                valid_mask = self.terrain['valid_mask'].ravel()
                
                # 只处理有效数据点
                valid_lons = lons[valid_mask]
                valid_lats = lats[valid_mask]
                valid_elevation = elevation[valid_mask]
                total_points = len(valid_lons)
            else:
                # 使用自定义分辨率生成坐标网格
                print(f"[CSV网格] 使用自定义分辨率: {Config.csv_pixel_size}米")
                
                # 从terrain_data重建坐标信息
                lons_data = self.terrain['lons']
                lats_data = self.terrain['lats']
                elevation_data = self.terrain['elevation']
                
                # 确保坐标数据是2D数组
                if lons_data.ndim == 1:
                    elevation_shape = elevation_data.shape
                    lons_2d = lons_data.reshape(elevation_shape)
                    lats_2d = lats_data.reshape(elevation_shape)
                else:
                    lons_2d = lons_data
                    lats_2d = lats_data
                
                # 计算空间范围（只使用有效区域）
                valid_mask_2d = self.terrain['valid_mask']
                valid_lons_2d = lons_2d[valid_mask_2d]
                valid_lats_2d = lats_2d[valid_mask_2d]
                
                x_min, x_max = valid_lons_2d.min(), valid_lons_2d.max()
                y_min, y_max = valid_lats_2d.min(), valid_lats_2d.max()
                
                print(f"[CSV网格] 有效区域空间范围: X({x_min:.2f}, {x_max:.2f}), Y({y_min:.2f}, {y_max:.2f})")
                
                # 计算CSV网格的分辨率
                csv_resolution = Config.csv_pixel_size
                
                # 计算新的网格尺寸
                csv_width = max(1, int(abs((x_max - x_min) / csv_resolution)))
                csv_height = max(1, int(abs((y_max - y_min) / csv_resolution)))
                
                print(f"[CSV网格] 大小: {csv_width}x{csv_height}, 分辨率: {csv_resolution:.2f}米")
                
                # 生成新的坐标网格
                x_coords = np.linspace(x_min, x_max, csv_width)
                y_coords = np.linspace(y_min, y_max, csv_height)
                csv_lons, csv_lats = np.meshgrid(x_coords, y_coords)
                
                # 从原始地形数据中插值获取对应的高程值
                original_lons = lons_2d.ravel()
                original_lats = lats_2d.ravel()
                original_elevation = elevation_data.ravel()
                original_valid_mask = self.terrain['valid_mask'].ravel()
                
                # 只使用有效数据点进行高程插值
                valid_original_lons = original_lons[original_valid_mask]
                valid_original_lats = original_lats[original_valid_mask]
                valid_original_elevation = original_elevation[original_valid_mask]
                
                # 插值到新的网格
                csv_elevation = griddata(
                    (valid_original_lons, valid_original_lats), 
                    valid_original_elevation, 
                    (csv_lons, csv_lats), 
                    method='linear',
                    fill_value=np.nan
                )
                
                # 创建有效掩码（非NaN值）
                valid_mask = ~np.isnan(csv_elevation)
                
                # 将2D数组ravel为1D
                valid_lons = csv_lons.ravel()[valid_mask.ravel()]
                valid_lats = csv_lats.ravel()[valid_mask.ravel()]
                valid_elevation = csv_elevation.ravel()[valid_mask.ravel()]
                total_points = len(valid_lons)
            
            print(f"总有效点数: {total_points}")
            
            # 创建临时目录
            with tempfile.TemporaryDirectory() as tmpdir:
                parquet_files = []
                
                # 优化的批处理循环
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=Config.use_amp and device.type == 'cuda'):
                    for i in tqdm(range(0, total_points, chunk_size), desc=f"处理{var}"):
                        chunk = slice(i, min(i + chunk_size, total_points))
                        chunk_size_actual = chunk.stop - chunk.start
                        
                        # 预分配数组
                        grid_features = np.empty((chunk_size_actual, 5), dtype=np.float32)
                        grid_features[:, 0] = valid_lons[chunk]
                        grid_features[:, 1] = valid_lats[chunk]
                        grid_features[:, 2] = valid_elevation[chunk]
                        grid_features[:, 3] = start_year
                        grid_features[:, 4] = start_month
                        
                        X_scaled = scaler.transform(grid_features)
                        X_tensor = torch.FloatTensor(X_scaled).to(device, non_blocking=True)
                        predictions = model(X_tensor).cpu().numpy().squeeze()
                        
                        # 创建DataFrame
                        df_chunk = pd.DataFrame({
                            'Lon': valid_lons[chunk],
                            'Lat': valid_lats[chunk],
                            'altitude': valid_elevation[chunk],
                            'YYYY': start_year,
                            'MM': start_month,
                            var: predictions
                        })
                        
                        # 保存为parquet文件
                        out_parquet = os.path.join(tmpdir, f'{var}_{start_year}_{start_month}_{i}.parquet')
                        df_chunk.to_parquet(out_parquet, engine='pyarrow', index=False)
                        parquet_files.append(out_parquet)
                        
                        # 清理内存
                        del df_chunk, predictions, grid_features, X_scaled, X_tensor
                        if i % (chunk_size * 5) == 0:  # 每5个chunk清理一次
                            gc.collect()
                
                # 合并所有parquet文件
                print("合并结果文件...")
                dfs = []
                for parquet_file in parquet_files:
                    df = pd.read_parquet(parquet_file)
                    dfs.append(df)
                
                final_df = pd.concat(dfs, ignore_index=True)
                
                # 保存最终结果
                os.makedirs(os.path.dirname(Config.final_output_csv), exist_ok=True)
                final_df.to_csv(Config.final_output_csv, index=False)
                
                print(f"单进程模式完成，结果保存到: {Config.final_output_csv}")
                print(f"总处理点数: {len(final_df)}")
                
                # 根据only_interpolate_to_csv参数决定是否继续
                if Config.only_interpolate_to_csv:
                    print("已按配置跳过后续步骤，仅输出插值CSV。")
                    return
                
                # 生成栅格文件
                print("开始生成栅格文件...")
                self._generate_raster_files([(start_year, start_month)])
                
        except Exception as e:
            print(f"单进程模式出错: {e}")
            import traceback
            print(traceback.format_exc())

    def _generate_raster_files(self, ym_list):
        """生成栅格文件（参考训练代码逻辑，直接对栅格进行插值）"""
        try:
            print("开始生成栅格文件...")
            
            # 确保输出目录存在
            os.makedirs(Config.output_raster_dir, exist_ok=True)
            
            # 直接对栅格进行插值（参考训练代码逻辑）
            for var in Config.target_vars:
                print(f"处理变量: {var}")
                
                for year, month in ym_list:
                    try:
                        # 直接对栅格进行插值（参考训练代码逻辑）
                        raster_data = self._generate_prediction_grid(var, year, month)
                        
                        # 保存栅格文件
                        self._save_geotiff(var, year, month, raster_data)
                        print(f"已生成栅格: {var}_{year}_{month}.tif")
                        
                    except Exception as e:
                        print(f"生成栅格文件 {var}_{year}_{month} 时出错: {e}")
                        continue
                        
        except Exception as e:
            print(f"生成栅格文件时出错: {e}")

    def _csv_to_raster(self, df, var):
        """将CSV数据转换为栅格格式（参考训练代码逻辑，只生成插值区域）"""
        try:
            # 获取地形数据的形状
            grid_shape = self.terrain['elevation'].shape
            raster_data = np.full(grid_shape, np.nan, dtype=np.float32)
            
            # 获取有效数据掩码
            valid_mask = self.terrain['valid_mask']
            valid_indices = np.where(valid_mask)
            
            print(f"CSV转栅格: CSV数据点数量({len(df)})与有效栅格点数量({len(valid_indices[0])})")
            
            # 参考训练代码逻辑：直接使用CSV数据生成栅格，不进行插值
            # 获取栅格坐标
            lons_2d = self.terrain['lons']
            lats_2d = self.terrain['lats']
            
            # 预先计算有效位置（避免在循环中重复计算）
            valid_positions = np.where(valid_mask)
            
            # 将CSV数据映射到栅格（只对有效区域）
            for idx, row in df.iterrows():
                lon, lat = row['Lon'], row['Lat']
                value = row[var]
                
                # 找到最接近的有效栅格点
                distances = np.sqrt((lons_2d[valid_mask] - lon)**2 + (lats_2d[valid_mask] - lat)**2)
                min_idx = np.argmin(distances)
                
                # 如果距离足够近，则赋值
                if distances[min_idx] < 1000:  # 1km容差
                    grid_row, grid_col = valid_positions[0][min_idx], valid_positions[1][min_idx]
                    raster_data[grid_row, grid_col] = value
            
            # 统计结果
            valid_count = np.sum(~np.isnan(raster_data))
            print(f"CSV转栅格: 成功映射{valid_count}个有效点")
            
            return raster_data
            
        except Exception as e:
            print(f"CSV转栅格时出错: {e}")
            return np.full(self.terrain['elevation'].shape, np.nan, dtype=np.float32)

    def _generate_prediction_grid(self, var, year, month):
        """生成预测网格（只使用有效区域）"""
        try:
            device = self.models[var]['device']
            chunk_size = Config.gpu_batch_size_per_device if device.type == 'cuda' else 50000
            lons = self.terrain['lons'].ravel()
            lats = self.terrain['lats'].ravel()
            elevation = self.terrain['elevation'].ravel()
            valid_mask = self.terrain['valid_mask'].ravel()
            
            # 只处理有效数据点
            valid_lons = lons[valid_mask]
            valid_lats = lats[valid_mask]
            valid_elevation = elevation[valid_mask]
            total_points = len(valid_lons)
            
            print(f"预测插值 - 只处理有效数据点: {total_points}/{len(lons)}")
            
            # 创建最终网格，只填充有效位置
            final_grid = np.full(len(lons), np.nan, dtype=np.float32)
            valid_predictions = np.empty(total_points, dtype=np.float32)
            
            model = self.models[var]['model']
            for i in tqdm(range(0, total_points, chunk_size), desc=f"Processing {var} grid chunks"):
                chunk = slice(i, min(i + chunk_size, total_points))
                grid_features = np.column_stack([
                    valid_lons[chunk],
                    valid_lats[chunk],
                    valid_elevation[chunk],
                    np.full(chunk.stop - chunk.start, year),
                    np.full(chunk.stop - chunk.start, month)
                ])
                X_scaled = self.models[var]['scaler'].transform(grid_features)
                with torch.no_grad(), autocast(enabled=Config.use_amp and device.type == 'cuda'):
                    X_tensor = torch.FloatTensor(X_scaled).to(device)
                    predictions = model(X_tensor).cpu().numpy().squeeze()
                valid_predictions[chunk] = predictions
                del grid_features, X_scaled, X_tensor, predictions
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
            
            # 将有效预测结果放回原始网格
            final_grid[valid_mask] = valid_predictions
            return final_grid.reshape(self.terrain['elevation'].shape)
        except Exception as e:
            print(f"Error in _generate_prediction_grid: {str(e)}")
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

            # 执行重采样（使用最近邻方法以保留稀疏数据）
            reproject(
                source=data,
                destination=resampled_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=new_transform,
                dst_crs=src_crs,
                resampling=Resampling.nearest
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
            lons = self.terrain['lons'].ravel()
            lats = self.terrain['lats'].ravel()
            elevation = self.terrain['elevation'].ravel()
            valid_mask = self.terrain['valid_mask'].ravel()
            
            # 只输出有效数据点
            valid_lons = lons[valid_mask]
            valid_lats = lats[valid_mask]
            valid_elevation = elevation[valid_mask]
            valid_grid = grid.ravel()[valid_mask]
            
            return pd.DataFrame({
                'Lon': valid_lons,
                'Lat': valid_lats,
                'altitude': valid_elevation,
                'YYYY': year,
                'MM': month,
                var: valid_grid
            })
        else:
            # 需要将插值结果重采样到CSV网格
            # 原始插值网格的坐标
            original_lons = self.terrain['lons'].ravel()
            original_lats = self.terrain['lats'].ravel()
            original_values = grid.ravel()
            
            # 插值到CSV网格
            csv_values = griddata(
                (original_lons, original_lats), 
                original_values, 
                (csv_lons, csv_lats), 
                method='linear',
                fill_value=np.nan
            )
            
            return pd.DataFrame({
                'Lon': csv_lons.ravel(),
                'Lat': csv_lats.ravel(),
                'altitude': csv_elevation.ravel(),
                'YYYY': year,
                'MM': month,
                var: csv_values.ravel()
            })

# ========== 主程序 =================
if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == '--subproc':
        process_var_group_all_months_entry()
        sys.exit(0)
    try:
        # 确保输出目录存在
        Config.create_output_dirs()
        
        # 配置GPU
        gpu_enabled, num_gpus = Config.setup_gpu()
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"检测到的GPU数量: {torch.cuda.device_count()}")
            print(f"当前设备: {torch.cuda.current_device()}")
            print(f"设备名称: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存使用比例: {Config.gpu_memory_fraction}")
        
        # 启动 GPU 监控
        if gpu_enabled:
            gpu_monitor_active = True
            monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
            monitor_thread.start()
        
        # 初始化系统
        system = InterpolationSystem()
        
        # 加载数据
        system.load_terrain()
        system.load_models()
        
        # 执行插值
        system.interpolate()
        
    except Exception as e:
        print(f"Critical error in main program: {str(e)}")
        raise
    finally:
        # 停止 GPU 监控
        gpu_monitor_active = False
        if gpu_enabled and 'monitor_thread' in locals():
            monitor_thread.join()
        
        # 清理内存
        gc.collect()
        if gpu_enabled:
            torch.cuda.empty_cache()
    
    print("Process completed successfully!") 