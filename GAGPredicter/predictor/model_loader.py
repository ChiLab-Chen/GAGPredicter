"""
模型加载器模块

负责加载和管理集成模型
"""

import torch
import pickle
import os
import glob
from typing import List, Dict, Optional, Tuple
import sys

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.model import DynamicCleavageRNN


def find_latest_ensemble_model(checkpoints_dir: Optional[str] = None) -> str:
    """
    查找最新的集成模型信息文件
    
    参数:
        checkpoints_dir: 检查点目录路径，如果为None则使用默认路径
        
    返回:
        最新集成模型信息文件的完整路径
        
    异常:
        FileNotFoundError: 如果未找到任何集成模型文件
    """
    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(project_root, 'models', 'checkpoints')
    
    ensemble_files = glob.glob(os.path.join(checkpoints_dir, 'ensemble_info_*.pkl'))
    
    if not ensemble_files:
        raise FileNotFoundError(
            f"未找到集成模型信息文件，请检查目录: {checkpoints_dir}\n"
            "请确保已运行 02_retrain.ipynb 生成集成模型"
        )
    
    # 按文件名排序，取最新的（时间戳最大的）
    ensemble_files.sort()
    latest_file = ensemble_files[-1]
    
    return latest_file


def load_ensemble_info(ensemble_info_path: Optional[str] = None) -> Dict:
    """
    加载集成模型信息
    
    参数:
        ensemble_info_path: 集成模型信息文件路径，如果为None则自动查找最新文件
        
    返回:
        集成模型信息字典
    """
    if ensemble_info_path is None:
        ensemble_info_path = find_latest_ensemble_model()
    
    print(f"加载集成模型信息: {ensemble_info_path}")
    
    with open(ensemble_info_path, 'rb') as f:
        ensemble_info = pickle.load(f)
    
    return ensemble_info


def load_fold_models(ensemble_info: Dict) -> List[DynamicCleavageRNN]:
    """
    加载所有fold模型
    
    参数:
        ensemble_info: 集成模型信息字典
        
    返回:
        加载的模型列表
    """
    models = []
    model_params = ensemble_info['model_params']
    fold_model_paths = ensemble_info['model_paths']
    
    print(f"加载 {len(fold_model_paths)} 个fold模型...")
    
    for i, model_path in enumerate(fold_model_paths):
        print(f"  加载 Fold {i+1}: {os.path.basename(model_path)}")
        
        # 创建模型实例
        model = DynamicCleavageRNN(**model_params)
        
        # 加载预训练权重
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()  # 设置为评估模式
        
        models.append(model)
    
    return models


def load_ensemble_model(ensemble_info_path: Optional[str] = None) -> Tuple[List[DynamicCleavageRNN], Dict]:
    """
    一次性加载完整的集成模型
    
    参数:
        ensemble_info_path: 集成模型信息文件路径
        
    返回:
        (models, ensemble_info): 模型列表和集成信息
    """
    # 加载集成模型信息
    ensemble_info = load_ensemble_info(ensemble_info_path)
    
    # 加载所有fold模型
    models = load_fold_models(ensemble_info)
    
    print(f"✅ 集成模型加载完成！包含 {len(models)} 个fold模型")
    
    # 显示集成模型性能统计
    avg_metrics = ensemble_info['avg_metrics']
    std_metrics = ensemble_info['std_metrics']
    
    print(f"\n集成模型平均性能:")
    print(f"AUC-ROC: {avg_metrics['auc_roc']:.4f} ± {std_metrics['auc_roc_std']:.4f}")
    print(f"F1: {avg_metrics['f1']:.4f} ± {std_metrics['f1_std']:.4f}")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy_std']:.4f}")
    
    return models, ensemble_info


def validate_model_files(ensemble_info: Dict) -> bool:
    """
    验证模型文件是否存在
    
    参数:
        ensemble_info: 集成模型信息字典
        
    返回:
        所有模型文件是否都存在
    """
    fold_model_paths = ensemble_info['model_paths']
    
    missing_files = []
    for model_path in fold_model_paths:
        if not os.path.exists(model_path):
            missing_files.append(model_path)
    
    if missing_files:
        print(f"❌ 以下模型文件缺失:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True


def get_model_info(ensemble_info_path: Optional[str] = None) -> Dict:
    """
    获取模型信息概要
    
    参数:
        ensemble_info_path: 集成模型信息文件路径
        
    返回:
        模型信息概要字典
    """
    ensemble_info = load_ensemble_info(ensemble_info_path)
    
    info = {
        "model_count": len(ensemble_info['model_paths']),
        "training_timestamp": ensemble_info.get('training_timestamp', 'Unknown'),
        "selection_metric": ensemble_info.get('selection_metric', 'Unknown'),
        "avg_performance": ensemble_info['avg_metrics'],
        "model_params": ensemble_info['model_params'],
        "vocab_size": len(ensemble_info['vocab_mapping']),
        "files_exist": validate_model_files(ensemble_info)
    }
    
    return info 