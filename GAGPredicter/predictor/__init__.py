"""
糖链序列切割位点预测器模块

本模块提供基于GRU集成模型的糖链序列切割位点预测功能。

主要组件：
- EnsemblePredictor: 集成预测器类
- predict_sequence: 单序列预测函数
- predict_batch: 批量预测函数
"""

from .predictor import EnsemblePredictor, create_predictor, predict_sequence, predict_sequences
from .model_loader import load_ensemble_model
from .utils import validate_sequence, format_prediction_result

__version__ = "1.0.0"
__author__ = "AI解谱项目团队"

__all__ = [
    'EnsemblePredictor',
    'create_predictor',
    'predict_sequence', 
    'predict_sequences',
    'load_ensemble_model', 
    'validate_sequence',
    'format_prediction_result'
] 