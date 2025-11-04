"""
集成预测器核心模块

提供基于GRU集成模型的糖链序列切割位点预测功能
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import os
import sys
import pickle
import glob
import pandas as pd
from datetime import datetime
import re

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入模型类
try:
    from models.model import DynamicCleavageRNN
except ImportError:
    # 兼容不同的导入路径
    sys.path.append(os.path.join(project_root, 'models'))
    from model import DynamicCleavageRNN

# 导入工具函数
from .utils import (
    validate_sequence,
    parse_sequence_units,
    calculate_expected_labels
)




class EnsemblePredictor:
    """集成预测器类 - 增强版本"""
    
    def __init__(self, ensemble_info_path=None, verbose=True):
        """
        初始化集成预测器
        
        参数:
        - ensemble_info_path: 集成模型信息文件路径
        - verbose: 是否显示详细信息
        """
        self.verbose = verbose
        
        if ensemble_info_path is None:
            ensemble_info_path = self._find_latest_ensemble_model()
        
        if self.verbose:
            print(f"加载集成模型信息: {ensemble_info_path}")
        
        # 加载集成模型元信息
        with open(ensemble_info_path, 'rb') as f:
            self.ensemble_info = pickle.load(f)
        
        # 提取关键信息
        self.model_params = self.ensemble_info['model_params']
        self.vocab_mapping = self.ensemble_info['vocab_mapping']
        self.fold_model_paths = self.ensemble_info['model_paths']

        # 修正模型路径到当前项目的 models/checkpoints（当原路径不存在时）
        corrected_paths = []
        checkpoints_dir = os.path.join(project_root, 'models', 'checkpoints')
        for p in self.fold_model_paths:
            if os.path.exists(p):
                corrected_paths.append(p)
            else:
                candidate = os.path.join(checkpoints_dir, os.path.basename(p))
                if os.path.exists(candidate):
                    corrected_paths.append(candidate)
                else:
                    raise FileNotFoundError(f"模型文件不存在: {p} (也尝试: {candidate})")
        self.fold_model_paths = corrected_paths
        
        # 加载所有fold模型
        self.models = []
        if self.verbose:
            print(f"加载 {len(self.fold_model_paths)} 个fold模型...")
        
        for i, model_path in enumerate(self.fold_model_paths):
            if self.verbose:
                print(f"  加载 Fold {i+1}: {os.path.basename(model_path)}")
            
            # 创建模型实例
            model = DynamicCleavageRNN(**self.model_params)
            
            # 加载预训练权重
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()  # 设置为评估模式
            
            self.models.append(model)
        
        if self.verbose:
            print(f"✅ 集成模型加载完成！包含 {len(self.models)} 个fold模型")
            
            # 显示集成模型性能统计
            avg_metrics = self.ensemble_info['avg_metrics']
            std_metrics = self.ensemble_info['std_metrics']
            
            print(f"\n集成模型平均性能:")
            print(f"AUC-ROC: {avg_metrics['auc_roc']:.4f} ± {std_metrics['auc_roc_std']:.4f}")
            print(f"F1: {avg_metrics['f1']:.4f} ± {std_metrics['f1_std']:.4f}")
            print(f"Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy_std']:.4f}")
    
    def _find_latest_ensemble_model(self):
        """查找最新的集成模型信息文件"""
        try:
            # 在Python脚本中使用__file__
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..'))
        except NameError:
            # 在Jupyter notebook中使用当前工作目录
            project_root = os.path.abspath('..')
        
        checkpoints_dir = os.path.join(project_root, 'models', 'checkpoints')
        ensemble_files = glob.glob(os.path.join(checkpoints_dir, 'ensemble_info_*.pkl'))
        
        if not ensemble_files:
            raise FileNotFoundError(
                "未找到集成模型信息文件，请先运行 02_retrain.ipynb 生成集成模型"
            )
        
        # 按文件名排序，取最新的
        ensemble_files.sort()
        latest_file = ensemble_files[-1]
        
        if self.verbose:
            print(f"自动选择最新的集成模型: {os.path.basename(latest_file)}")
        
        return latest_file
    
    def predict_sequence(self, sequence_str: str, threshold: float = 0.5) -> Tuple[List[int], List[float], List[List[float]]]:
        """
        预测单个糖链序列的切割位点
        
        这个函数是核心预测接口，处理单个糖链序列并返回集成预测结果。
        
        参数:
            sequence_str: 糖链序列字符串，格式如 "dHexD2-O-HexNS-O-HexD2"
            threshold: 二分类决策阈值，默认0.5
        
        返回:
            predictions: 预测结果列表，每个元素为0或1
            probabilities: 预测概率列表，每个元素为0-1之间的浮点数
            ensemble_probs: 每个fold的详细预测概率，用于分析fold间一致性
        
        处理流程:
            1. 验证序列格式
            2. 解析糖链序列格式
            3. 计算预测位置数量
            4. 转换为模型输入格式
            5. 对每个fold模型进行预测
            6. 集成所有fold的预测结果
        """
        try:
            # 序列格式验证
            is_valid, error_msg = validate_sequence(sequence_str)
            if not is_valid:
                raise ValueError(f"序列格式错误: {error_msg}")
            
            # 解析糖链序列格式
            units = parse_sequence_units(sequence_str)

            # 对糖单元进行标准化以匹配词汇表（处理常见别名/后缀）
            try:
                from .utils import normalize_units_to_vocab
                units = normalize_units_to_vocab(units, self.vocab_mapping)
            except Exception:
                pass
            
            # 计算需要预测的位置数：每个糖苷键连接点预测2个位置
            expected_labels = calculate_expected_labels(sequence_str)
            
            # 处理单个糖单元的边界情况（无连接点，无需预测）
            if expected_labels == 0:
                return [], [], []
            
            # 词汇表验证
            for unit in units:
                if unit not in self.vocab_mapping:
                    allowed = ', '.join(sorted(list(self.vocab_mapping.keys()))[:30])
                    raise ValueError(f"未知的糖单元: {unit}。请确保糖单元在训练词汇表中。示例: {allowed} ...")
            
            # 输入张量准备
            token_ids = [self.vocab_mapping[unit] for unit in units]
            sequence_tensor = torch.tensor([token_ids])  # 添加batch维度
            
            # 生成mask张量，标记需要预测的位置
            mask = torch.ones(1, expected_labels)
            
            # 集成预测执行
            fold_predictions = []
            
            with torch.no_grad():  # 禁用梯度计算以节省内存
                for model in self.models:
                    # 前向传播
                    outputs = model(sequence_tensor, mask)
                    
                    # 应用sigmoid激活函数得到概率
                    probs = torch.sigmoid(outputs)
                    
                    # 确保输出维度正确（移除batch维度）
                    if probs.dim() > 1:
                        probs = probs.squeeze(0)
                    
                    fold_predictions.append(probs.cpu().numpy())
            
            # 集成结果计算
            ensemble_probs = np.mean(fold_predictions, axis=0)
            
            # 根据阈值生成二分类预测结果
            predictions = (ensemble_probs > threshold).astype(int)
            
            return predictions.tolist(), ensemble_probs.tolist(), fold_predictions
            
        except Exception as e:
            if self.verbose:
                print(f"预测过程中出错: {e}")
                print(f"序列: {sequence_str}")
            raise e
    
    def predict_single(self, sequence_str, threshold=0.5):
        """
        预测单个糖链序列 - 增强版本
        
        参数:
        - sequence_str: 糖链序列字符串
        - threshold: 二分类决策阈值
        
        返回:
        - dict: 包含预测结果、统计信息和一致性分析的字典
        """
        try:
            # 验证序列格式
            is_valid, error_msg = validate_sequence(sequence_str)
            if not is_valid:
                raise ValueError(error_msg)
            
            # 调用原有的预测方法
            predictions, probabilities, fold_probs = self.predict_sequence(sequence_str, threshold)
            
            # 计算统计信息
            if len(predictions) > 0:
                cleavage_sites_count = sum(predictions)
                total_positions = len(predictions)
                cleavage_ratio = cleavage_sites_count / total_positions
                avg_probability = np.mean(probabilities)
                max_probability = np.max(probabilities)
                min_probability = np.min(probabilities)
                
                # 计算fold间一致性
                consistency_scores = []
                for i in range(len(predictions)):
                    fold_preds_at_i = [(fold_prob[i] > threshold) for fold_prob in fold_probs]
                    consistency = sum(fold_preds_at_i) / len(fold_preds_at_i)
                    if consistency < 0.5:
                        consistency = 1 - consistency
                    consistency_scores.append(consistency)
                
                avg_consistency = np.mean(consistency_scores)
                min_consistency = np.min(consistency_scores)
                
                # 找出一致性较低的位置
                low_consistency_positions = [
                    i for i, score in enumerate(consistency_scores) if score < 0.8
                ]
                
                statistics = {
                    'cleavage_sites_count': cleavage_sites_count,
                    'total_positions': total_positions,
                    'cleavage_ratio': cleavage_ratio,
                    'avg_probability': avg_probability,
                    'max_probability': max_probability,
                    'min_probability': min_probability
                }
                
                consistency = {
                    'avg_consistency': avg_consistency,
                    'min_consistency': min_consistency,
                    'low_consistency_positions': low_consistency_positions,
                    'consistency_scores': consistency_scores
                }
                
            else:
                statistics = {
                    'cleavage_sites_count': 0,
                    'total_positions': 0,
                    'cleavage_ratio': 0,
                    'avg_probability': 0,
                    'max_probability': 0,
                    'min_probability': 0
                }
                
                consistency = {
                    'avg_consistency': 1.0,
                    'min_consistency': 1.0,
                    'low_consistency_positions': [],
                    'consistency_scores': []
                }
            
            return {
                'sequence': sequence_str,
                'predictions': predictions,
                'probabilities': probabilities,
                'fold_probabilities': fold_probs,
                'threshold': threshold,
                'statistics': statistics,
                'consistency': consistency
            }
            
        except Exception as e:
            raise e
    
    def predict_batch(self, sequences: List[str], threshold: float = 0.5, show_progress: bool = True) -> List[Dict]:
        """
        批量预测多个糖链序列
        
        这个函数处理多个序列的批量预测，提供进度显示和错误处理。
        
        参数:
            sequences: 糖链序列字符串列表
            threshold: 二分类决策阈值
            show_progress: 是否显示进度信息
        
        返回:
            results: 预测结果字典列表，每个字典包含完整的预测信息
        """
        results = []
        
        if show_progress and self.verbose:
            print(f"开始批量预测 {len(sequences)} 个序列...")
        
        for i, seq in enumerate(sequences):
            try:
                # 调用单序列预测函数
                result = self.predict_single(seq, threshold)
                results.append(result)
                
                # 进度显示（每10个序列显示一次）
                if show_progress and self.verbose and (i + 1) % 10 == 0:
                    print(f"已完成 {i + 1}/{len(sequences)} 个序列的预测")
                    
            except Exception as e:
                # 处理单个序列预测失败的情况
                if self.verbose:
                    print(f"序列 {i+1} 预测失败: {e}")
                results.append({
                    'sequence': seq,
                    'error': str(e),
                    'status': 'failed'
                })
        
        if show_progress and self.verbose:
            success_count = sum(1 for r in results if 'error' not in r)
            print(f"✅ 批量预测完成！成功: {success_count}/{len(sequences)}")
        
        return results
    
    def predict_from_file(self, input_file_path, output_file_path=None, threshold=0.5, output_format='excel'):
        """
        从文件预测 - 增强版本
        
        参数:
        - input_file_path: 输入文件路径
        - output_file_path: 输出文件路径
        - threshold: 预测阈值
        - output_format: 输出格式 ('excel' 或 'csv')
        
        返回:
        - (results, actual_output_path): 预测结果列表和实际输出路径
        """
        try:
            # 读取输入文件
            if self.verbose:
                print(f"📂 读取输入文件: {input_file_path}")
            
            if input_file_path.endswith('.xlsx') or input_file_path.endswith('.xls'):
                df_input = pd.read_excel(input_file_path, header=None)
            elif input_file_path.endswith('.csv'):
                df_input = pd.read_csv(input_file_path, header=None)
            else:
                raise ValueError("不支持的文件格式。请使用Excel (.xlsx, .xls) 或CSV (.csv) 文件。")
            
            # 提取序列列表
            sequences = df_input.iloc[:, 0].tolist()
            
            # 过滤有效序列
            original_count = len(sequences)
            sequences = [seq for seq in sequences if pd.notna(seq) and isinstance(seq, str) and seq.strip()]
            filtered_count = len(sequences)
            
            if self.verbose:
                print(f"📊 文件读取完成: 原始行数 {original_count}, 有效序列数 {filtered_count}")
            
            if filtered_count == 0:
                raise ValueError("未找到有效的序列数据")
            
            # 执行批量预测
            if self.verbose:
                print(f"🔬 开始批量预测...")
            
            prediction_results = self.predict_batch(sequences, threshold)
            
            # 整理结果
            if self.verbose:
                print(f"📋 整理预测结果...")
            
            output_data = []
            successful_predictions = 0
            failed_predictions = 0
            
            for result in prediction_results:
                if 'error' in result:
                    failed_predictions += 1
                    output_data.append({
                        '序列': result['sequence'],
                        '预测结果': 'ERROR',
                        '预测概率': 'ERROR',
                        '切割位点数量': 'ERROR',
                        '平均概率': 'ERROR',
                        '最高概率': 'ERROR',
                        '错误信息': result['error']
                    })
                else:
                    successful_predictions += 1
                    
                    # 格式化预测结果
                    preds_str = ','.join(map(str, result['predictions']))
                    probs_str = ','.join([f"{p:.4f}" for p in result['probabilities']])
                    
                    # 计算统计信息
                    cleavage_count = sum(result['predictions'])
                    avg_prob = np.mean(result['probabilities'])
                    max_prob = np.max(result['probabilities'])
                    
                    output_data.append({
                        '序列': result['sequence'],
                        '预测结果': preds_str,
                        '预测概率': probs_str,
                        '切割位点数量': cleavage_count,
                        '平均概率': f"{avg_prob:.4f}",
                        '最高概率': f"{max_prob:.4f}",
                        '错误信息': ''
                    })
            
            # 保存结果
            df_output = pd.DataFrame(output_data)
            
            # 生成输出文件路径
            if output_file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if output_format == 'excel':
                    output_file_path = f'ensemble_predictions_{timestamp}.xlsx'
                else:
                    output_file_path = f'ensemble_predictions_{timestamp}.csv'
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # 保存文件
            if output_format == 'excel':
                df_output.to_excel(output_file_path, index=False)
            else:
                df_output.to_csv(output_file_path, index=False)
            
            if self.verbose:
                print(f"📊 批量预测完成: 成功 {successful_predictions}, 失败 {failed_predictions}")
                print(f"✅ 预测结果已保存到: {output_file_path}")
            
            return prediction_results, output_file_path
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 批量预测失败: {e}")
            return None, None
    
    def get_model_info(self):
        """获取模型信息"""
        try:
            avg_metrics = self.ensemble_info.get('avg_metrics', {})
            
            # 提取支持的糖单元
            supported_units = list(self.vocab_mapping.keys())
            
            # 获取训练时间戳
            training_timestamp = "未知"
            if self.fold_model_paths:
                # 从第一个模型文件名中提取时间戳
                first_model = os.path.basename(self.fold_model_paths[0])
                import re
                timestamp_match = re.search(r'(\d{8}_\d{6})', first_model)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    try:
                        from datetime import datetime
                        dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        training_timestamp = dt.strftime('%Y年%m月%d日 %H:%M:%S')
                    except:
                        training_timestamp = timestamp_str
            
            return {
                'model_count': len(self.models),
                'vocab_size': len(self.vocab_mapping),
                'supported_units': supported_units,
                'training_timestamp': training_timestamp,
                'selection_metric': 'AUC-ROC',
                'performance': {
                    'auc_roc': f"{avg_metrics.get('auc_roc', 0):.4f} ± {self.ensemble_info.get('std_metrics', {}).get('auc_roc_std', 0):.4f}",
                    'f1': f"{avg_metrics.get('f1', 0):.4f} ± {self.ensemble_info.get('std_metrics', {}).get('f1_std', 0):.4f}",
                    'accuracy': f"{avg_metrics.get('accuracy', 0):.4f} ± {self.ensemble_info.get('std_metrics', {}).get('accuracy_std', 0):.4f}"
                }
            }
        except Exception as e:
            return {
                'error': f"获取模型信息失败: {str(e)}"
            }
    
    def validate_sequence_format(self, sequence: str) -> Tuple[bool, str]:
        """
        验证序列格式（包装utils中的函数）
        
        参数:
            sequence: 糖链序列字符串
            
        返回:
            (is_valid, error_message): 验证结果和错误信息
        """
        return validate_sequence(sequence)


# 便捷函数
def create_predictor(ensemble_info_path: Optional[str] = None, verbose: bool = True) -> EnsemblePredictor:
    """
    创建集成预测器实例的便捷函数
    
    参数:
        ensemble_info_path: 集成模型信息文件路径
        verbose: 是否显示详细信息
        
    返回:
        EnsemblePredictor实例
    """
    return EnsemblePredictor(ensemble_info_path, verbose)


def predict_sequence(sequence_str, threshold=0.5):
    """
    便捷函数：预测单个序列
    
    参数:
    - sequence_str: 糖链序列字符串
    - threshold: 预测阈值
    
    返回:
    - dict: 预测结果字典
    """
    global _global_predictor
    
    if _global_predictor is None:
        _global_predictor = EnsemblePredictor(verbose=False)
    
    return _global_predictor.predict_single(sequence_str, threshold)


def predict_sequences(sequences: List[str], threshold: float = 0.5, predictor: Optional[EnsemblePredictor] = None) -> List[Dict]:
    """
    批量预测序列的便捷函数
    
    参数:
        sequences: 糖链序列字符串列表
        threshold: 预测阈值
        predictor: 预测器实例，如果为None则创建新实例
        
    返回:
        预测结果字典列表
    """
    if predictor is None:
        predictor = create_predictor(verbose=False)
    
    return predictor.predict_batch(sequences, threshold, show_progress=False)


# 全局预测器实例
_global_predictor = None 