"""
预测器工具函数模块

提供序列验证、结果格式化等辅助功能
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional


def validate_sequence(sequence_str):
    """
    验证糖链序列格式
    
    参数:
    - sequence_str: 糖链序列字符串
    
    返回:
    - (bool, str): (是否有效, 错误消息)
    """
    try:
        if not sequence_str or not isinstance(sequence_str, str):
            return False, "序列不能为空且必须是字符串"
        
        sequence_str = sequence_str.strip()
        
        if not sequence_str:
            return False, "序列不能为空"
        
        # 检查基本格式
        if sequence_str.count('-O-') == 0:
            # 单个糖单元的情况，检查是否为合理的糖单元名称
            if len(sequence_str) < 3 or len(sequence_str) > 20:
                return False, "单个糖单元名称长度应在3-20个字符之间"
            
            # 检查单个糖单元的字符
            cleaned_unit = sequence_str.replace('6S', '').replace('NS', '').replace('D2', '').replace('Ac', '').replace('ol', '').replace('2S', '').replace('4S', '').replace('3S', '')
            if not all(c.isalnum() or c == '_' for c in cleaned_unit):
                return False, f"糖单元'{sequence_str}'包含非法字符"
        else:
            # 多个糖单元的情况
            units = sequence_str.split('-O-')
            
            if len(units) < 2:
                return False, "多糖序列至少需要包含2个糖单元"
            
            for i, unit in enumerate(units):
                if not unit or not unit.strip():
                    return False, f"第{i+1}个糖单元不能为空"
                
                unit = unit.strip()
                if len(unit) < 3 or len(unit) > 20:
                    return False, f"第{i+1}个糖单元'{unit}'名称长度应在3-20个字符之间"
                
                # 检查是否包含非法字符
                # 允许字母、数字和下划线，移除常见的糖单元修饰符后检查
                cleaned_unit = unit.replace('6S', '').replace('NS', '').replace('D2', '').replace('Ac', '').replace('ol', '').replace('2S', '').replace('4S', '').replace('3S', '')
                if not all(c.isalnum() or c == '_' for c in cleaned_unit):
                    return False, f"第{i+1}个糖单元'{unit}'包含非法字符"
        
        return True, "序列格式正确"
        
    except Exception as e:
        return False, f"序列验证出错: {str(e)}"


def format_prediction_result(
    sequence: str,
    predictions: List[int],
    probabilities: List[float],
    fold_probabilities: Optional[List[List[float]]] = None,
    threshold: float = 0.5
) -> Dict:
    """
    格式化预测结果
    
    参数:
        sequence: 原始序列
        predictions: 预测结果列表
        probabilities: 预测概率列表
        fold_probabilities: 各fold的详细概率
        threshold: 预测阈值
        
    返回:
        格式化的预测结果字典
    """
    result = {
        "sequence": sequence,
        "predictions": predictions,
        "probabilities": [round(p, 4) for p in probabilities],
        "threshold": threshold,
        "statistics": {}
    }
    
    if predictions:
        # 基本统计信息
        result["statistics"] = {
            "cleavage_sites_count": sum(predictions),
            "total_positions": len(predictions),
            "cleavage_ratio": round(sum(predictions) / len(predictions), 4),
            "avg_probability": round(np.mean(probabilities), 4),
            "max_probability": round(np.max(probabilities), 4),
            "min_probability": round(np.min(probabilities), 4),
            "std_probability": round(np.std(probabilities), 4)
        }
        
        # fold间一致性分析
        if fold_probabilities:
            consistency_scores = []
            for i in range(len(predictions)):
                # 计算每个位置上fold间的一致性
                fold_preds_at_i = [(fold_prob[i] > threshold) for fold_prob in fold_probabilities]
                consistency = sum(fold_preds_at_i) / len(fold_preds_at_i)
                if consistency < 0.5:
                    consistency = 1 - consistency
                consistency_scores.append(consistency)
            
            result["consistency"] = {
                "avg_consistency": round(np.mean(consistency_scores), 4),
                "min_consistency": round(np.min(consistency_scores), 4),
                "low_consistency_positions": [
                    i for i, score in enumerate(consistency_scores) if score < 0.8
                ]
            }
    
    return result


def parse_sequence_units(sequence: str) -> List[str]:
    """
    解析糖链序列为糖单元列表
    
    参数:
        sequence: 糖链序列字符串
        
    返回:
        糖单元列表
    """
    if "-O-" not in sequence:
        return [sequence]
    return sequence.split("-O-")


def calculate_expected_labels(sequence: str) -> int:
    """
    计算序列的预期标签数量
    
    参数:
        sequence: 糖链序列字符串
        
    返回:
        预期的标签数量（每个连接点2个位置）
    """
    units = parse_sequence_units(sequence)
    return 2 * (len(units) - 1) if len(units) > 1 else 0


def save_predictions_to_file(
    predictions: List[Dict],
    output_path: str,
    format: str = 'excel'
) -> bool:
    """
    保存预测结果到文件
    
    参数:
        predictions: 预测结果列表
        output_path: 输出文件路径
        format: 文件格式 ('excel' 或 'csv')
        
    返回:
        保存是否成功
    """
    try:
        # 准备数据
        output_data = []
        for pred in predictions:
            if 'error' in pred:
                output_data.append({
                    '序列': pred['sequence'],
                    '预测结果': 'ERROR',
                    '预测概率': 'ERROR',
                    '切割位点数量': 'ERROR',
                    '平均概率': 'ERROR',
                    '最高概率': 'ERROR',
                    '错误信息': pred['error']
                })
            else:
                stats = pred.get('statistics', {})
                output_data.append({
                    '序列': pred['sequence'],
                    '预测结果': ','.join(map(str, pred['predictions'])),
                    '预测概率': ','.join([f"{p:.4f}" for p in pred['probabilities']]),
                    '切割位点数量': stats.get('cleavage_sites_count', 0),
                    '平均概率': stats.get('avg_probability', 0),
                    '最高概率': stats.get('max_probability', 0),
                    '错误信息': ''
                })
        
        # 创建DataFrame
        df = pd.DataFrame(output_data)
        
        # 保存文件
        if format.lower() == 'excel':
            df.to_excel(output_path, index=False)
        elif format.lower() == 'csv':
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        else:
            raise ValueError(f"不支持的文件格式: {format}")
        
        return True
        
    except Exception as e:
        print(f"保存文件失败: {e}")
        return False


def load_sequences_from_file(file_path: str) -> List[str]:
    """
    从文件加载序列列表
    
    参数:
        file_path: 输入文件路径
        
    返回:
        序列字符串列表
    """
    try:
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, header=None)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=None)
        else:
            raise ValueError("不支持的文件格式。请使用Excel (.xlsx, .xls) 或CSV (.csv) 文件。")
        
        # 提取序列列表（假设第一列包含序列）
        sequences = df.iloc[:, 0].tolist()
        
        # 过滤掉空值和非字符串值
        sequences = [seq for seq in sequences if pd.notna(seq) and isinstance(seq, str) and seq.strip()]
        
        return sequences
        
    except Exception as e:
        print(f"读取文件失败: {e}")
        return [] 