"""
Predictor Utility Functions Module

Provides auxiliary functions for sequence validation, result formatting, etc.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional


def validate_sequence(sequence_str):
    try:
        if not sequence_str or not isinstance(sequence_str, str):
            return False, "Sequence cannot be empty and must be a string"
        
        sequence_str = sequence_str.strip()
        
        if not sequence_str:
            return False, "Sequence cannot be empty"
        
        if sequence_str.count('-O-') == 0:
            if len(sequence_str) < 3 or len(sequence_str) > 20:
                return False, "Single sugar unit name length should be between 3-20 characters"
            
            cleaned_unit = sequence_str.replace('6S', '').replace('NS', '').replace('D2', '').replace('Ac', '').replace('ol', '').replace('2S', '').replace('4S', '').replace('3S', '')
            if not all(c.isalnum() or c == '_' for c in cleaned_unit):
                return False, f"Sugar unit '{sequence_str}' contains illegal characters"
        else:
            units = sequence_str.split('-O-')
            
            if len(units) < 2:
                return False, "Polysaccharide sequence must contain at least 2 sugar units"
            
            for i, unit in enumerate(units):
                if not unit or not unit.strip():
                    return False, f"Sugar unit {i+1} cannot be empty"
                
                unit = unit.strip()
                if len(unit) < 3 or len(unit) > 20:
                    return False, f"Sugar unit {i+1} '{unit}' name length should be between 3-20 characters"
                
                cleaned_unit = unit.replace('6S', '').replace('NS', '').replace('D2', '').replace('Ac', '').replace('ol', '').replace('2S', '').replace('4S', '').replace('3S', '')
                if not all(c.isalnum() or c == '_' for c in cleaned_unit):
                    return False, f"Sugar unit {i+1} '{unit}' contains illegal characters"
        
        return True, "Sequence format is correct"
        
    except Exception as e:
        return False, f"Sequence validation error: {str(e)}"


def format_prediction_result(
    sequence: str,
    predictions: List[int],
    probabilities: List[float],
    fold_probabilities: Optional[List[List[float]]] = None,
    threshold: float = 0.5
) -> Dict:
    result = {
        "sequence": sequence,
        "predictions": predictions,
        "probabilities": [round(p, 4) for p in probabilities],
        "threshold": threshold,
        "statistics": {}
    }
    
    if predictions:
        result["statistics"] = {
            "cleavage_sites_count": sum(predictions),
            "total_positions": len(predictions),
            "cleavage_ratio": round(sum(predictions) / len(predictions), 4),
            "avg_probability": round(np.mean(probabilities), 4),
            "max_probability": round(np.max(probabilities), 4),
            "min_probability": round(np.min(probabilities), 4),
            "std_probability": round(np.std(probabilities), 4)
        }
        
        if fold_probabilities:
            consistency_scores = []
            for i in range(len(predictions)):
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
    if "-O-" not in sequence:
        return [sequence]
    return sequence.split("-O-")


def calculate_expected_labels(sequence: str) -> int:
    units = parse_sequence_units(sequence)
    return 2 * (len(units) - 1) if len(units) > 1 else 0


def save_predictions_to_file(
    predictions: List[Dict],
    output_path: str,
    format: str = 'excel'
) -> bool:
    try:
        output_data = []
        for pred in predictions:
            if 'error' in pred:
                output_data.append({
                    'Sequence': pred['sequence'],
                    'Prediction Results': 'ERROR',
                    'Prediction Probabilities': 'ERROR',
                    'Cleavage Site Count': 'ERROR',
                    'Average Probability': 'ERROR',
                    'Maximum Probability': 'ERROR',
                    'Error Message': pred['error']
                })
            else:
                stats = pred.get('statistics', {})
                output_data.append({
                    'Sequence': pred['sequence'],
                    'Prediction Results': ','.join(map(str, pred['predictions'])),
                    'Prediction Probabilities': ','.join([f"{p:.4f}" for p in pred['probabilities']]),
                    'Cleavage Site Count': stats.get('cleavage_sites_count', 0),
                    'Average Probability': stats.get('avg_probability', 0),
                    'Maximum Probability': stats.get('max_probability', 0),
                    'Error Message': ''
                })
        
        df = pd.DataFrame(output_data)
        
        if format.lower() == 'excel':
            df.to_excel(output_path, index=False)
        elif format.lower() == 'csv':
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        else:
            raise ValueError(f"Unsupported file format: {format}")
        
        return True
        
    except Exception as e:
        print(f"Failed to save file: {e}")
        return False


def load_sequences_from_file(file_path: str) -> List[str]:
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, header=None)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=None)
        else:
            raise ValueError("Unsupported file format. Please use Excel (.xlsx, .xls) or CSV (.csv) files.")
        
        sequences = df.iloc[:, 0].tolist()
        
        sequences = [seq for seq in sequences if pd.notna(seq) and isinstance(seq, str) and seq.strip()]
        
        return sequences
        
    except Exception as e:
        print(f"Failed to read file: {e}")
        return []
