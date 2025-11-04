"""
Integrated Predictor Core Module

Provides glycan sequence cleavage site prediction functionality based on GRU ensemble model
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


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


try:
    from models.model import DynamicCleavageRNN
except ImportError:
    
    sys.path.append(os.path.join(project_root, 'models'))
    from model import DynamicCleavageRNN


from .utils import (
    validate_sequence,
    parse_sequence_units,
    calculate_expected_labels
)




class EnsemblePredictor:
    
    
    def __init__(self, ensemble_info_path=None, verbose=True):
        
        self.verbose = verbose
        
        if ensemble_info_path is None:
            ensemble_info_path = self._find_latest_ensemble_model()
        
        if self.verbose:
            print(f"Loading ensemble model info: {ensemble_info_path}")
        
        
        with open(ensemble_info_path, 'rb') as f:
            self.ensemble_info = pickle.load(f)
        
        
        self.model_params = self.ensemble_info['model_params']
        self.vocab_mapping = self.ensemble_info['vocab_mapping']
        self.fold_model_paths = self.ensemble_info['model_paths']

        
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
                    raise FileNotFoundError(f"Model file not found: {p} (also tried: {candidate})")
        self.fold_model_paths = corrected_paths
        
        
        self.models = []
        if self.verbose:
            print(f"Loading {len(self.fold_model_paths)} fold models...")
        
        for i, model_path in enumerate(self.fold_model_paths):
            if self.verbose:
                print(f"  Loading Fold {i+1}: {os.path.basename(model_path)}")
            
            
            model = DynamicCleavageRNN(**self.model_params)
            
            
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()  
            
            self.models.append(model)
        
        if self.verbose:
            print(f"✅ Ensemble model loading completed! Contains {len(self.models)} fold models")
            
            
            avg_metrics = self.ensemble_info['avg_metrics']
            std_metrics = self.ensemble_info['std_metrics']
            
            print(f"\nEnsemble model average performance:")
            print(f"AUC-ROC: {avg_metrics['auc_roc']:.4f} ± {std_metrics['auc_roc_std']:.4f}")
            print(f"F1: {avg_metrics['f1']:.4f} ± {std_metrics['f1_std']:.4f}")
            print(f"Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy_std']:.4f}")
    
    def _find_latest_ensemble_model(self):
        
        try:
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..'))
        except NameError:
            
            project_root = os.path.abspath('..')
        
        checkpoints_dir = os.path.join(project_root, 'models', 'checkpoints')
        ensemble_files = glob.glob(os.path.join(checkpoints_dir, 'ensemble_info_*.pkl'))
        
        if not ensemble_files:
            raise FileNotFoundError(
                "Ensemble model info file not found, please run 02_retrain.ipynb to generate ensemble model first"
            )
        
        
        ensemble_files.sort()
        latest_file = ensemble_files[-1]
        
        if self.verbose:
            print(f"Automatically selecting the latest ensemble model: {os.path.basename(latest_file)}")
        
        return latest_file
    
    def predict_sequence(self, sequence_str: str, threshold: float = 0.5) -> Tuple[List[int], List[float], List[List[float]]]:
        
        try:
            
            is_valid, error_msg = validate_sequence(sequence_str)
            if not is_valid:
                raise ValueError(f"Sequence format error: {error_msg}")
            
            
            units = parse_sequence_units(sequence_str)

            
            try:
                from .utils import normalize_units_to_vocab
                units = normalize_units_to_vocab(units, self.vocab_mapping)
            except Exception:
                pass
            
            
            expected_labels = calculate_expected_labels(sequence_str)
            
            
            if expected_labels == 0:
                return [], [], []
            
            
            for unit in units:
                if unit not in self.vocab_mapping:
                    allowed = ', '.join(sorted(list(self.vocab_mapping.keys()))[:30])
                    raise ValueError(f"Unknown sugar unit: {unit}. Please ensure the sugar unit is in the training vocabulary. Examples: {allowed} ...")
            
            
            token_ids = [self.vocab_mapping[unit] for unit in units]
            sequence_tensor = torch.tensor([token_ids])  
            
            
            mask = torch.ones(1, expected_labels)
            
            
            fold_predictions = []
            
            with torch.no_grad():  
                for model in self.models:
                    
                    outputs = model(sequence_tensor, mask)
                    
                    
                    probs = torch.sigmoid(outputs)
                    
                    
                    if probs.dim() > 1:
                        probs = probs.squeeze(0)
                    
                    fold_predictions.append(probs.cpu().numpy())
            
            
            ensemble_probs = np.mean(fold_predictions, axis=0)
            
            
            predictions = (ensemble_probs > threshold).astype(int)
            
            return predictions.tolist(), ensemble_probs.tolist(), fold_predictions
            
        except Exception as e:
            if self.verbose:
                print(f"Error during prediction: {e}")
                print(f"Sequence: {sequence_str}")
            raise e
    
    def predict_single(self, sequence_str, threshold=0.5):
        
        try:
            
            is_valid, error_msg = validate_sequence(sequence_str)
            if not is_valid:
                raise ValueError(error_msg)
            
            
            predictions, probabilities, fold_probs = self.predict_sequence(sequence_str, threshold)
            
            
            if len(predictions) > 0:
                cleavage_sites_count = sum(predictions)
                total_positions = len(predictions)
                cleavage_ratio = cleavage_sites_count / total_positions
                avg_probability = np.mean(probabilities)
                max_probability = np.max(probabilities)
                min_probability = np.min(probabilities)
                
                
                consistency_scores = []
                for i in range(len(predictions)):
                    fold_preds_at_i = [(fold_prob[i] > threshold) for fold_prob in fold_probs]
                    consistency = sum(fold_preds_at_i) / len(fold_preds_at_i)
                    if consistency < 0.5:
                        consistency = 1 - consistency
                    consistency_scores.append(consistency)
                
                avg_consistency = np.mean(consistency_scores)
                min_consistency = np.min(consistency_scores)
                
                
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
        
        results = []
        
        if show_progress and self.verbose:
            print(f"Starting batch prediction of {len(sequences)} sequences...")
        
        for i, seq in enumerate(sequences):
            try:
                
                result = self.predict_single(seq, threshold)
                results.append(result)
                
                
                if show_progress and self.verbose and (i + 1) % 10 == 0:
                    print(f"Completed prediction of {i + 1}/{len(sequences)} sequences")
                    
            except Exception as e:
                
                if self.verbose:
                    print(f"Prediction failed for sequence {i+1}: {e}")
                results.append({
                    'sequence': seq,
                    'error': str(e),
                    'status': 'failed'
                })
        
        if show_progress and self.verbose:
            success_count = sum(1 for r in results if 'error' not in r)
            print(f"✅ Batch prediction completed! Success: {success_count}/{len(sequences)}")
        
        return results
    
    def predict_from_file(self, input_file_path, output_file_path=None, threshold=0.5, output_format='excel'):
        
        try:
            
            if self.verbose:
                print(f"📂 Reading input file: {input_file_path}")
            
            if input_file_path.endswith('.xlsx') or input_file_path.endswith('.xls'):
                df_input = pd.read_excel(input_file_path, header=None)
            elif input_file_path.endswith('.csv'):
                df_input = pd.read_csv(input_file_path, header=None)
            else:
                raise ValueError("Unsupported file format. Please use Excel (.xlsx, .xls) or CSV (.csv) files.")
            
            
            sequences = df_input.iloc[:, 0].tolist()
            
            
            original_count = len(sequences)
            sequences = [seq for seq in sequences if pd.notna(seq) and isinstance(seq, str) and seq.strip()]
            filtered_count = len(sequences)
            
            if self.verbose:
                print(f"📊 File reading completed: Original rows {original_count}, Valid sequences {filtered_count}")
            
            if filtered_count == 0:
                raise ValueError("No valid sequence data found")
            
            
            if self.verbose:
                print(f"🔬 Starting batch prediction...")
            
            prediction_results = self.predict_batch(sequences, threshold)
            
            
            if self.verbose:
                print(f"📋 Organizing prediction results...")
            
            output_data = []
            successful_predictions = 0
            failed_predictions = 0
            
            for result in prediction_results:
                if 'error' in result:
                    failed_predictions += 1
                    output_data.append({
                        'Sequence': result['sequence'],
                        'Prediction Results': 'ERROR',
                        'Prediction Probabilities': 'ERROR',
                        'Cleavage Site Count': 'ERROR',
                        'Average Probability': 'ERROR',
                        'Maximum Probability': 'ERROR',
                        'Error Message': result['error']
                    })
                else:
                    successful_predictions += 1
                    
                    
                    preds_str = ','.join(map(str, result['predictions']))
                    probs_str = ','.join([f"{p:.4f}" for p in result['probabilities']])
                    
                    
                    cleavage_count = sum(result['predictions'])
                    avg_prob = np.mean(result['probabilities'])
                    max_prob = np.max(result['probabilities'])
                    
                    output_data.append({
                        'Sequence': result['sequence'],
                        'Prediction Results': preds_str,
                        'Prediction Probabilities': probs_str,
                        'Cleavage Site Count': cleavage_count,
                        'Average Probability': f"{avg_prob:.4f}",
                        'Maximum Probability': f"{max_prob:.4f}",
                        'Error Message': ''
                    })
            
            
            df_output = pd.DataFrame(output_data)
            
            
            if output_file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if output_format == 'excel':
                    output_file_path = f'ensemble_predictions_{timestamp}.xlsx'
                else:
                    output_file_path = f'ensemble_predictions_{timestamp}.csv'
            
            
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            
            if output_format == 'excel':
                df_output.to_excel(output_file_path, index=False)
            else:
                df_output.to_csv(output_file_path, index=False)
            
            if self.verbose:
                print(f"📊 Batch prediction completed: Success {successful_predictions}, Failed {failed_predictions}")
                print(f"✅ Prediction results saved to: {output_file_path}")
            
            return prediction_results, output_file_path
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Batch prediction failed: {e}")
            return None, None
    
    def get_model_info(self):
        
        try:
            avg_metrics = self.ensemble_info.get('avg_metrics', {})
            
            
            supported_units = list(self.vocab_mapping.keys())
            
            
            training_timestamp = "Unknown"
            if self.fold_model_paths:
                
                first_model = os.path.basename(self.fold_model_paths[0])
                import re
                timestamp_match = re.search(r'(\d{8}_\d{6})', first_model)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    try:
                        from datetime import datetime
                        dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        training_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
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
                'error': f"Failed to get model info: {str(e)}"
            }
    
    def validate_sequence_format(self, sequence: str) -> Tuple[bool, str]:
        
        return validate_sequence(sequence)


def create_predictor(ensemble_info_path: Optional[str] = None, verbose: bool = True) -> EnsemblePredictor:
    
    return EnsemblePredictor(ensemble_info_path, verbose)


def predict_sequence(sequence_str, threshold=0.5):
    
    global _global_predictor
    
    if _global_predictor is None:
        _global_predictor = EnsemblePredictor(verbose=False)
    
    return _global_predictor.predict_single(sequence_str, threshold)


def predict_sequences(sequences: List[str], threshold: float = 0.5, predictor: Optional[EnsemblePredictor] = None) -> List[Dict]:
    
    if predictor is None:
        predictor = create_predictor(verbose=False)
    
    return predictor.predict_batch(sequences, threshold, show_progress=False)


_global_predictor = None
