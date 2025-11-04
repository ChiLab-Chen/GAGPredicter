"""
Ensemble Model Loader Module

Responsible for loading and managing ensemble models
"""

import torch
import pickle
import os
import glob
from typing import List, Dict, Optional, Tuple
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.model import DynamicCleavageRNN


def find_latest_ensemble_model(checkpoints_dir: Optional[str] = None) -> str:
    """
    Find the latest ensemble model info file
    
    Parameters:
        checkpoints_dir: Checkpoint directory path, if None uses default path
        
    Returns:
        Full path to the latest ensemble model info file
        
    Exceptions:
        FileNotFoundError: If no ensemble model files are found
    """
    if checkpoints_dir is None:
        checkpoints_dir = os.path.join(project_root, 'models', 'checkpoints')
    
    ensemble_files = glob.glob(os.path.join(checkpoints_dir, 'ensemble_info_*.pkl'))
    
    if not ensemble_files:
        raise FileNotFoundError(
            f"Ensemble model info file not found, please check directory: {checkpoints_dir}\n"
            "Make sure you have run 02_retrain.ipynb to generate the ensemble model"
        )
    
    # Sort by filename and take the latest (largest timestamp)
    ensemble_files.sort()
    latest_file = ensemble_files[-1]
    
    return latest_file


def load_ensemble_info(ensemble_info_path: Optional[str] = None) -> Dict:
    """
    Load ensemble model info
    
    Parameters:
        ensemble_info_path: Ensemble model info file path, if None automatically finds the latest file
        
    Returns:
        Ensemble model info dictionary
    """
    if ensemble_info_path is None:
        ensemble_info_path = find_latest_ensemble_model()
    
    print(f"Loading ensemble model info: {ensemble_info_path}")
    
    with open(ensemble_info_path, 'rb') as f:
        ensemble_info = pickle.load(f)
    
    return ensemble_info


def load_fold_models(ensemble_info: Dict) -> List[DynamicCleavageRNN]:
    """
    Load all fold models
    
    Parameters:
        ensemble_info: Ensemble model info dictionary
        
    Returns:
        List of loaded models
    """
    models = []
    model_params = ensemble_info['model_params']
    fold_model_paths = ensemble_info['model_paths']
    
    print(f"Loading {len(fold_model_paths)} fold models...")
    
    for i, model_path in enumerate(fold_model_paths):
        print(f"  Loading Fold {i+1}: {os.path.basename(model_path)}")
        
        # Create model instance
        model = DynamicCleavageRNN(**model_params)
        
        # Load pretrained weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()  # Set to evaluation mode
        
        models.append(model)
    
    return models


def load_ensemble_model(ensemble_info_path: Optional[str] = None) -> Tuple[List[DynamicCleavageRNN], Dict]:
    """
    Load complete ensemble model in one go
    
    Parameters:
        ensemble_info_path: Ensemble model info file path
        
    Returns:
        (models, ensemble_info): Model list and ensemble info
    """
    # Load ensemble model info
    ensemble_info = load_ensemble_info(ensemble_info_path)
    
    # Load all fold models
    models = load_fold_models(ensemble_info)
    
    print(f"✅ Ensemble model loading completed! Contains {len(models)} fold models")
    
    # Display ensemble model performance statistics
    avg_metrics = ensemble_info['avg_metrics']
    std_metrics = ensemble_info['std_metrics']
    
    print(f"\nEnsemble model average performance:")
    print(f"AUC-ROC: {avg_metrics['auc_roc']:.4f} ± {std_metrics['auc_roc_std']:.4f}")
    print(f"F1: {avg_metrics['f1']:.4f} ± {std_metrics['f1_std']:.4f}")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy_std']:.4f}")
    
    return models, ensemble_info


def validate_model_files(ensemble_info: Dict) -> bool:
    """
    Validate that model files exist
    
    Parameters:
        ensemble_info: Ensemble model info dictionary
        
    Returns:
        Whether all model files exist
    """
    fold_model_paths = ensemble_info['model_paths']
    
    missing_files = []
    for model_path in fold_model_paths:
        if not os.path.exists(model_path):
            missing_files.append(model_path)
    
    if missing_files:
        print(f"❌ The following model files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True


def get_model_info(ensemble_info_path: Optional[str] = None) -> Dict:
    """
    Get model info summary
    
    Parameters:
        ensemble_info_path: Ensemble model info file path
        
    Returns:
        Model info summary dictionary
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
