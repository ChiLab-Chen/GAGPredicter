# GAGPredict Pro - Heparan Sulfate Cleavage Site Prediction Tool

GAGPredict Pro is a deep learning-based graphical user interface tool designed to predict enzyme cleavage sites in Heparan Sulfate (HS) glycan sequences. The tool utilizes a Recurrent Neural Network (RNN) model to accurately predict the breaking positions of glycan molecules during enzymatic digestion.

## 🧬 Project Overview

This project is based on an RNN deep learning model specifically developed to predict cleavage sites in Heparan Sulfate (HS) glycan sequences. HS is an important glycosaminoglycan that plays crucial roles in various biological processes including cell signaling, angiogenesis, and inflammatory responses. Accurately predicting HS cleavage sites is significant for understanding structure-function relationships.

## 🚀 Key Features

- **Graphical User Interface**: Intuitive GUI that requires no programming experience
- **Single Sequence Prediction**: Supports cleavage site prediction for individual glycan sequences
- **Batch Prediction**: Supports batch import of multiple sequences from Excel files for prediction
- **Result Visualization**: Displays prediction results and statistical information in an intuitive manner
- **Ensemble Model**: Uses 5-fold cross-validation ensemble model to improve prediction accuracy
- **Result Export**: Prediction results can be exported to Excel files for further analysis

## 📁 Project Structure

```
GAGPredicter/
├── gui_app.py              # Main GUI application
├── Squence.xlsx            # Sample input sequence file
├── prediction_results.xlsx # Prediction results output file
├── config/                 # Model configuration files
│   ├── best_params.yaml    # Best model parameter configuration
│   └── fold_splits.pkl     # Cross-validation split information
├── models/                 # Model files
│   ├── model.py            # RNN model definition
│   └── checkpoints/        # Pre-trained model weights
├── predictor/              # Predictor module
│   ├── predictor.py        # Core predictor class
│   ├── utils.py            # Utility functions
│   └── model_loader.py     # Model loader
└── README.md               # Project documentation
```

## 🛠️ Dependencies

- Python 3.7+
- PyTorch >= 1.9.0
- CustomTkinter >= 5.0.0
- Pandas >= 1.3.0
- NumPy >= 1.21.0
- OpenPyXL >= 3.0.9
- PyYAML >= 6.0

## ▶️ Usage

### Launch the Application
```bash
python gui_app.py
```

### Single Sequence Prediction
1. Select "Single Sequence Prediction" mode in the main interface
2. Enter the glycan sequence in the input box, e.g.: `dHexD2-O-HexNS-O-HexD2`
3. Click the "Predict" button
4. View the prediction results and statistical information

### Batch Prediction
1. Select "Batch Prediction" mode in the main interface
2. Prepare an Excel file containing glycan sequences (one sequence per row)
3. Click the "Select File" button to choose the input file
4. Click the "Start Prediction" button
5. Results will be automatically saved to `prediction_results.xlsx` upon completion

## 🧪 Input Format

### Single Sequence Format
Glycan sequences should be written in the following format:
```
SugarUnit1-O-SugarUnit2-O-SugarUnit3-...-O-SugarUnitN
```

Examples:
```
dHexD2-O-HexNS-O-HexD2
HexA-O-GlcNS6S-O-HexA-O-GlcNS6S
```

### Batch File Format
Excel files should contain a single column of glycan sequences with sequence data in the first column and no header row.

## 📊 Output Results

Prediction results include the following information:
- **Prediction Results**: 0/1 sequence, where 1 indicates a predicted cleavage site
- **Prediction Probability**: Cleavage probability for each position
- **Number of Cleavage Sites**: Total number of predicted cleavage sites
- **Average Probability**: Average cleavage probability across all positions
- **Maximum Probability**: Probability of the most likely cleavage site

## 🧠 Model Information

- **Model Type**: Recurrent Neural Network (RNN)
- **Ensemble Method**: 5-fold cross-validation ensemble
- **Best Performance**:
  - AUC-ROC: 0.9469 ± 0.0188
  - F1 Score: 0.8485 ± 0.0308
  - Accuracy: 0.9344 ± 0.0176

## 📞 Support

For issues, please contact the project maintainer or submit a GitHub issue.

## 📄 License

This project is intended for academic research purposes only.
