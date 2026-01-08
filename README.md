# Hysteresis-ML Project

This project implements a neural network-based machine learning pipeline for hysteresis parameter prediction. The workflow includes hyperparameter optimization, model training, and prediction capabilities.

---

## Project Structure

### Python Scripts

#### 1. `1.10-fold cross-validation.txt`
**Purpose**: Hyperparameter optimization using 10-fold cross-validation

**Functionality**:
- Performs grid search over hyperparameter space (neurons, learning rate, epochs)
- Uses 10-fold cross-validation to evaluate model performance
- Implements a 2-layer neural network with LeakyReLU activation
- Evaluates models using MAE, RMSE, R², and MAPE metrics

**Input**:
- `data.xlsx` - Training dataset with 11 input features and 2 output labels (y1, y2)

**Output**:
- Console output displaying best hyperparameters
- No files saved (evaluation only)

**Key Parameters**:
- Grid search ranges: neurons1 (10-19), neurons2 (8-14), learning_rate (0.01-0.05), epochs (200-400)
- Train/test split: 80/20 with random seed 37
- Cross-validation: 10 folds with random seed 71

---

#### 2. `2.backbone parameter training.txt`
**Purpose**: Train backbone model with normalized outputs

**Functionality**:
- Trains a neural network with both X and y normalization
- Uses fixed hyperparameters determined from cross-validation
- Saves trained model and scalers for future use

**Input**:
- `data.xlsx` - Training dataset with 11 input features and 2 output labels

**Output**:
- `saved_models/fixed_model_{timestamp}/` folder containing:
  - `model.h5` - Trained neural network model
  - `scalerX.save` - MinMaxScaler for input features
  - `scalerY.save` - MinMaxScaler for output labels
  - `info.txt` - Training configuration details

**Key Parameters**:
- neurons1=14, neurons2=11, learning_rate=0.01, epochs=400, batch_size=30
- Random seed: 37
- Both inputs and outputs are normalized using MinMaxScaler

---

#### 3. `3.Hysteresis parameter training.txt`
**Purpose**: Train hysteresis model with constrained outputs

**Functionality**:
- Trains a neural network with output constraints (range [0, 3])
- Uses sigmoid activation + Lambda layer to constrain outputs
- Only normalizes input features (X), not outputs (y)
- Implements EarlyStopping for training optimization

**Input**:
- `data.xlsx` - Training dataset with 11 input features and 2 output labels

**Output**:
- `final_model_with_earlystop/` folder containing:
  - `model.h5` - Trained neural network model
  - `scaler_X.save` - MinMaxScaler for input features only
  - `result.xlsx` - Training and testing performance metrics

**Key Parameters**:
- neurons1=19, neurons2=12, learning_rate=0.02, epochs=200, batch_size=32
- Random seed: 37
- Output constraint: [0, 3] using sigmoid * 3
- EarlyStopping: patience=20 on validation loss

---

#### 4. `4.Prediction model.txt`
**Purpose**: Batch prediction using multiple trained models

**Functionality**:
- Loads multiple pre-trained models from saved directories
- Handles models with and without y-scalers
- Performs predictions on new input data
- Consolidates all predictions into a single output file

**Input**:
- `data2.xlsx` - New dataset with 11 input features (no labels needed)
- Pre-trained models in `saved_models/` subdirectories

**Output**:
- `all_predictions.xlsx` - Contains original features plus predictions from all models
  - Columns: Original 11 features + pred_model{i}_out{j} for each model and output

**Model Configuration**:
- Model 1: `saved_models/1/` (with y scaler)
- Model 2: `saved_models/2/` (with y scaler)
- Model 3: `saved_models/3/` (without y scaler)
- Model 4: `saved_models/4/` (without y scaler)

---

### Data Files

#### `data_all.xlsx`
Main dataset file containing training data with 11 input features and 2 target outputs.

**Format**:
- No header row
- Columns 1-11: Input features
- Columns 12-13: Target outputs (y1, y2)

---

### Saved Models Directory

#### `saved_models/`
Contains subdirectories with trained models and associated files.

**Structure**:
Each subdirectory (1, 2, 3, 4) may contain:
- `model.h5` - Keras neural network model
- `scaler_X.save` - MinMaxScaler for input normalization
- `scaler_y.save` - MinMaxScaler for output normalization (optional)
- `info.txt` - Model training configuration and parameters

**Model Types**:
- **Models 1 & 2**: Use output normalization (include scaler_y.save)
- **Models 3 & 4**: No output normalization (constrained outputs [0, 3])

---

## Workflow

### 1. Hyperparameter Optimization
Run `1.10-fold cross-validation.txt` to find optimal hyperparameters:
```
Input: data.xlsx
Output: Best hyperparameters printed to console
```

### 2. Model Training
Train models in sequence for different parameter types:

**Step 2.1 - Backbone Parameter Training**:
```
Run: 2.backbone parameter training.txt
Input: data.xlsx
Output: saved_models/fixed_model_{timestamp}/
Purpose: Train backbone parameters with normalized outputs
```

**Step 2.2 - Hysteresis Parameter Training**:
```
Run: 3.Hysteresis parameter training.txt
Input: data.xlsx
Output: final_model_with_earlystop/
Purpose: Train hysteresis parameters with constrained outputs [0, 3]
```

### 3. Prediction
Run `4.Prediction model.txt` to generate predictions:
```
Input: data2.xlsx + saved_models/1,2,3,4/
Output: all_predictions.xlsx
```

---

## Requirements and Verified Environment

This project has been tested and verified in the following software environment. Only libraries explicitly used in the source code are listed below.

### Python Environment
- **Python**: 3.11.14

### Operating System
- Windows 10 / Windows 11

⚠️ **Note**: TensorFlow 2.13.x is the last stable version compatible with Python 3.11.

### Required Python Packages

The following libraries are directly used in the project scripts:

```
tensorflow==2.13.0
keras==2.13.1
numpy==1.24.3
pandas==2.3.3
scikit-learn==1.4.0
scipy==1.15.3
joblib==1.5.3
openpyxl==3.1.5
h5py==3.15.1
```

### Installation

To install all required packages, run:
```bash
pip install tensorflow==2.13.0 keras==2.13.1 numpy==1.24.3 pandas==2.3.3 scikit-learn==1.4.0 scipy==1.15.3 joblib==1.5.3 openpyxl==3.1.5 h5py==3.15.1
```

