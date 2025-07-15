# Zero-Day OCSVM Detection System

## Project Overview
A streamlined Streamlit application focused on training One-Class Support Vector Machines (OCSVM) on normal traffic from KDDTrain+ dataset and testing them against authentic zero-day attacks from KDDTest+. The system follows a specific workflow: train on normal traffic patterns, then evaluate detection performance on previously unseen attack types.

## Recent Changes
- **2025-07-15**: Created multi-page application with separate ANN and OCSVM implementations
- **2025-07-15**: Fixed numpy format string error by converting values to native Python types
- **2025-07-15**: Implemented comprehensive ANN page with full supervised learning workflow
- **2025-07-15**: Added OCSVM page with one-class anomaly detection approach
- **2025-07-15**: Both approaches now train and test against zero-day attacks independently
- **2025-07-15**: Added hyperparameter optimization for OCSVM with grid search
- **2025-07-15**: Created binary prediction interface for consistent anomaly detection
- **2025-07-15**: Added support for RBF, linear, poly, and sigmoid kernels in OCSVM

## Project Architecture

### Core Components
1. **Main Application** (`zero_day_app.py`)
   - Multi-page interface with ANN and OCSVM approaches
   - Page 1: ANN - Train on known attacks, test against zero-day
   - Page 2: OCSVM - Train on normal traffic, test against zero-day
   - Each page has 3 tabs: Train, Test, Analyze

2. **Model Implementations**
   - `models/anomaly_detector.py`: ANN-based supervised learning
   - `models/ocsvm_detector.py`: OCSVM-based anomaly detection
   - Both support comprehensive evaluation and hyperparameter tuning

3. **Utilities**
   - `utils/kdd_zero_day_filter.py`: KDD dataset filtering for proper separation
   - `utils/model_evaluation.py`: Model evaluation and metrics

4. **Data**
   - `attached_assets/KDDTrain+ copy.txt`: Training data with known attacks
   - `attached_assets/KDDTest+ copy.txt`: Test data with zero-day attacks

### Key Features
- **Dual Approach Comparison**: Side-by-side ANN vs OCSVM implementations
- **Proper Data Separation**: KDDTrain+ for training, KDDTest+ zero-day attacks for testing
- **Authentic Zero-Day Evaluation**: Tests only on attack types not seen during training
- **ANN Training**: Supervised learning on known attacks (back, neptune, satan, smurf, etc.)
- **OCSVM Training**: Unsupervised learning on normal traffic patterns only
- **Zero-Day Attack Testing**: Tests on apache2, mailbomb, processtable, snmpgetattack, etc.
- **Professional Dataset**: 125,974+ KDDTrain+ records, 22,544+ KDDTest+ records
- **Comprehensive Analysis**: Detailed performance metrics and attack-specific results
- **Clean Workflow**: Simple 3-step process for each approach

## Development Notes

### Running Locally
When running in development environments (IntelliJ, VS Code, etc.), you may see:
```
Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
```
This warning is normal and harmless - the application functions correctly despite this message.

### KDD Dataset Structure
- 41 features including protocol, service, duration, network statistics
- 23 attack types: DoS, Probe, R2L, U2R categories
- Binary classification: normal vs anomaly detection
- Automatic preprocessing converts categorical features to numeric

### Zero-Day Detection Workflow
**ANN Approach:**
1. **Train Phase**: Load KDDTrain+ → Train on known attacks → Supervised learning
2. **Test Phase**: Load KDDTest+ → Filter zero-day attacks → Test generalization
3. **Analyze Phase**: Evaluate performance → Review attack-specific results

**OCSVM Approach:**
1. **Train Phase**: Load KDDTrain+ → Filter normal traffic → Train OCSVM
2. **Test Phase**: Load KDDTest+ → Filter zero-day attacks → Test anomaly detection
3. **Analyze Phase**: Evaluate performance → Review attack-specific results

## User Preferences
- Focus purely on zero-day detection capabilities
- Use authentic KDD Cup 1999 data with proper separation
- Streamlined single-purpose application
- Clear evaluation of detection performance on unseen attacks

## Technical Specifications
- **Framework**: Streamlit for web interface
- **ML Library**: Scikit-learn for OCSVM implementation
- **Data Processing**: Pandas, NumPy, Scikit-learn preprocessing
- **Visualization**: Plotly for interactive charts
- **Dataset**: KDD Cup 1999 with proper normal/zero-day separation
- **Evaluation**: Custom zero-day specific metrics and analysis