# Zero-Day OCSVM Detection System

## Project Overview
A streamlined Streamlit application focused on training One-Class Support Vector Machines (OCSVM) on normal traffic from KDDTrain+ dataset and testing them against authentic zero-day attacks from KDDTest+. The system follows a specific workflow: train on normal traffic patterns, then evaluate detection performance on previously unseen attack types.

## Recent Changes
- **2025-07-15**: Replaced ANN with OCSVM for more effective zero-day detection
- **2025-07-15**: Implemented One-Class SVM focusing on normal traffic patterns
- **2025-07-15**: Added hyperparameter optimization with grid search
- **2025-07-15**: Created binary prediction interface for anomaly detection
- **2025-07-15**: Added support for RBF, linear, poly, and sigmoid kernels
- **2025-07-15**: Implemented proper anomaly scoring for zero-day threats
- **2025-07-15**: Streamlined training to focus on normal traffic only

## Project Architecture

### Core Components
1. **Main Application** (`zero_day_app.py`)
   - Streamlined 3-tab interface
   - Tab 1: Train ANN on KDDTrain+ (known attacks)
   - Tab 2: Test against KDDTest+ zero-day attacks
   - Tab 3: Analyze detection performance

2. **Model Implementation** (`models/ocsvm_detector.py`)
   - OCSVM-based anomaly detection
   - Scikit-learn implementation with multiple kernels
   - Hyperparameter optimization and evaluation

3. **Utilities**
   - `utils/kdd_zero_day_filter.py`: KDD dataset filtering for proper separation
   - `utils/model_evaluation.py`: Model evaluation and metrics

4. **Data**
   - `attached_assets/KDDTrain+ copy.txt`: Training data with known attacks
   - `attached_assets/KDDTest+ copy.txt`: Test data with zero-day attacks

### Key Features
- **Proper Data Separation**: KDDTrain+ for training, KDDTest+ zero-day attacks for testing
- **Authentic Zero-Day Evaluation**: Tests only on attack types not seen during training
- **Known Attack Training**: Trains on back, neptune, satan, smurf, teardrop, warezclient, etc.
- **Zero-Day Attack Testing**: Tests on apache2, mailbomb, processtable, snmpgetattack, etc.
- **Professional Dataset**: 125,974+ KDDTrain+ records, 22,544+ KDDTest+ records
- **Comprehensive Analysis**: Detailed performance metrics and attack-specific results
- **Clean Workflow**: Simple 3-step process focused on zero-day detection

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
1. **Train Phase**: Load KDDTrain+ → Filter normal traffic → Train OCSVM
2. **Test Phase**: Load KDDTest+ → Filter zero-day attacks → Test detection
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