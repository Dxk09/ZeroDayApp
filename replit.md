# Zero-Day Anomaly Detection System

## Project Overview
A comprehensive Streamlit-based cybersecurity anomaly detection system that uses machine learning to identify zero-day threats in network traffic. The system provides both Deep Neural Network (ANN) and One-Class Support Vector Machine (OCSVM) approaches for anomaly detection, with built-in model comparison capabilities.

## Recent Changes
- **2025-07-15**: Enhanced system for authentic zero-day attack detection and analysis
- **2025-07-15**: Added sophisticated Zero-Day Attack Simulator with realistic attack patterns
- **2025-07-15**: Implemented Advanced Persistent Threat (APT) simulation capabilities
- **2025-07-15**: Created comprehensive Zero-Day Testing Suite with evasion analysis
- **2025-07-15**: Added Threat Intelligence Engine for attack pattern analysis
- **2025-07-15**: Integrated polymorphic malware and AI-powered attack simulations
- **2025-07-15**: Enhanced model training focus on zero-day threat detection

## Project Architecture

### Core Components
1. **Data Upload & Preprocessing** (`pages/1_Data_Upload.py`)
   - Built-in KDD Cup 1999 dataset integration
   - Automatic categorical feature encoding
   - Binary label creation (normal=0, anomaly=1)
   - Custom file upload support for additional datasets

2. **Model Training** (`pages/2_Model_Training.py`)
   - Deep Neural Network (ANN) implementation
   - One-Class Support Vector Machine (OCSVM) implementation
   - Model comparison and accuracy analysis
   - Training progress visualization

3. **Model Classes**
   - `models/anomaly_detector.py`: ANN-based anomaly detection
   - `models/ocsvm_detector.py`: OCSVM-based anomaly detection

4. **Utilities**
   - `utils/data_preprocessing.py`: Data processing utilities
   - `utils/model_evaluation.py`: Model evaluation and metrics
   - `utils/visualization.py`: Visualization components

### Key Features
- **Zero-Day Attack Simulation**: Realistic APT, polymorphic malware, AI-powered, and supply chain attacks
- **Advanced Evasion Testing**: Sophisticated techniques to test model robustness
- **Threat Intelligence Engine**: Comprehensive attack pattern analysis and recommendations
- **Dual Algorithm Support**: Both ANN and OCSVM for different zero-day detection approaches
- **Built-in KDD Dataset**: 125,974+ authentic network records with real attack patterns
- **Model Comparison**: Side-by-side accuracy analysis for zero-day detection performance
- **Professional Security Data**: Authentic cybersecurity attack types for realistic testing

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

### Model Training Flow
1. Load KDD dataset → Automatic preprocessing → Ready for training
2. Choose ANN or OCSVM → Configure parameters → Train model
3. Evaluate performance → Compare models → Select best approach

## User Preferences
- Focus on authentic cybersecurity data (KDD Cup 1999)
- Provide both deep learning and classical ML options
- Enable model comparison for decision-making
- Streamlined workflow with minimal manual steps

## Technical Specifications
- **Framework**: Streamlit for web interface
- **ML Libraries**: PyTorch (ANN), Scikit-learn (OCSVM)
- **Data Processing**: Pandas, NumPy, Scikit-learn preprocessing
- **Visualization**: Plotly for interactive charts
- **Dataset**: KDD Cup 1999 Network Intrusion Detection