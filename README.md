# Customer Churn Prediction in Telecom Industry

A PyTorch-based binary classifier to predict customer churn in a telecommunications company using deep learning techniques.

## 📊 Project Overview

This project aims to predict whether a telecom customer will churn (leave the service) based on various customer features including tenure, service plans, usage patterns, and billing information. The model addresses the challenge of imbalanced classes commonly found in churn prediction scenarios while optimizing for precision and recall.

## 🎯 Key Features

- **Binary Classification**: Predicts customer churn probability (0-1)
- **Deep Learning**: Custom PyTorch neural network architecture
- **Comprehensive Preprocessing**: Feature engineering and normalization pipeline
- **Stratified Sampling**: Maintains class distribution across train/validation/test splits
- **Model Persistence**: Save and load trained models for future use

## 📈 Results

- **Test Accuracy**: 80.82%
- **Validation Accuracy**: 79.37%
- **Training Loss**: Converged to ~0.416 after 10 epochs
- **Model Size**: Lightweight architecture suitable for production deployment

## 📋 Dataset

The project uses the Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) containing:

### Customer Information
- **Demographics**: Gender, partner status, dependents
- **Account Details**: Tenure, contract type, payment method, paperless billing
- **Services**: Phone service, internet service, online security, online backup, device protection, tech support, streaming services
- **Financial**: Monthly charges, total charges
- **Target**: Churn status (Yes/No)

### Dataset Statistics
- **Total Samples**: 7,032 customers (after preprocessing)
- **Features**: 30 (after encoding and preprocessing)
- **Class Distribution**: Imbalanced (typical for churn datasets)

## 🛠️ Requirements

```txt
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## 🚀 Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/MonikaChaulagain/Customer-Churn-prediction.git
cd Customer-Churn-prediction
```

2. **Install dependencies**:
```bash
pip install torch pandas numpy scikit-learn
```

3. **Prepare dataset**:
   - Ensure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is placed in the `csv/` directory
   - Dataset structure: `csv/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## 📊 Data Preprocessing Pipeline

### 1. Data Cleaning
- Convert `TotalCharges` to numeric format
- Remove rows with missing values
- Drop `customerID` column (non-predictive)

### 2. Target Variable Encoding
```python
# Binary encoding for churn
'Yes' → 1, 'No' → 0
```

### 3. Feature Engineering

**Binary Categorical Features**:
- `gender`, `Partner`, `Dependents`, `PaperlessBilling`, `PhoneService`
- Encoded as: `Yes/Male` → 1, `No/Female` → 0

**Multi-categorical Features** (One-hot encoded):
- `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`
- `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- `Contract`, `PaymentMethod`

**Numerical Features** (Standardized):
- `tenure`, `MonthlyCharges`, `TotalCharges`
- Applied StandardScaler for normalization

## 🏗️ Model Architecture

### Neural Network Design
```python
CustomerChurn(
  (fc1): Linear(in_features=30, out_features=16, bias=True)
  (Relu): ReLU()
  (fc2): Linear(in_features=16, out_features=1, bias=True)
)
```

**Architecture Details**:
- **Input Layer**: 30 features
- **Hidden Layer**: 16 neurons with ReLU activation
- **Output Layer**: 1 neuron (binary classification)
- **Loss Function**: BCEWithLogitsLoss (handles sigmoid internally)
- **Optimizer**: Adam (lr=0.001)

## 🎮 Usage

### Training the Model
```bash
python customer_churn_prediction.py
```

### Loading Saved Model
```python
import torch
from your_model_file import CustomerChurn

# Initialize model
model = CustomerChurn(30, 16, 1)

# Load trained weights
model.load_state_dict(torch.load("customer_churn_prediction.pt"))
model.eval()
```

### Making Predictions
```python
# Prepare new customer data (must match preprocessing steps)
new_customer_features = preprocess_customer_data(customer_data)

# Convert to tensor
input_tensor = torch.tensor(new_customer_features, dtype=torch.float32)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    churn_probability = torch.sigmoid(output).item()
    
print(f"Churn Probability: {churn_probability:.4f}")
print(f"Prediction: {'Will Churn' if churn_probability > 0.5 else 'Will Stay'}")
```

## 📁 Project Structure

```
Customer-Churn-prediction/
│
├── csv/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer_churn_prediction.py          # Main training script
├── customer_churn_prediction.pt          # Trained model weights
├── README.md                             # Project documentation
└── requirements.txt                      # Dependencies (optional)
```

## 🔍 Training Process

### Data Splitting Strategy
- **Training**: 80% (5,625 samples)
- **Validation**: 10% (703 samples) 
- **Testing**: 10% (704 samples)
- **Method**: Stratified sampling to maintain class distribution

### Training Configuration
```python
batch_size = 64
num_epochs = 10
learning_rate = 0.001
optimizer = Adam
loss_function = BCEWithLogitsLoss
```

### Training Progress
```
Epoch 1: Loss: 0.4454
Epoch 2: Loss: 0.4309
Epoch 3: Loss: 0.4261
...
Epoch 10: Loss: 0.4159
```

## 📊 Model Performance

| Metric | Value |
|--------|--------|
| Training Loss (Final) | 0.4159 |
| Validation Loss | 0.4504 |
| Validation Accuracy | 79.37% |
| Test Accuracy | 80.82% |

## 🔄 Future Improvements

### Model Enhancements
- [ ] Add dropout layers for regularization
- [ ] Implement batch normalization
- [ ] Experiment with deeper architectures
- [ ] Try ensemble methods

### Class Imbalance Handling
- [ ] Implement SMOTE oversampling
- [ ] Use class weights in loss function
- [ ] Try focal loss for hard examples
- [ ] Threshold tuning for optimal precision/recall

### Evaluation & Monitoring
- [ ] Add precision, recall, F1-score metrics
- [ ] ROC-AUC analysis
- [ ] Confusion matrix visualization
- [ ] Cross-validation implementation
- [ ] Learning curves plotting

### Production Readiness
- [ ] Model versioning
- [ ] API endpoint development
- [ ] Model monitoring dashboard
- [ ] A/B testing framework

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Monika Chaulagain**
- GitHub: [@MonikaChaulagain](https://github.com/MonikaChaulagain)

## 🙏 Acknowledgments

- Dataset: IBM Sample Data Sets
- Framework: PyTorch Team
- Inspiration: Telecommunications industry churn prediction challenges

---

**Note**: This is a demonstration project for educational and portfolio purposes. For production deployment, additional validation, monitoring, and compliance measures would be required.
