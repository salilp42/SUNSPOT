# SUNSPOT 🌞

A novel approach to sunspot activity prediction using Temporal Fusion Transformers (TFT) and backtesting strategies.

## 🎯 Project Aims

1. Develop an accurate time series forecasting model for sunspot activity prediction
2. Implement and evaluate TFT (Temporal Fusion Transformer) for complex temporal patterns
3. Validate predictions through rigorous backtesting
4. Explore the relationship between model confidence and prediction accuracy

## 🔍 Novel Aspects

- **Advanced Architecture**: Utilizes state-of-the-art Temporal Fusion Transformer architecture for time series forecasting
- **Integrated Backtesting**: Combines ML predictions with trading strategy validation
- **Quantile Predictions**: Provides uncertainty estimates through quantile regression
- **Multi-horizon Forecasting**: Capable of both short and long-term predictions

## 📦 Installation

```bash
git clone https://github.com/salilp42/SUNSPOT.git
cd SUNSPOT
pip install -r requirements.txt
```

## 🚀 Usage

```python
from sunspot_predictor import SunspotPredictor

# Initialize predictor
predictor = SunspotPredictor()

# Load and preprocess data
train, val = predictor.load_data()

# Train model
predictor.create_model()
predictor.train_model()

# Evaluate performance
mape_score, rmse_score = predictor.evaluate_model()
```

## 🏗️ Project Structure

```
SUNSPOT/
├── README.md
├── LICENSE
├── requirements.txt
├── sunspot_predictor.py    # Main implementation
└── sunspot_results/        # Directory for output files
```

## 🚧 Work in Progress

This project is actively under development. Current focus areas:
- Optimizing model hyperparameters
- Implementing robust error handling in backtesting
- Adding more sophisticated trading strategies
- Expanding the feature engineering pipeline

## 🛠 Technical Stack

- Python 3.11+
- PyTorch
- Darts (Time Series Library)
- Backtesting.py
- Pandas & NumPy

## 📊 Data Source

The project uses the Sunspots dataset from Darts, which contains monthly sunspot numbers from 1749 to the present.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

- [salilp42](https://github.com/salilp42)

## 🤝 Contributing

This is a work in progress and contributions are welcome! Please feel free to submit a Pull Request.
