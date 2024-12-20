import os
import pandas as pd
import torch
from darts import TimeSeries
from darts.datasets import SunspotsDataset
from darts.models import TFTModel
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from backtesting import Backtest, Strategy

class SunspotPredictor:
    def __init__(self, output_dir="sunspot_results"):
        """Initialize the SunspotPredictor with configuration."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Model parameters
        self.input_chunk_length = 36
        self.output_chunk_length = 12
        self.hidden_size = 64
        self.lstm_layers = 1
        self.num_attention_heads = 4
        self.dropout = 0.1
        self.batch_size = 32
        self.n_epochs = 20
        
        # Initialize components
        self.scaler = Scaler()
        self.model = None
        self.series = None
        self.train = None
        self.val = None
        
    def load_data(self):
        """Load and preprocess the sunspots dataset."""
        # Load the dataset
        self.series = SunspotsDataset().load()
        
        # Create train/validation split
        train_len = int(len(self.series) * 0.8)
        self.train, self.val = self.series[:train_len], self.series[train_len:]
        
        # Scale the data
        self.train = self.scaler.fit_transform(self.train)
        self.val = self.scaler.transform(self.val)
        
        return self.train, self.val
    
    def create_model(self):
        """Create and configure the TFT model."""
        self.model = TFTModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9])
        )
        return self.model
    
    def train_model(self):
        """Train the TFT model on the training data."""
        if self.model is None:
            self.create_model()
            
        print("Training the TFT model on the training set...")
        self.model.fit(self.train)
        print("Training completed.")
        
    def evaluate_model(self):
        """Evaluate the model's performance on validation data."""
        # Make predictions
        predictions = self.model.predict(n=len(self.val))
        predictions = self.scaler.inverse_transform(predictions)
        actual_values = self.scaler.inverse_transform(self.val)
        
        # Calculate metrics
        validation_mape = float(mape(actual_values, predictions))
        validation_rmse = float(rmse(actual_values, predictions))
        
        print(f"Validation MAPE: {validation_mape}")
        print(f"Validation RMSE: {validation_rmse}")
        
        return validation_mape, validation_rmse

class TFTForecastStrategy(Strategy):
    """Trading strategy based on TFT predictions."""
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.position_held = 0
        
    def init(self):
        """Initialize the strategy."""
        super().init()
        self.position_held = 0
    
    def next(self):
        """Define the trading logic."""
        if len(self.data) < self.predictor.input_chunk_length:
            return
            
        # Get current position in the data
        current_idx = len(self.data) - 1
        
        # Create a TimeSeries object from recent data
        recent_data = TimeSeries.from_series(
            self.data.Close[current_idx - self.predictor.input_chunk_length + 1:current_idx + 1]
        )
        
        # Make prediction
        prediction = self.predictor.model.predict(recent_data, n=1)
        current_price = self.data.Close[-1]
        predicted_price = prediction.values()[-1][0]
        
        # Trading logic
        if predicted_price > current_price * 1.02 and self.position_held <= 0:
            self.buy()
            self.position_held = 1
        elif predicted_price < current_price * 0.98 and self.position_held >= 0:
            self.sell()
            self.position_held = -1

def main():
    # Initialize predictor
    predictor = SunspotPredictor()
    
    # Load and preprocess data
    train, val = predictor.load_data()
    
    # Create and train model
    predictor.create_model()
    predictor.train_model()
    
    # Evaluate model
    mape_score, rmse_score = predictor.evaluate_model()
    
    # Prepare data for backtesting
    df_bt = predictor.series.pd_dataframe()
    
    # Run backtesting
    bt = Backtest(
        df_bt,
        TFTForecastStrategy,
        cash=10000,
        commission=0.002,
        exclusive_orders=True
    )
    
    stats = bt.run()
    print("\nBacktesting Results:")
    print(stats)
