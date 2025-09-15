import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ConfigLoader
from utils.logger import setup_logger

logger = setup_logger(__name__)

class AIModelManager:
    """
    Manages AI models for trading predictions
    """
    
    def __init__(self):
        self.models = {}
        # Use the actual models directory where ONNX/PKL files exist
        self.model_dir = Path(__file__).parent
        self.model_status = {}
        self.training_data = None
        self.is_initialized = False
        self.model_files = {
            "lightgbm": "lightgbm_xauusd.pkl",
            "lightgbm_onnx": "lightgbm_xauusd.onnx", 
            "gru": "gru_xauusd.onnx",
            "gru_keras": "gru_xauusd.best.keras",
            "regime": "regime_xauusd.pkl"
        }
        
    async def initialize(self):
        """Initialize the AI model manager"""
        try:
            logger.info("Initializing AI Model Manager")
            
            # Load existing models
            await self._load_models()
            
            # Start model monitoring
            asyncio.create_task(self._model_monitoring_loop())
            
            self.is_initialized = True
            logger.info("AI Model Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Model Manager: {str(e)}")
            raise
    
    async def _load_models(self):
        """Load existing trained models"""
        try:
            # Load LightGBM model (PKL format)
            lgb_path = self.model_dir / self.model_files["lightgbm"]
            if lgb_path.exists():
                self.models["lightgbm"] = joblib.load(lgb_path)
                logger.info(f"LightGBM model loaded from {lgb_path}")
                self.model_status["lightgbm"] = "ready"
            else:
                logger.warning(f"LightGBM model not found: {lgb_path}")
                self.model_status["lightgbm"] = "missing"
            
            # Load regime classifier (PKL format)
            regime_path = self.model_dir / self.model_files["regime"]
            if regime_path.exists():
                self.models["regime"] = joblib.load(regime_path)
                logger.info(f"Regime classifier loaded from {regime_path}")
                self.model_status["regime"] = "ready"
            else:
                logger.warning(f"Regime classifier not found: {regime_path}")
                self.model_status["regime"] = "missing"
            
            # Load GRU model (ONNX format)
            try:
                import onnxruntime as ort
                gru_path = self.model_dir / self.model_files["gru"]
                if gru_path.exists():
                    self.models["gru"] = ort.InferenceSession(str(gru_path))
                    logger.info(f"GRU ONNX model loaded from {gru_path}")
                    self.model_status["gru"] = "ready"
                else:
                    logger.warning(f"GRU ONNX model not found: {gru_path}")
                    self.model_status["gru"] = "missing"
            except ImportError:
                logger.warning("ONNX Runtime not available - GRU model cannot be loaded")
                self.model_status["gru"] = "error"
            
            # Load LightGBM ONNX model as alternative
            try:
                import onnxruntime as ort
                lgb_onnx_path = self.model_dir / self.model_files["lightgbm_onnx"]
                if lgb_onnx_path.exists():
                    self.models["lightgbm_onnx"] = ort.InferenceSession(str(lgb_onnx_path))
                    logger.info(f"LightGBM ONNX model loaded from {lgb_onnx_path}")
                    self.model_status["lightgbm_onnx"] = "ready"
                else:
                    logger.warning(f"LightGBM ONNX model not found: {lgb_onnx_path}")
                    self.model_status["lightgbm_onnx"] = "missing"
            except ImportError:
                logger.warning("ONNX Runtime not available - LightGBM ONNX model cannot be loaded")
                self.model_status["lightgbm_onnx"] = "error"
            
            # Load Keras GRU model as alternative
            try:
                from tensorflow import keras
                gru_keras_path = self.model_dir / self.model_files["gru_keras"]
                if gru_keras_path.exists():
                    self.models["gru_keras"] = keras.models.load_model(str(gru_keras_path))
                    logger.info(f"GRU Keras model loaded from {gru_keras_path}")
                    self.model_status["gru_keras"] = "ready"
                else:
                    logger.warning(f"GRU Keras model not found: {gru_keras_path}")
                    self.model_status["gru_keras"] = "missing"
            except ImportError:
                logger.warning("TensorFlow not available - Keras GRU model cannot be loaded")
                self.model_status["gru_keras"] = "error"
                    
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    async def _model_monitoring_loop(self):
        """Monitor model performance and retrain if needed"""
        while True:
            try:
                await self._check_model_performance()
                await asyncio.sleep(3600)  # Check every hour
            except Exception as e:
                logger.error(f"Error in model monitoring loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def predict(self, features: np.ndarray, model_name: str = "lightgbm") -> Dict[str, Any]:
        """Make prediction using specified model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not loaded or available models: {list(self.models.keys())}")
            
            model = self.models[model_name]
            
            # Handle different model types
            if model_name in ["lightgbm", "regime"]:
                # PKL models (scikit-learn/lightgbm)
                if hasattr(model, 'predict'):
                    if features.ndim == 1:
                        features = features.reshape(1, -1)
                    prediction = model.predict(features)
                else:
                    raise ValueError(f"PKL model {model_name} does not have predict method")
                    
            elif model_name in ["gru", "lightgbm_onnx"]:
                # ONNX models
                try:
                    input_name = model.get_inputs()[0].name
                    if features.ndim < 3:  # ONNX models often expect 3D input for sequence models
                        features = features.reshape(1, features.shape[0], features.shape[1] if features.ndim > 1 else 1)
                    prediction = model.run(None, {input_name: features.astype(np.float32)})[0]
                except Exception as onnx_error:
                    logger.error(f"ONNX prediction error for {model_name}: {str(onnx_error)}")
                    # Try simpler input format
                    if features.ndim > 1:
                        features = features.flatten().reshape(1, -1)
                    prediction = model.run(None, {input_name: features.astype(np.float32)})[0]
                    
            elif model_name == "gru_keras":
                # Keras models
                if features.ndim < 3:  # Keras RNN models expect 3D input
                    features = features.reshape(1, features.shape[0], features.shape[1] if features.ndim > 1 else 1)
                prediction = model.predict(features, verbose=0)
                
            else:
                # Generic model handling
                if hasattr(model, 'predict'):
                    prediction = model.predict(features)
                else:
                    prediction = model(features)
            
            # Ensure prediction is a numpy array
            if not isinstance(prediction, np.ndarray):
                prediction = np.array(prediction)
            
            # Calculate confidence based on prediction distribution
            if prediction.size > 1:
                # For classification: use max probability as confidence
                if prediction.ndim > 1 and prediction.shape[1] > 1:
                    confidence = float(np.max(prediction, axis=1).mean())
                else:
                    confidence = float(np.mean(np.abs(prediction)))
            else:
                confidence = 0.5  # Default confidence for single predictions
            
            # Convert prediction to classification if needed
            if model_name == "regime":
                # Regime classifier output
                pred_class = int(np.argmax(prediction)) if prediction.size > 1 else int(prediction[0])
            elif prediction.size > 1 and prediction.ndim > 1:
                # Multi-class classification
                pred_class = int(np.argmax(prediction, axis=1)[0])
            else:
                # Binary or regression - convert to classification
                pred_value = float(prediction.flatten()[0])
                if pred_value > 0.6:
                    pred_class = 2  # Buy
                elif pred_value < -0.6:
                    pred_class = 0  # Sell
                else:
                    pred_class = 1  # Hold
            
            return {
                "prediction": [pred_class],
                "raw_prediction": prediction.tolist() if hasattr(prediction, 'tolist') else [float(prediction)],
                "confidence": float(confidence),
                "model_used": model_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction with {model_name}: {str(e)}")
            return {
                "prediction": [1],  # Default to HOLD
                "confidence": 0.0,
                "model_used": model_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _ensemble_predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make ensemble prediction using all available models"""
        try:
            predictions = []
            confidences = []
            
            for model_name, model in self.models.items():
                try:
                    prediction = await self.predict(features, model_name)
                    predictions.append(prediction["prediction"][0])
                    confidences.append(prediction["confidence"])
                except Exception as e:
                    logger.warning(f"Error with model {model_name}: {str(e)}")
                    continue
            
            if not predictions:
                return {"prediction": "hold", "confidence": 0.0, "model": "ensemble"}
            
            # Weighted average of predictions
            avg_confidence = sum(confidences) / len(confidences)
            avg_prediction = sum(predictions) / len(predictions)
                
            return {
                "prediction": [int(avg_prediction)],
                "confidence": avg_confidence,
                "model_used": "ensemble",
                "models_used": len(predictions),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            return {"prediction": "hold", "confidence": 0.0, "model": "ensemble"}
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {
            "models_loaded": len(self.models),
            "available_models": list(self.models.keys()),
            "model_details": {},
            "model_files": self.model_files,
            "model_directory": str(self.model_dir),
            "last_updated": datetime.now().isoformat()
        }
        
        # Add details for each expected model
        for model_key, filename in self.model_files.items():
            file_path = self.model_dir / filename
            model_loaded = model_key in self.models
            
            status["model_details"][model_key] = {
                "filename": filename,
                "file_exists": file_path.exists(),
                "file_size_mb": round(file_path.stat().st_size / (1024*1024), 2) if file_path.exists() else 0,
                "loaded": model_loaded,
                "status": self.model_status.get(model_key, "unknown"),
                "type": type(self.models[model_key]).__name__ if model_loaded else "Not loaded",
                "last_used": getattr(self.models.get(model_key), 'last_used', None)
            }
        
        return status
    
    async def retrain_all_models(self):
        """Retrain all models"""
        try:
            logger.info("Retraining all models")
            
            for model_name in self.models.keys():
                await self._retrain_model(model_name)
            
            logger.info("All models retrained successfully")
            
        except Exception as e:
            logger.error(f"Error retraining all models: {str(e)}")
    
    async def get_training_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get training data for a specific symbol and timeframe from MT5"""
        try:
            from .mt5_bridge import MT5Bridge
            
            # Initialize MT5 bridge for data fetching
            mt5_bridge = MT5Bridge()
            if not mt5_bridge.is_connected:
                await mt5_bridge.initialize()
            
            # Calculate data range - get last 1000 bars for training
            end_date = datetime.now()
            
            # Convert timeframe string to MT5 timeframe
            timeframe_map = {
                "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                "1h": 60, "4h": 240, "1d": 1440
            }
            mt5_timeframe = timeframe_map.get(timeframe, 60)  # Default to 1H
            
            # Fetch historical data from MT5
            historical_data = await mt5_bridge.get_historical_data(
                symbol=symbol,
                timeframe=mt5_timeframe,
                count=1000
            )
            
            if not historical_data or len(historical_data) < 50:
                logger.warning(f"Insufficient historical data for {symbol} {timeframe}")
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "features": [],
                    "labels": [],
                    "error": "Insufficient historical data",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Convert to pandas DataFrame for feature engineering
            df = pd.DataFrame(historical_data)
            
            # Technical indicator feature engineering
            features = self._engineer_features(df)
            labels = self._generate_labels(df)
            
            # Ensure we have valid data
            if len(features) == 0 or len(labels) == 0:
                logger.warning(f"Feature engineering failed for {symbol} {timeframe}")
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "features": [],
                    "labels": [],
                    "error": "Feature engineering failed",
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.info(f"Training data prepared: {len(features)} samples for {symbol} {timeframe}")
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "features": features.tolist() if hasattr(features, 'tolist') else features,
                "labels": labels.tolist() if hasattr(labels, 'tolist') else labels,
                "data_points": len(features),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting training data for {symbol} {timeframe}: {str(e)}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "features": [],
                "labels": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer technical indicator features from OHLCV data"""
        try:
            features = []
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                # Try alternative column names
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                })
            
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            
            # Volatility features
            df['volatility_5'] = df['price_change'].rolling(window=5).std()
            df['volatility_20'] = df['price_change'].rolling(window=20).std()
            
            # Volume features (if available)
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma_10']
            else:
                df['volume_ratio'] = 1.0  # Default if no volume data
            
            # RSI-like momentum
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Select feature columns
            feature_cols = [
                'price_change', 'high_low_ratio', 'close_open_ratio',
                'price_sma_5_ratio', 'price_sma_10_ratio', 'price_sma_20_ratio', 'price_sma_50_ratio',
                'volatility_5', 'volatility_20', 'volume_ratio', 'rsi'
            ]
            
            # Create feature matrix
            feature_matrix = df[feature_cols].fillna(0).values
            
            # Remove first 50 rows to avoid NaN from technical indicators
            if len(feature_matrix) > 50:
                feature_matrix = feature_matrix[50:]
            
            return feature_matrix
            
        except Exception as e:
            logger.error(f"Feature engineering error: {str(e)}")
            return np.array([])
    
    def _generate_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate trading labels from price data"""
        try:
            # Generate labels based on future price movement
            # 0: Sell, 1: Hold, 2: Buy
            
            # Calculate future returns (next 5 periods)
            future_returns = df['close'].shift(-5) / df['close'] - 1
            
            # Define thresholds for trading decisions
            buy_threshold = 0.005   # 0.5% upward movement
            sell_threshold = -0.005 # 0.5% downward movement
            
            labels = np.ones(len(df))  # Default to hold (1)
            labels[future_returns > buy_threshold] = 2   # Buy
            labels[future_returns < sell_threshold] = 0  # Sell
            
            # Remove first 50 rows to match features
            if len(labels) > 50:
                labels = labels[50:]
            
            # Remove last 5 rows (no future data available)
            if len(labels) > 5:
                labels = labels[:-5]
            
            return labels.astype(int)
            
        except Exception as e:
            logger.error(f"Label generation error: {str(e)}")
            return np.array([])