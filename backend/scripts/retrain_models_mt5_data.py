#!/usr/bin/env python3
"""
ARIA ELITE - MT5 Real Data Model Retraining Script
Retrain AI models using real MT5 historical XAUUSD data
"""

import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import math

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.append(str(backend_path))

try:
    import MetaTrader5 as mt5
    import lightgbm as lgb
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils.class_weight import compute_class_weight
    import talib
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install MetaTrader5 lightgbm tensorflow scikit-learn TA-Lib")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5DataModelTrainer:
    """Retrain AI models using real MT5 historical data"""
    
    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.mt5 = None
        
    def connect_mt5(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                return False
                
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get MT5 account info")
                return False
                
            logger.info(f"Connected to MT5: Account {account_info.login}, Server {account_info.server}")
            self.mt5 = mt5
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection failed: {str(e)}")
            return False
    
    def fetch_xauusd_data(self, timeframe: str = "H1", bars: int = 50000) -> pd.DataFrame:
        """Fetch real XAUUSD historical data from MT5"""
        logger.info(f"Fetching {bars} bars of XAUUSD {timeframe} data from MT5")
        
        # Map timeframe strings to MT5 constants
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        tf_constant = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
        
        # Fetch historical data
        rates = mt5.copy_rates_from_pos("XAUUSD", tf_constant, 0, bars)
        
        if rates is None or len(rates) == 0:
            raise ValueError(f"No XAUUSD data available for timeframe {timeframe}")
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        })
        
        logger.info(f"Fetched {len(df)} XAUUSD bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators using TA-Lib"""
        logger.info("Calculating technical indicators using TA-Lib")
        
        # Ensure all data is float64 and handle any NaN values
        df = df.copy()
        df['high'] = pd.to_numeric(df['high'], errors='coerce').astype(np.float64)
        df['low'] = pd.to_numeric(df['low'], errors='coerce').astype(np.float64)
        df['close'] = pd.to_numeric(df['close'], errors='coerce').astype(np.float64)
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype(np.float64)
        df['open'] = pd.to_numeric(df['open'], errors='coerce').astype(np.float64)
        
        # Convert to numpy arrays for TA-Lib (must be float64/double)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        open_price = df['open'].values
        
        # Momentum Indicators
        df['rsi'] = talib.RSI(close, timeperiod=14)
        df['rsi_30'] = talib.RSI(close, timeperiod=30)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)
        df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        df['cci'] = talib.CCI(high, low, close, timeperiod=14)
        df['momentum'] = talib.MOM(close, timeperiod=10)
        df['roc'] = talib.ROC(close, timeperiod=10)
        
        # Trend Indicators  
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)
        df['ema_50'] = talib.EMA(close, timeperiod=50)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility Indicators
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        df['natr'] = talib.NATR(high, low, close, timeperiod=14)
        
        # Volume Indicators
        df['ad'] = talib.AD(high, low, close, volume)
        df['obv'] = talib.OBV(close, volume)
        
        # Pattern Recognition (key patterns)
        df['cdl_doji'] = talib.CDLDOJI(open_price, high, low, close)
        df['cdl_hammer'] = talib.CDLHAMMER(open_price, high, low, close)
        df['cdl_engulfing'] = talib.CDLENGULFING(open_price, high, low, close)
        
        # Price Action Features
        df['price_change'] = pd.Series(close).pct_change()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        df['high_low_ratio'] = (high - low) / close
        
        # Advanced Features
        df['rsi_divergence'] = df['rsi'].diff()
        df['macd_crossover'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
        df['price_vs_sma20'] = (close - df['sma_20']) / df['sma_20']
        df['price_vs_ema50'] = (close - df['ema_50']) / df['ema_50']
        
        # Market Structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
        df['trend_strength'] = df['sma_20'] - df['sma_50']
        
        logger.info(f"Calculated {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} technical indicators")
        return df
    
    def create_target_labels(self, df: pd.DataFrame, future_periods: int = 5, profit_threshold: float = 0.003) -> pd.DataFrame:
        """Create target labels based on future price movements"""
        logger.info(f"Creating target labels with {future_periods} period lookahead")
        
        # Calculate future returns
        future_close = df['close'].shift(-future_periods)
        current_close = df['close']
        future_return = (future_close - current_close) / current_close
        
        # Create labels: 0=Sell, 1=Buy, 2=Hold
        df['target'] = 2  # Default to Hold
        df.loc[future_return > profit_threshold, 'target'] = 1  # Buy signal
        df.loc[future_return < -profit_threshold, 'target'] = 0  # Sell signal
        
        # Add confidence based on magnitude of return
        df['target_confidence'] = np.abs(future_return)
        
        logger.info(f"Label distribution - Buy: {sum(df['target']==1)}, Sell: {sum(df['target']==0)}, Hold: {sum(df['target']==2)}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare feature matrix and target vector"""
        # Select features (exclude non-numeric and target columns)
        exclude_cols = ['timestamp', 'target', 'target_confidence', 'open', 'high', 'low']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        # Clean data
        df_clean = df[feature_cols + ['target']].dropna()
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        
        logger.info(f"Prepared features: {len(feature_cols)} features, {len(df_clean)} samples")
        logger.info(f"Feature columns: {feature_cols[:10]}...") # Show first 10
        
        return X, y, feature_cols
    
    def train_lightgbm_model(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> str:
        """Train LightGBM model with real MT5 data"""
        logger.info("Training LightGBM model with real MT5 XAUUSD data")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Optimized parameters for financial data
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_weight': 0.001,
            'min_split_gain': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'random_state': 42,
            'force_row_wise': True
        }
        
        # Train with early stopping
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_class == y_test)
        
        # Feature importance
        importance = model.feature_importance()
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"LightGBM Model Performance:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Top 5 features: {feature_importance[:5]}")
        
        # Save model
        model_path = self.models_dir / "lightgbm_xauusd.onnx"
        
        model_wrapper = {
            'model': model,
            'feature_names': feature_names,
            'feature_importance': feature_importance,
            'model_type': 'lightgbm',
            'version': '2.0_mt5_real_data',
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_wrapper, model_path)
        logger.info(f"LightGBM model saved to {model_path}")
        
        return str(model_path)
    
    def train_gru_model(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> str:
        """Train GRU model with real MT5 data"""
        logger.info("Training GRU neural network with real MT5 XAUUSD data")
        
        # Prepare sequential data
        sequence_length = 60
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences
        sequences, targets = [], []
        for i in range(sequence_length, len(X_scaled)):
            sequences.append(X_scaled[i-sequence_length:i])
            targets.append(y[i])
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Split data
        split_idx = int(len(sequences) * 0.8)
        X_train, X_test = sequences[:split_idx], sequences[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]
        
        # Compute class weights to address potential imbalance
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
        
        # Build enhanced GRU model with robust regularization
        model = keras.Sequential([
            keras.layers.Input(shape=(sequence_length, len(feature_names))),
            keras.layers.GaussianNoise(0.01),
            keras.layers.SpatialDropout1D(0.1),
            keras.layers.GRU(
                128,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.l2(1e-4),
                recurrent_dropout=0.1
            ),
            keras.layers.Dropout(0.3),
            keras.layers.GRU(
                64,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.l2(1e-4),
                recurrent_dropout=0.1
            ),
            keras.layers.Dropout(0.3),
            keras.layers.GRU(
                32,
                return_sequences=False,
                kernel_regularizer=keras.regularizers.l2(1e-4)
            ),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
            keras.layers.Dense(3, activation='softmax', kernel_regularizer=keras.regularizers.l2(1e-4))
        ])
        
        # Compile with cosine-decay LR schedule and gradient clipping
        steps_per_epoch = max(1, math.ceil(len(X_train) / 64))
        lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.001,
            first_decay_steps=steps_per_epoch * 10,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-6
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train with improved callbacks
        checkpoint_path = self.models_dir / "gru_xauusd.best.keras"
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, min_delta=0.001, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=False)
        ]
        
        history = model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=200,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            shuffle=False,
            verbose=1
        )
        
        # Evaluate
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"GRU Model Performance:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Training samples: {len(X_train)}")
        
        # Save model
        model_path = self.models_dir / "gru_xauusd.onnx"
        
        model_wrapper = {
            'model': model,
            'scaler': scaler,
            'sequence_length': sequence_length,
            'feature_names': feature_names,
            'model_type': 'gru',
            'version': '2.0_mt5_real_data',
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_wrapper, model_path)
        logger.info(f"GRU model saved to {model_path}")
        
        return str(model_path)
    
    def train_regime_classifier(self, df: pd.DataFrame, feature_names: list) -> str:
        """Train market regime classifier with real MT5 data"""
        logger.info("Training Market Regime Classifier with real MT5 data")
        
        # Advanced regime classification based on multiple factors
        df_regime = df.copy()
        
        # Volatility regimes
        df_regime['volatility_20'] = df_regime['close'].pct_change().rolling(20).std() * np.sqrt(252)
        vol_low, vol_high = df_regime['volatility_20'].quantile([0.33, 0.67])
        
        # Trend strength
        df_regime['trend_20'] = df_regime['close'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        trend_threshold = df_regime['trend_20'].abs().quantile(0.5)
        
        # Market structure
        df_regime['rsi_regime'] = pd.cut(df_regime['rsi'], bins=[0, 30, 70, 100], labels=[0, 1, 2])
        
        # Create regime labels
        regime_labels = np.zeros(len(df_regime))
        
        for i in range(len(df_regime)):
            vol = df_regime['volatility_20'].iloc[i]
            trend = df_regime['trend_20'].iloc[i]
            rsi = df_regime['rsi'].iloc[i]
            
            if pd.isna(vol) or pd.isna(trend):
                regime_labels[i] = 2  # Default neutral
                continue
                
            if vol < vol_low:
                if abs(trend) > trend_threshold:
                    regime_labels[i] = 0  # Low vol trending
                else:
                    regime_labels[i] = 2  # Low vol sideways
            elif vol > vol_high:
                if abs(trend) > trend_threshold:
                    regime_labels[i] = 1  # High vol trending  
                else:
                    regime_labels[i] = 3  # High vol sideways
            else:
                regime_labels[i] = 2  # Medium vol
        
        df_regime['regime'] = regime_labels
        
        # Prepare features
        regime_features = [col for col in feature_names if col in df_regime.columns]
        df_clean = df_regime[regime_features + ['regime']].dropna()
        
        X = df_clean[regime_features].values
        y = df_clean['regime'].values
        
        # Train enhanced Random Forest
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        
        # Feature importance
        importance = model.feature_importances_
        feature_importance = list(zip(regime_features, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Regime Classifier Performance:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Top 5 features: {feature_importance[:5]}")
        
        # Save model
        model_path = self.models_dir / "regime_xauusd.pkl"
        
        regime_names = ['Low Vol Trending', 'High Vol Trending', 'Neutral/Sideways', 'High Vol Sideways']
        
        model_wrapper = {
            'model': model,
            'feature_names': regime_features,
            'feature_importance': feature_importance,
            'regime_names': regime_names,
            'model_type': 'regime_classifier',
            'version': '2.0_mt5_real_data',
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_wrapper, model_path)
        logger.info(f"Regime classifier saved to {model_path}")
        
        return str(model_path)

def main():
    """Main execution function"""
    logger.info("Starting ARIA ELITE MT5 Real Data Model Retraining")
    
    trainer = MT5DataModelTrainer()
    
    # Connect to MT5
    if not trainer.connect_mt5():
        logger.error("Failed to connect to MT5. Ensure MT5 terminal is running.")
        return
    
    try:
        # Fetch real XAUUSD data
        df = trainer.fetch_xauusd_data(timeframe="H1", bars=50000)
        
        # Calculate technical indicators
        df = trainer.calculate_technical_indicators(df)
        
        # Create target labels
        df = trainer.create_target_labels(df)
        
        # Prepare features
        X, y, feature_names = trainer.prepare_features(df)
        
        logger.info("="*70)
        logger.info("TRAINING MODELS WITH REAL MT5 DATA")
        logger.info("="*70)
        
        models = []
        
        # Train LightGBM
        lgb_path = trainer.train_lightgbm_model(X, y, feature_names)
        models.append(lgb_path)
        
        # Train GRU  
        gru_path = trainer.train_gru_model(X, y, feature_names)
        models.append(gru_path)
        
        # Train Regime Classifier
        regime_path = trainer.train_regime_classifier(df, feature_names)
        models.append(regime_path)
        
        logger.info("="*70)
        logger.info("MT5 REAL DATA MODEL TRAINING COMPLETED")
        logger.info("="*70)
        
        for model_path in models:
            logger.info(f"✓ {model_path}")
            
        logger.info("="*70)
        logger.info("ARIA ELITE now uses REAL MT5 XAUUSD data for institutional trading")
        
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        raise
    finally:
        if trainer.mt5:
            mt5.shutdown()

if __name__ == "__main__":
    main()
