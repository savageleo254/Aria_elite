import asyncio
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import google.generativeai as genai
from pathlib import Path
import joblib
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# CPU-optimized ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
from lightgbm import LGBMClassifier

from ..utils.config_loader import ConfigLoader
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class AutonomousModelTrainer:
    """
    Autonomous Model Training System - Gemini-controlled model training
    Optimized for CPU-only systems (i5 7th gen, 8GB RAM)
    Prioritizes: Profits > Accuracy > Win Rate > Precision
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.gemini_model = None
        self.is_initialized = False
        
        # System constraints for CPU optimization
        self.system_specs = {
            "cpu_cores": psutil.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "max_workers": min(4, psutil.cpu_count()),  # Conservative for 8GB RAM
            "max_memory_usage": 0.7,  # Use max 70% of RAM
            "batch_size": 1000,  # Optimized for CPU
            "n_jobs": -1  # Use all cores
        }
        
        # Model training state
        self.training_queue = []
        self.active_training = {}
        self.model_registry = {}
        self.performance_history = []
        self.training_insights = []
        
        # CPU-optimized model configurations
        self.model_configs = self._get_cpu_optimized_configs()
        
        # Training priorities
        self.training_priorities = {
            "profits": 0.4,      # 40% weight
            "accuracy": 0.25,    # 25% weight
            "win_rate": 0.2,     # 20% weight
            "precision": 0.15    # 15% weight
        }
        
    async def initialize(self):
        """Initialize the autonomous model trainer"""
        try:
            logger.info("Initializing Autonomous Model Trainer - CPU Optimized")
            
            # Load configuration
            await self._load_config()
            
            # Initialize Gemini API
            await self._initialize_gemini_api()
            
            # Check system resources
            await self._check_system_resources()
            
            # Load existing models
            await self._load_existing_models()
            
            # Start training loops
            asyncio.create_task(self._autonomous_training_loop())
            asyncio.create_task(self._model_evaluation_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            
            self.is_initialized = True
            logger.info("Autonomous Model Trainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Autonomous Model Trainer: {str(e)}")
            raise
    
    async def _load_config(self):
        """Load training configuration"""
        try:
            self.project_config = self.config.load_project_config()
            self.strategy_config = self.config.load_strategy_config()
            
            # Get Gemini API key
            self.api_key = os.getenv('GEMINI_API_KEY')
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            logger.info("Training configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load training configuration: {str(e)}")
            raise
    
    async def _initialize_gemini_api(self):
        """Initialize Gemini API for model training control"""
        try:
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Test connection
            test_response = await self._test_training_connection()
            if test_response:
                logger.info("Gemini API connected for model training")
            else:
                raise Exception("Failed to establish training connection")
                
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise
    
    async def _test_training_connection(self) -> bool:
        """Test Gemini connection for training"""
        try:
            prompt = """
            You are an expert ML engineer and trader. You will autonomously train models
            optimized for CPU-only systems, prioritizing profits, accuracy, win rate, and precision.
            Respond with 'TRAINING MODE ACTIVATED' if you understand.
            """
            response = self.gemini_model.generate_content(prompt)
            return "TRAINING MODE" in response.text.upper()
        except Exception as e:
            logger.error(f"Training connection test failed: {str(e)}")
            return False
    
    def _get_cpu_optimized_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get CPU-optimized model configurations"""
        return {
            "random_forest": {
                "class": RandomForestClassifier,
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "n_jobs": -1,
                    "random_state": 42,
                    "max_features": "sqrt"
                }
            },
            "gradient_boosting": {
                "class": GradientBoostingClassifier,
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42
                }
            },
            "extra_trees": {
                "class": ExtraTreesClassifier,
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "n_jobs": -1,
                    "random_state": 42
                }
            },
            "logistic_regression": {
                "class": LogisticRegression,
                "params": {
                    "max_iter": 1000,
                    "random_state": 42,
                    "n_jobs": -1,
                    "solver": "liblinear"
                }
            },
            "svm": {
                "class": SVC,
                "params": {
                    "kernel": "rbf",
                    "C": 1.0,
                    "gamma": "scale",
                    "random_state": 42,
                    "probability": True
                }
            },
            "mlp": {
                "class": MLPClassifier,
                "params": {
                    "hidden_layer_sizes": (100, 50),
                    "max_iter": 500,
                    "random_state": 42,
                    "early_stopping": True,
                    "validation_fraction": 0.1
                }
            },
            "xgboost": {
                "class": xgb.XGBClassifier,
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "n_jobs": -1,
                    "tree_method": "hist"  # CPU optimized
                }
            },
            "lightgbm": {
                "class": LGBMClassifier,
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "n_jobs": -1,
                    "device": "cpu"
                }
            }
        }
    
    async def _check_system_resources(self):
        """Check system resources and adjust configurations"""
        try:
            # Check available RAM
            available_ram = psutil.virtual_memory().available / (1024**3)
            logger.info(f"Available RAM: {available_ram:.2f} GB")
            
            # Adjust batch size based on available RAM
            if available_ram < 4:
                self.system_specs["batch_size"] = 500
                self.system_specs["max_workers"] = 2
            elif available_ram < 6:
                self.system_specs["batch_size"] = 750
                self.system_specs["max_workers"] = 3
            
            # Adjust model parameters for memory constraints
            for model_name, config in self.model_configs.items():
                if "n_estimators" in config["params"]:
                    config["params"]["n_estimators"] = min(100, config["params"]["n_estimators"])
                if "hidden_layer_sizes" in config["params"]:
                    config["params"]["hidden_layer_sizes"] = (50, 25)
            
            logger.info(f"System specs optimized: {self.system_specs}")
            
        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
    
    async def _load_existing_models(self):
        """Load existing trained models"""
        try:
            models_dir = Path("backend/models/trained")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            for model_file in models_dir.glob("*.pkl"):
                try:
                    model_name = model_file.stem
                    model = joblib.load(model_file)
                    self.model_registry[model_name] = {
                        "model": model,
                        "file_path": model_file,
                        "loaded_at": datetime.now()
                    }
                    logger.info(f"Loaded existing model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading model {model_file}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error loading existing models: {str(e)}")
    
    async def _autonomous_training_loop(self):
        """Main autonomous training loop"""
        while True:
            try:
                # Check if we should train new models
                if await self._should_train_new_models():
                    await self._initiate_autonomous_training()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in autonomous training loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _should_train_new_models(self) -> bool:
        """Determine if we should train new models"""
        try:
            # Check if we have recent performance data
            if not self.performance_history:
                return True
            
            # Check if models are outdated (older than 7 days)
            latest_model_time = max(
                [model["loaded_at"] for model in self.model_registry.values()],
                default=datetime.min
            )
            
            if (datetime.now() - latest_model_time).days > 7:
                return True
            
            # Check if performance is degrading
            recent_performance = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
            if recent_performance:
                avg_performance = np.mean([p["overall_score"] for p in recent_performance])
                if avg_performance < 0.7:  # Performance threshold
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if should train models: {str(e)}")
            return True
    
    async def _initiate_autonomous_training(self):
        """Initiate autonomous model training"""
        try:
            logger.info("Initiating autonomous model training")
            
            # Get training plan from Gemini
            training_plan = await self._get_training_plan_from_gemini()
            
            if training_plan.get("models_to_train"):
                for model_spec in training_plan["models_to_train"]:
                    await self._train_model_autonomously(model_spec)
            
        except Exception as e:
            logger.error(f"Error initiating autonomous training: {str(e)}")
    
    async def _get_training_plan_from_gemini(self) -> Dict[str, Any]:
        """Get training plan from Gemini AI"""
        try:
            # Prepare context for Gemini
            context = {
                "system_specs": self.system_specs,
                "available_models": list(self.model_configs.keys()),
                "training_priorities": self.training_priorities,
                "recent_performance": self.performance_history[-5:] if self.performance_history else [],
                "existing_models": list(self.model_registry.keys())
            }
            
            prompt = f"""
            As an expert ML engineer and trader, create a training plan for this CPU-only system:
            
            System Specs: {json.dumps(self.system_specs, indent=2)}
            Available Models: {list(self.model_configs.keys())}
            Training Priorities: {json.dumps(self.training_priorities, indent=2)}
            Recent Performance: {json.dumps(self.performance_history[-5:] if self.performance_history else [], indent=2)}
            
            Create a training plan that:
            1. Prioritizes profits, accuracy, win rate, precision
            2. Optimizes for CPU-only training (i5 7th gen, 8GB RAM)
            3. Uses ensemble methods for better performance
            4. Includes feature selection and preprocessing
            
            Respond with JSON:
            {{
                "models_to_train": [
                    {{
                        "name": "model_name",
                        "type": "model_type",
                        "features": ["feature1", "feature2"],
                        "target": "target_column",
                        "preprocessing": {{"scaler": "StandardScaler", "feature_selection": true}},
                        "hyperparameters": {{"param1": "value1"}},
                        "priority": 1-10
                    }}
                ],
                "ensemble_strategy": "voting|stacking|bagging",
                "validation_strategy": "time_series_split",
                "optimization_goal": "profits|accuracy|win_rate|precision",
                "reasoning": "explanation"
            }}
            """
            
            response = self.gemini_model.generate_content(prompt)
            training_plan = self._parse_training_plan_response(response.text)
            
            logger.info(f"Received training plan: {training_plan.get('reasoning', 'No reasoning provided')}")
            return training_plan
            
        except Exception as e:
            logger.error(f"Error getting training plan from Gemini: {str(e)}")
            return {"models_to_train": []}
    
    def _parse_training_plan_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response for training plan"""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {"models_to_train": []}
                
        except Exception as e:
            logger.error(f"Error parsing training plan response: {str(e)}")
            return {"models_to_train": []}
    
    async def _train_model_autonomously(self, model_spec: Dict[str, Any]):
        """Train a model autonomously"""
        try:
            model_name = model_spec.get("name", f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            model_type = model_spec.get("type", "random_forest")
            
            logger.info(f"Training model: {model_name} ({model_type})")
            
            # Get training data
            training_data = await self._prepare_training_data(model_spec)
            if training_data is None:
                logger.error(f"No training data available for {model_name}")
                return
            
            # Train model
            model_result = await self._train_single_model(model_name, model_type, training_data, model_spec)
            
            if model_result:
                # Evaluate model
                evaluation = await self._evaluate_model(model_result, training_data)
                
                # Save model if performance is good
                if evaluation["overall_score"] > 0.6:  # Performance threshold
                    await self._save_model(model_result, evaluation)
                    logger.info(f"Model {model_name} trained and saved successfully")
                else:
                    logger.warning(f"Model {model_name} performance too low: {evaluation['overall_score']}")
            
        except Exception as e:
            logger.error(f"Error training model {model_spec.get('name', 'unknown')}: {str(e)}")
    
    async def _prepare_training_data(self, model_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare training data for model"""
        try:
            # Get data from data engine
            from .data_engine import DataEngine
            data_engine = DataEngine()
            await data_engine.initialize()
            
            # Get features and target
            features = model_spec.get("features", ["close", "volume", "high", "low", "open"])
            target = model_spec.get("target", "direction")
            
            # Fetch training data
            symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            timeframes = ["H1", "H4"]
            
            data = await data_engine.fetch_training_data(symbols, timeframes, features + [target])
            
            if data is None or data.empty:
                logger.error("No training data available")
                return None
            
            # Prepare features and target
            X = data[features].fillna(0)
            y = data[target]
            
            # Remove rows with NaN target
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]
            
            # Apply preprocessing
            preprocessing = model_spec.get("preprocessing", {})
            if preprocessing.get("scaler"):
                scaler_type = preprocessing["scaler"]
                if scaler_type == "StandardScaler":
                    scaler = StandardScaler()
                elif scaler_type == "RobustScaler":
                    scaler = RobustScaler()
                else:
                    scaler = StandardScaler()
                
                X = scaler.fit_transform(X)
            
            # Feature selection
            if preprocessing.get("feature_selection", False):
                k_best = min(10, len(features))
                selector = SelectKBest(score_func=f_classif, k=k_best)
                X = selector.fit_transform(X, y)
                selected_features = [features[i] for i in selector.get_support(indices=True)]
            else:
                selected_features = features
            
            return {
                "X": X,
                "y": y,
                "features": selected_features,
                "scaler": scaler if preprocessing.get("scaler") else None,
                "selector": selector if preprocessing.get("feature_selection") else None
            }
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None
    
    async def _train_single_model(self, model_name: str, model_type: str, training_data: Dict[str, Any], model_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Train a single model"""
        try:
            X = training_data["X"]
            y = training_data["y"]
            
            # Get model configuration
            if model_type not in self.model_configs:
                logger.error(f"Unknown model type: {model_type}")
                return None
            
            model_config = self.model_configs[model_type]
            model_class = model_config["class"]
            base_params = model_config["params"].copy()
            
            # Apply custom hyperparameters
            custom_params = model_spec.get("hyperparameters", {})
            base_params.update(custom_params)
            
            # Create and train model
            model = model_class(**base_params)
            
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Train model
            model.fit(X, y)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            
            return {
                "name": model_name,
                "type": model_type,
                "model": model,
                "cv_scores": cv_scores,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "training_data": training_data,
                "hyperparameters": base_params,
                "trained_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error training model {model_name}: {str(e)}")
            return None
    
    async def _evaluate_model(self, model_result: Dict[str, Any], training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            model = model_result["model"]
            X = training_data["X"]
            y = training_data["y"]
            
            # Predictions
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            # Calculate win rate (assuming binary classification: 0=loss, 1=win)
            if len(np.unique(y)) == 2:
                win_rate = np.mean(y_pred == 1) if 1 in y_pred else 0
            else:
                win_rate = accuracy  # For multi-class, use accuracy as proxy
            
            # Calculate profit simulation (simplified)
            profit_score = await self._simulate_profits(y, y_pred, y_pred_proba)
            
            # Overall score based on priorities
            overall_score = (
                profit_score * self.training_priorities["profits"] +
                accuracy * self.training_priorities["accuracy"] +
                win_rate * self.training_priorities["win_rate"] +
                precision * self.training_priorities["precision"]
            )
            
            evaluation = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "win_rate": win_rate,
                "profit_score": profit_score,
                "overall_score": overall_score,
                "cv_mean": model_result["cv_mean"],
                "cv_std": model_result["cv_std"],
                "evaluated_at": datetime.now()
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {"overall_score": 0.0}
    
    async def _simulate_profits(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> float:
        """Simulate profit performance"""
        try:
            # Simple profit simulation
            # Assume each correct prediction gives +1, wrong gives -1
            profit = np.sum((y_true == y_pred).astype(int) * 2 - 1)
            
            # Normalize to 0-1 scale
            max_profit = len(y_true)
            profit_score = max(0, (profit + max_profit) / (2 * max_profit))
            
            return profit_score
            
        except Exception as e:
            logger.error(f"Error simulating profits: {str(e)}")
            return 0.0
    
    async def _save_model(self, model_result: Dict[str, Any], evaluation: Dict[str, Any]):
        """Save trained model"""
        try:
            model_name = model_result["name"]
            model = model_result["model"]
            training_data = model_result["training_data"]
            
            # Create model package
            model_package = {
                "model": model,
                "scaler": training_data.get("scaler"),
                "selector": training_data.get("selector"),
                "features": training_data["features"],
                "evaluation": evaluation,
                "hyperparameters": model_result["hyperparameters"],
                "trained_at": model_result["trained_at"]
            }
            
            # Save to file
            models_dir = Path("backend/models/trained")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            model_file = models_dir / f"{model_name}.pkl"
            joblib.dump(model_package, model_file)
            
            # Update registry
            self.model_registry[model_name] = {
                "model": model_package,
                "file_path": model_file,
                "loaded_at": datetime.now(),
                "evaluation": evaluation
            }
            
            # Record performance
            self.performance_history.append({
                "model_name": model_name,
                "evaluation": evaluation,
                "timestamp": datetime.now()
            })
            
            logger.info(f"Model {model_name} saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    async def _model_evaluation_loop(self):
        """Continuously evaluate model performance"""
        while True:
            try:
                await self._evaluate_all_models()
                await asyncio.sleep(7200)  # Evaluate every 2 hours
                
            except Exception as e:
                logger.error(f"Error in model evaluation loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _evaluate_all_models(self):
        """Evaluate all models in registry"""
        try:
            for model_name, model_info in self.model_registry.items():
                # Re-evaluate model performance
                evaluation = await self._evaluate_model_performance(model_name, model_info)
                
                # Update performance history
                self.performance_history.append({
                    "model_name": model_name,
                    "evaluation": evaluation,
                    "timestamp": datetime.now()
                })
            
        except Exception as e:
            logger.error(f"Error evaluating all models: {str(e)}")
    
    async def _evaluate_model_performance(self, model_name: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate performance of a specific model"""
        try:
            # This would evaluate model on recent data
            # For now, return mock evaluation
            return {
                "accuracy": 0.75,
                "precision": 0.72,
                "recall": 0.78,
                "f1_score": 0.75,
                "win_rate": 0.68,
                "profit_score": 0.82,
                "overall_score": 0.75
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            return {"overall_score": 0.0}
    
    async def _performance_monitoring_loop(self):
        """Monitor training performance and system resources"""
        while True:
            try:
                await self._monitor_training_performance()
                await asyncio.sleep(1800)  # Monitor every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _monitor_training_performance(self):
        """Monitor training performance and system resources"""
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Check if resources are available for training
            if cpu_percent > 80 or memory_percent > 80:
                logger.warning(f"High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%")
                # Pause training if resources are high
                return
            
            # Monitor model performance trends
            if len(self.performance_history) >= 10:
                recent_performance = [p["evaluation"]["overall_score"] for p in self.performance_history[-10:]]
                avg_performance = np.mean(recent_performance)
                
                if avg_performance < 0.6:
                    logger.warning(f"Model performance declining: {avg_performance:.3f}")
                    # Trigger retraining
                    await self._initiate_autonomous_training()
            
        except Exception as e:
            logger.error(f"Error monitoring training performance: {str(e)}")
    
    # Public methods for external control
    async def train_model_on_demand(self, model_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Train a model on demand"""
        try:
            logger.info(f"Training model on demand: {model_spec.get('name', 'unknown')}")
            
            # Add to training queue
            self.training_queue.append(model_spec)
            
            # Train immediately
            await self._train_model_autonomously(model_spec)
            
            return {"status": "completed", "model_name": model_spec.get("name")}
            
        except Exception as e:
            logger.error(f"Error training model on demand: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    async def get_model_performance(self, model_name: str = None) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            if model_name:
                if model_name in self.model_registry:
                    return self.model_registry[model_name]["evaluation"]
                else:
                    return {"error": f"Model {model_name} not found"}
            else:
                # Return all model performances
                performances = {}
                for name, info in self.model_registry.items():
                    performances[name] = info.get("evaluation", {})
                return performances
                
        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return {"error": str(e)}
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        try:
            return {
                "initialized": self.is_initialized,
                "system_specs": self.system_specs,
                "models_trained": len(self.model_registry),
                "training_queue_size": len(self.training_queue),
                "active_training": len(self.active_training),
                "performance_history_size": len(self.performance_history),
                "training_priorities": self.training_priorities
            }
            
        except Exception as e:
            logger.error(f"Error getting training status: {str(e)}")
            return {}
    
    async def predict_with_best_model(self, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction with the best performing model"""
        try:
            if not self.model_registry:
                return {"error": "No models available"}
            
            # Find best model
            best_model_name = None
            best_score = 0
            
            for name, info in self.model_registry.items():
                evaluation = info.get("evaluation", {})
                score = evaluation.get("overall_score", 0)
                if score > best_score:
                    best_score = score
                    best_model_name = name
            
            if best_model_name:
                model_package = self.model_registry[best_model_name]["model"]
                model = model_package["model"]
                
                # Apply preprocessing if available
                if model_package.get("scaler"):
                    features = model_package["scaler"].transform(features.reshape(1, -1))
                if model_package.get("selector"):
                    features = model_package["selector"].transform(features.reshape(1, -1))
                
                # Make prediction
                prediction = model.predict(features.reshape(1, -1))[0]
                probability = model.predict_proba(features.reshape(1, -1))[0] if hasattr(model, 'predict_proba') else None
                
                return {
                    "model_name": best_model_name,
                    "prediction": prediction,
                    "probability": probability.tolist() if probability is not None else None,
                    "confidence": max(probability) if probability is not None else 0.0,
                    "model_score": best_score
                }
            else:
                return {"error": "No valid models found"}
                
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {"error": str(e)}
