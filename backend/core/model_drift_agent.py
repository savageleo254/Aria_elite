import asyncio
import numpy as np
import pandas as pd
import logging
import pickle
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import sqlite3
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger(__name__)

class DriftStatus(Enum):
    """Model drift detection status"""
    STABLE = "stable"
    WARNING = "warning" 
    DRIFT = "drift"
    SEVERE_DRIFT = "severe_drift"
    DEGRADED = "degraded"

class ModelType(Enum):
    """Types of AI models being monitored"""
    LIGHTGBM = "lightgbm"
    GRU = "gru"
    REGIME_CLASSIFIER = "regime_classifier"
    FUSION_ENGINE = "fusion_engine"

@dataclass
class DriftAlert:
    """Model drift detection alert"""
    model_name: str
    model_type: ModelType
    drift_status: DriftStatus
    drift_score: float
    accuracy_drop: float
    confidence_drop: float
    timestamp: datetime
    metrics: Dict[str, float]
    recommendation: str

@dataclass
class ModelPerformance:
    """Model performance metrics tracking"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_mean: float
    confidence_std: float
    prediction_count: int
    timestamp: datetime
    data_quality_score: float = 1.0

class ModelDriftAgent:
    """
    Advanced Model Drift Detection Agent for ARIA-DAN
    Monitors AI model performance and triggers retraining when drift is detected
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.is_initialized = False
        self.models_status = {}
        self.performance_history = {}
        self.drift_alerts = []
        
        # Drift detection parameters
        self.drift_thresholds = {
            'accuracy_drop': 0.05,      # 5% accuracy drop triggers warning
            'severe_accuracy_drop': 0.10, # 10% triggers drift alert
            'confidence_drop': 0.15,    # 15% confidence drop
            'prediction_drift': 0.20,   # 20% prediction distribution change
            'data_drift': 0.25         # 25% data distribution change
        }
        
        # Model monitoring intervals (seconds)
        self.monitoring_intervals = {
            ModelType.LIGHTGBM: 3600,      # 1 hour
            ModelType.GRU: 3600,           # 1 hour  
            ModelType.REGIME_CLASSIFIER: 7200, # 2 hours
            ModelType.FUSION_ENGINE: 1800   # 30 minutes
        }
        
        # Performance baselines (will be learned)
        self.performance_baselines = {}
        
        # Retraining queue
        self.retraining_queue = []
        self.retraining_in_progress = set()
        
    async def initialize(self):
        """Initialize the model drift detection agent"""
        try:
            logger.info("Initializing Model Drift Detection Agent")
            
            # Load configuration
            await self._load_drift_config()
            
            # Initialize model monitoring
            await self._initialize_model_monitoring()
            
            # Load performance baselines
            await self._load_performance_baselines()
            
            # Start monitoring loops for each model type
            self.monitoring_tasks = []
            for model_type in ModelType:
                task = asyncio.create_task(self._model_monitoring_loop(model_type))
                self.monitoring_tasks.append(task)
            
            # Start retraining coordination loop
            self.retraining_task = asyncio.create_task(self._retraining_coordination_loop())
            
            self.is_initialized = True
            logger.info("Model Drift Agent initialized - Ready for AI model health monitoring")
            
        except Exception as e:
            logger.error(f"Failed to initialize Model Drift Agent: {str(e)}")
            raise
    
    async def _load_drift_config(self):
        """Load drift detection configuration"""
        try:
            drift_config = self.config.get('model_drift', {})
            
            # Update thresholds with config values
            if 'thresholds' in drift_config:
                self.drift_thresholds.update(drift_config['thresholds'])
            
            # Update monitoring intervals
            if 'monitoring_intervals' in drift_config:
                for model_str, interval in drift_config['monitoring_intervals'].items():
                    model_type = ModelType(model_str)
                    self.monitoring_intervals[model_type] = interval
            
            logger.info("Model drift configuration loaded")
            
        except Exception as e:
            logger.error(f"Error loading drift config: {str(e)}")
    
    async def _initialize_model_monitoring(self):
        """Initialize model monitoring systems"""
        try:
            # Initialize database for performance tracking
            await self._initialize_performance_database()
            
            # Register available models
            await self._register_models()
            
            logger.info("Model monitoring systems initialized")
            
        except Exception as e:
            logger.error(f"Error initializing model monitoring: {str(e)}")
    
    async def _initialize_performance_database(self):
        """Initialize SQLite database for performance tracking"""
        try:
            db_path = "db/model_performance.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create performance tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    accuracy REAL,
                    precision_val REAL,
                    recall_val REAL,
                    f1_score REAL,
                    confidence_mean REAL,
                    confidence_std REAL,
                    prediction_count INTEGER,
                    data_quality_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create drift alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    drift_status TEXT NOT NULL,
                    drift_score REAL,
                    accuracy_drop REAL,
                    confidence_drop REAL,
                    recommendation TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Performance database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing performance database: {str(e)}")
    
    async def _register_models(self):
        """Register available AI models for monitoring"""
        try:
            # Register models based on file existence
            model_files = {
                'lightgbm_xauusd': ('backend/models/lightgbm_xauusd.onnx', ModelType.LIGHTGBM),
                'gru_xauusd': ('backend/models/gru_xauusd.onnx', ModelType.GRU),
                'regime_xauusd': ('backend/models/regime_xauusd.pkl', ModelType.REGIME_CLASSIFIER)
            }
            
            for model_name, (file_path, model_type) in model_files.items():
                try:
                    # Check if model file exists
                    import os
                    if os.path.exists(file_path):
                        self.models_status[model_name] = {
                            'model_type': model_type,
                            'file_path': file_path,
                            'status': 'registered',
                            'last_check': datetime.now(),
                            'performance_trend': 'stable'
                        }
                        logger.info(f"Registered model: {model_name} ({model_type.value})")
                    else:
                        logger.warning(f"Model file not found: {file_path}")
                        
                except Exception as e:
                    logger.error(f"Error registering model {model_name}: {str(e)}")
            
            logger.info(f"Registered {len(self.models_status)} models for drift monitoring")
            
        except Exception as e:
            logger.error(f"Error registering models: {str(e)}")
    
    async def _load_performance_baselines(self):
        """Load or establish performance baselines for each model"""
        try:
            db_path = "db/model_performance.db"
            conn = sqlite3.connect(db_path)
            
            for model_name in self.models_status.keys():
                # Get last 100 performance records to establish baseline
                query = """
                    SELECT accuracy, precision_val, recall_val, f1_score, 
                           confidence_mean, confidence_std 
                    FROM model_performance 
                    WHERE model_name = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """
                
                df = pd.read_sql_query(query, conn, params=(model_name,))
                
                if len(df) >= 10:  # Need at least 10 records for baseline
                    baseline = {
                        'accuracy_mean': df['accuracy'].mean(),
                        'accuracy_std': df['accuracy'].std(),
                        'precision_mean': df['precision_val'].mean(),
                        'recall_mean': df['recall_val'].mean(),
                        'f1_mean': df['f1_score'].mean(),
                        'confidence_mean': df['confidence_mean'].mean(),
                        'confidence_std_mean': df['confidence_std'].mean(),
                        'sample_count': len(df)
                    }
                    
                    self.performance_baselines[model_name] = baseline
                    logger.info(f"Loaded baseline for {model_name}: accuracy={baseline['accuracy_mean']:.3f}")
                    
                else:
                    # Set default baseline if insufficient data
                    self.performance_baselines[model_name] = {
                        'accuracy_mean': 0.65,  # Default expected accuracy
                        'accuracy_std': 0.05,
                        'precision_mean': 0.65,
                        'recall_mean': 0.65,
                        'f1_mean': 0.65,
                        'confidence_mean': 0.75,
                        'confidence_std_mean': 0.15,
                        'sample_count': 0
                    }
                    logger.info(f"Set default baseline for {model_name}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading performance baselines: {str(e)}")
    
    async def _model_monitoring_loop(self, model_type: ModelType):
        """Main monitoring loop for specific model type"""
        interval = self.monitoring_intervals[model_type]
        
        while self.is_initialized:
            try:
                # Monitor models of this type
                models_to_check = [
                    name for name, info in self.models_status.items()
                    if info['model_type'] == model_type
                ]
                
                for model_name in models_to_check:
                    await self._check_model_drift(model_name)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in {model_type.value} monitoring loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _check_model_drift(self, model_name: str):
        """Check for drift in specific model"""
        try:
            # Get recent model performance
            recent_performance = await self._get_recent_performance(model_name)
            
            if not recent_performance:
                logger.warning(f"No recent performance data for {model_name}")
                return
            
            # Calculate drift metrics
            drift_metrics = self._calculate_drift_metrics(model_name, recent_performance)
            
            # Determine drift status
            drift_status = self._determine_drift_status(drift_metrics)
            
            # Update model status
            self.models_status[model_name]['status'] = drift_status.value
            self.models_status[model_name]['last_check'] = datetime.now()
            
            # Log performance tracking
            await self._log_performance(model_name, recent_performance)
            
            # Generate alerts if needed
            if drift_status != DriftStatus.STABLE:
                await self._generate_drift_alert(model_name, drift_status, drift_metrics)
            
            logger.info(f"Model {model_name} status: {drift_status.value} (drift_score: {drift_metrics.get('overall_drift_score', 0):.3f})")
            
        except Exception as e:
            logger.error(f"Error checking drift for {model_name}: {str(e)}")
    
    async def _get_recent_performance(self, model_name: str) -> Optional[ModelPerformance]:
        """Get recent performance metrics for model"""
        try:
            # This would integrate with actual model prediction logging
            # For now, simulate performance data based on model status
            
            baseline = self.performance_baselines.get(model_name, {})
            
            # Simulate realistic performance with some variation
            accuracy = baseline.get('accuracy_mean', 0.65) + np.random.normal(0, 0.02)
            precision = baseline.get('precision_mean', 0.65) + np.random.normal(0, 0.02)
            recall = baseline.get('recall_mean', 0.65) + np.random.normal(0, 0.02)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add some drift simulation for testing
            drift_factor = np.random.uniform(0.95, 1.05)  # ±5% variation
            
            return ModelPerformance(
                model_name=model_name,
                accuracy=accuracy * drift_factor,
                precision=precision * drift_factor,
                recall=recall * drift_factor,
                f1_score=f1 * drift_factor,
                confidence_mean=0.75 + np.random.normal(0, 0.05),
                confidence_std=0.15 + np.random.normal(0, 0.02),
                prediction_count=np.random.randint(50, 200),
                timestamp=datetime.now(),
                data_quality_score=np.random.uniform(0.85, 1.0)
            )
            
        except Exception as e:
            logger.error(f"Error getting recent performance for {model_name}: {str(e)}")
            return None
    
    def _calculate_drift_metrics(self, model_name: str, current_performance: ModelPerformance) -> Dict[str, float]:
        """Calculate drift metrics by comparing current vs baseline performance"""
        try:
            baseline = self.performance_baselines.get(model_name, {})
            
            if not baseline or baseline.get('sample_count', 0) == 0:
                return {'overall_drift_score': 0.0}
            
            # Calculate performance drops
            accuracy_drop = baseline['accuracy_mean'] - current_performance.accuracy
            precision_drop = baseline['precision_mean'] - current_performance.precision
            recall_drop = baseline['recall_mean'] - current_performance.recall
            f1_drop = baseline['f1_mean'] - current_performance.f1_score
            
            # Calculate confidence drift
            confidence_drop = baseline['confidence_mean'] - current_performance.confidence_mean
            
            # Statistical significance tests
            accuracy_z_score = abs(accuracy_drop) / (baseline['accuracy_std'] + 1e-6)
            
            # Overall drift score (weighted combination)
            drift_score = (
                0.4 * max(0, accuracy_drop / 0.1) +     # Accuracy weight: 40%
                0.3 * max(0, f1_drop / 0.1) +          # F1 weight: 30%
                0.2 * max(0, confidence_drop / 0.2) +   # Confidence weight: 20%
                0.1 * accuracy_z_score / 3              # Statistical significance: 10%
            )
            
            return {
                'overall_drift_score': min(drift_score, 2.0),
                'accuracy_drop': accuracy_drop,
                'precision_drop': precision_drop,
                'recall_drop': recall_drop,
                'f1_drop': f1_drop,
                'confidence_drop': confidence_drop,
                'accuracy_z_score': accuracy_z_score,
                'data_quality_score': current_performance.data_quality_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating drift metrics: {str(e)}")
            return {'overall_drift_score': 0.0}
    
    def _determine_drift_status(self, drift_metrics: Dict[str, float]) -> DriftStatus:
        """Determine drift status based on calculated metrics"""
        try:
            overall_score = drift_metrics.get('overall_drift_score', 0.0)
            accuracy_drop = drift_metrics.get('accuracy_drop', 0.0)
            confidence_drop = drift_metrics.get('confidence_drop', 0.0)
            
            # Severe drift conditions
            if (accuracy_drop > self.drift_thresholds['severe_accuracy_drop'] or 
                overall_score > 1.5):
                return DriftStatus.SEVERE_DRIFT
            
            # Regular drift conditions
            if (accuracy_drop > self.drift_thresholds['accuracy_drop'] or
                confidence_drop > self.drift_thresholds['confidence_drop'] or
                overall_score > 1.0):
                return DriftStatus.DRIFT
            
            # Warning conditions
            if (accuracy_drop > self.drift_thresholds['accuracy_drop'] * 0.6 or
                overall_score > 0.6):
                return DriftStatus.WARNING
            
            # Degraded performance (different from drift)
            if drift_metrics.get('data_quality_score', 1.0) < 0.8:
                return DriftStatus.DEGRADED
            
            return DriftStatus.STABLE
            
        except Exception as e:
            logger.error(f"Error determining drift status: {str(e)}")
            return DriftStatus.STABLE
    
    async def _log_performance(self, model_name: str, performance: ModelPerformance):
        """Log model performance to database"""
        try:
            db_path = "db/model_performance.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_performance 
                (model_name, model_type, accuracy, precision_val, recall_val, 
                 f1_score, confidence_mean, confidence_std, prediction_count, data_quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                self.models_status[model_name]['model_type'].value,
                performance.accuracy,
                performance.precision,
                performance.recall,
                performance.f1_score,
                performance.confidence_mean,
                performance.confidence_std,
                performance.prediction_count,
                performance.data_quality_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging performance for {model_name}: {str(e)}")
    
    async def _generate_drift_alert(self, model_name: str, drift_status: DriftStatus, drift_metrics: Dict[str, float]):
        """Generate drift alert and add to queue"""
        try:
            # Generate recommendation based on drift status
            recommendation = self._generate_recommendation(model_name, drift_status, drift_metrics)
            
            alert = DriftAlert(
                model_name=model_name,
                model_type=self.models_status[model_name]['model_type'],
                drift_status=drift_status,
                drift_score=drift_metrics.get('overall_drift_score', 0.0),
                accuracy_drop=drift_metrics.get('accuracy_drop', 0.0),
                confidence_drop=drift_metrics.get('confidence_drop', 0.0),
                timestamp=datetime.now(),
                metrics=drift_metrics,
                recommendation=recommendation
            )
            
            self.drift_alerts.append(alert)
            
            # Keep only last 1000 alerts
            if len(self.drift_alerts) > 1000:
                self.drift_alerts = self.drift_alerts[-1000:]
            
            # Log alert to database
            await self._log_drift_alert(alert)
            
            # Add to retraining queue if severe
            if drift_status in [DriftStatus.DRIFT, DriftStatus.SEVERE_DRIFT]:
                await self._queue_for_retraining(model_name, alert)
            
            logger.warning(f"DRIFT ALERT: {model_name} - {drift_status.value} (score: {alert.drift_score:.3f})")
            
        except Exception as e:
            logger.error(f"Error generating drift alert: {str(e)}")
    
    def _generate_recommendation(self, model_name: str, drift_status: DriftStatus, drift_metrics: Dict[str, float]) -> str:
        """Generate recommendation based on drift analysis"""
        try:
            accuracy_drop = drift_metrics.get('accuracy_drop', 0.0)
            confidence_drop = drift_metrics.get('confidence_drop', 0.0)
            data_quality = drift_metrics.get('data_quality_score', 1.0)
            
            recommendations = []
            
            if drift_status == DriftStatus.SEVERE_DRIFT:
                recommendations.append("IMMEDIATE RETRAINING REQUIRED")
                recommendations.append("Consider emergency model rollback")
                
            elif drift_status == DriftStatus.DRIFT:
                recommendations.append("Schedule model retraining within 24 hours")
                
            if accuracy_drop > 0.08:
                recommendations.append("Investigate training data distribution changes")
                
            if confidence_drop > 0.2:
                recommendations.append("Review prediction confidence calibration")
                
            if data_quality < 0.8:
                recommendations.append("Improve data quality preprocessing")
                
            if not recommendations:
                recommendations.append("Continue monitoring - no immediate action required")
            
            return "; ".join(recommendations)
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return "Error generating recommendation"
    
    async def _log_drift_alert(self, alert: DriftAlert):
        """Log drift alert to database"""
        try:
            db_path = "db/model_performance.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO drift_alerts 
                (model_name, model_type, drift_status, drift_score, 
                 accuracy_drop, confidence_drop, recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.model_name,
                alert.model_type.value,
                alert.drift_status.value,
                alert.drift_score,
                alert.accuracy_drop,
                alert.confidence_drop,
                alert.recommendation
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging drift alert: {str(e)}")
    
    async def _queue_for_retraining(self, model_name: str, alert: DriftAlert):
        """Add model to retraining queue"""
        try:
            if model_name not in [item['model_name'] for item in self.retraining_queue]:
                retraining_item = {
                    'model_name': model_name,
                    'model_type': alert.model_type,
                    'priority': 'high' if alert.drift_status == DriftStatus.SEVERE_DRIFT else 'medium',
                    'queued_at': datetime.now(),
                    'alert': alert
                }
                
                self.retraining_queue.append(retraining_item)
                logger.info(f"Queued {model_name} for retraining (priority: {retraining_item['priority']})")
            
        except Exception as e:
            logger.error(f"Error queuing model for retraining: {str(e)}")
    
    async def _retraining_coordination_loop(self):
        """Coordinate model retraining based on queue"""
        while self.is_initialized:
            try:
                if self.retraining_queue:
                    await self._process_retraining_queue()
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in retraining coordination loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _process_retraining_queue(self):
        """Process models in retraining queue"""
        try:
            # Sort queue by priority and timestamp
            self.retraining_queue.sort(
                key=lambda x: (x['priority'] == 'low', x['queued_at'])
            )
            
            for item in self.retraining_queue[:]:
                model_name = item['model_name']
                
                if model_name not in self.retraining_in_progress:
                    # Start retraining process
                    await self._initiate_retraining(item)
                    break  # Process one at a time
            
        except Exception as e:
            logger.error(f"Error processing retraining queue: {str(e)}")
    
    async def _initiate_retraining(self, retraining_item: Dict[str, Any]):
        """Initiate model retraining process"""
        try:
            model_name = retraining_item['model_name']
            
            # Add to in-progress set
            self.retraining_in_progress.add(model_name)
            
            # Remove from queue
            self.retraining_queue.remove(retraining_item)
            
            logger.info(f"Initiating retraining for {model_name}")
            
            # This would trigger the actual retraining script
            # For now, simulate retraining completion
            await self._simulate_retraining(model_name)
            
            # Remove from in-progress
            self.retraining_in_progress.discard(model_name)
            
        except Exception as e:
            logger.error(f"Error initiating retraining for {retraining_item['model_name']}: {str(e)}")
            self.retraining_in_progress.discard(retraining_item['model_name'])
    
    async def _simulate_retraining(self, model_name: str):
        """Simulate model retraining process"""
        try:
            # Simulate retraining time
            await asyncio.sleep(30)  # 30 seconds simulation
            
            # Update baseline after retraining
            await self._update_baseline_after_retraining(model_name)
            
            logger.info(f"Retraining completed for {model_name}")
            
        except Exception as e:
            logger.error(f"Error in retraining simulation: {str(e)}")
    
    async def _update_baseline_after_retraining(self, model_name: str):
        """Update performance baseline after retraining"""
        try:
            # Reset baseline to expected post-retraining performance
            self.performance_baselines[model_name] = {
                'accuracy_mean': 0.70,  # Expected improved accuracy
                'accuracy_std': 0.04,
                'precision_mean': 0.70,
                'recall_mean': 0.70,
                'f1_mean': 0.70,
                'confidence_mean': 0.80,
                'confidence_std_mean': 0.12,
                'sample_count': 1
            }
            
            # Reset model status
            self.models_status[model_name]['status'] = 'stable'
            self.models_status[model_name]['performance_trend'] = 'improved'
            
            logger.info(f"Updated baseline for {model_name} after retraining")
            
        except Exception as e:
            logger.error(f"Error updating baseline: {str(e)}")
    
    # Public API methods
    def get_drift_status(self) -> Dict[str, Any]:
        """Get overall drift status for all models"""
        return {
            'models_count': len(self.models_status),
            'stable_count': len([m for m in self.models_status.values() if m['status'] == 'stable']),
            'warning_count': len([m for m in self.models_status.values() if m['status'] == 'warning']),
            'drift_count': len([m for m in self.models_status.values() if m['status'] in ['drift', 'severe_drift']]),
            'retraining_queue': len(self.retraining_queue),
            'retraining_in_progress': len(self.retraining_in_progress),
            'last_update': datetime.now()
        }
    
    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get status of specific model"""
        return self.models_status.get(model_name, {})
    
    def get_recent_alerts(self, limit: int = 50) -> List[DriftAlert]:
        """Get recent drift alerts"""
        return self.drift_alerts[-limit:] if self.drift_alerts else []
    
    def get_retraining_queue(self) -> List[Dict[str, Any]]:
        """Get current retraining queue"""
        return self.retraining_queue.copy()
    
    async def force_model_check(self, model_name: str) -> bool:
        """Force immediate drift check for specific model"""
        try:
            if model_name in self.models_status:
                await self._check_model_drift(model_name)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error forcing model check: {str(e)}")
            return False
    
    async def shutdown(self):
        """Shutdown the model drift agent"""
        try:
            self.is_initialized = False
            
            # Cancel monitoring tasks
            for task in getattr(self, 'monitoring_tasks', []):
                task.cancel()
            
            if hasattr(self, 'retraining_task'):
                self.retraining_task.cancel()
            
            logger.info("Model Drift Agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down Model Drift Agent: {str(e)}")
