"""
Model Training Script for hh.ru Salary Prediction
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class HHSalaryModelTrainer:
    """Complete trainer for hh.ru salary prediction model"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize trainer
        
        Args:
            data_path: Path to preprocessed data CSV
        """
        self.data_path = Path(data_path) if data_path else None
        self.df = None
        self.X = None
        self.y = None
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.feature_names = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_data(self, data_path: str = None):
        """
        Load and prepare data
        
        Args:
            data_path: Path to data file (overrides constructor)
        """
        try:
            if data_path:
                self.data_path = Path(data_path)
            
            if not self.data_path.exists():
                # Try to find data in common locations
                possible_paths = [
                    Path("data/processed/preprocessed_data.csv"),
                    Path("data/preprocessed_data.csv"),
                    Path("preprocessed_data.csv"),
                    Path("../data/processed/preprocessed_data.csv")
                ]
                
                for path in possible_paths:
                    if path.exists():
                        self.data_path = path
                        break
                else:
                    raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            logger.info(f"📂 Loading data from: {self.data_path}")
            
            # Load CSV
            self.df = pd.read_csv(self.data_path)
            
            logger.info(f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            logger.info(f"Columns: {list(self.df.columns)}")
            
            # Identify target column
            target_candidates = ['salary', 'compensation', 'Salary', 'Compensation', 'target']
            target_col = None
            
            for candidate in target_candidates:
                if candidate in self.df.columns:
                    target_col = candidate
                    break
            
            if target_col is None:
                # Use last column as target
                target_col = self.df.columns[-1]
                logger.warning(f"Target column not found, using last column: {target_col}")
            
            # Prepare features and target
            self.y = self.df[target_col].values
            self.X = self.df.drop(columns=[target_col]).values
            self.feature_names = list(self.df.drop(columns=[target_col]).columns)
            
            logger.info(f"📊 Data prepared:")
            logger.info(f"   Features: {self.X.shape[1]} ({len(self.feature_names)} names)")
            logger.info(f"   Target: {target_col}")
            logger.info(f"   Samples: {self.X.shape[0]}")
            logger.info(f"   Target stats: min={self.y.min():.2f}, "
                       f"max={self.y.max():.2f}, mean={self.y.mean():.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {str(e)}")
            raise
    
    def train_models(self, test_size: float = 0.2, random_state: int = 42):
        """
        Train multiple models and select the best one
        
        Args:
            test_size: Proportion for test split
            random_state: Random seed
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, 
                test_size=test_size, 
                random_state=random_state,
                shuffle=True
            )
            
            logger.info(f"📈 Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
            
            # Initialize scaler
            self.scaler = RobustScaler()  # More robust to outliers
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Define models to try
            models_to_train = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=0
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=random_state,
                    verbose=0
                ),
                'ridge_regression': Ridge(
                    alpha=1.0,
                    random_state=random_state
                ),
                'linear_regression': LinearRegression()
            }
            
            # Train and evaluate each model
            for name, model in models_to_train.items():
                logger.info(f"🔄 Training {name}...")
                
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                metrics = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred),
                    'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
                    'train_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0],
                    'features': X_train.shape[1]
                }
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train,
                    cv=5, scoring='r2', n_jobs=-1
                )
                metrics['cv_r2_mean'] = cv_scores.mean()
                metrics['cv_r2_std'] = cv_scores.std()
                
                # Store model and metrics
                self.models[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred
                }
                
                logger.info(f"   {name}: R²={metrics['r2']:.4f}, "
                           f"MAE={metrics['mae']:.2f}, "
                           f"RMSE={metrics['rmse']:.2f}")
                
                # Update best model
                if metrics['r2'] > self.best_score:
                    self.best_score = metrics['r2']
                    self.best_model = name
            
            logger.info(f"🏆 Best model: {self.best_model} (R²={self.best_score:.4f})")
            
            return self.models
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            raise
    
    def save_model(self, output_dir: str = "resources", save_all: bool = False):
        """
        Save trained model and artifacts
        
        Args:
            output_dir: Output directory
            save_all: Save all models or just the best one
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save best model
            best_model_data = self.models[self.best_model]
            
            model_data = {
                'model': best_model_data['model'],
                'metrics': best_model_data['metrics'],
                'feature_names': self.feature_names,
                'training_date': self.timestamp,
                'model_type': self.best_model,
                'input_shape': self.X.shape[1] if self.X is not None else None
            }
            
            model_file = output_path / "model_weights.joblib"
            joblib.dump(model_data, model_file, compress=3)
            logger.info(f"💾 Model saved: {model_file}")
            
            # Save scaler
            scaler_data = {
                'feature_scaler': self.scaler,
                'scaler_type': type(self.scaler).__name__
            }
            
            scaler_file = output_path / "scalers.joblib"
            joblib.dump(scaler_data, scaler_file, compress=3)
            logger.info(f"💾 Scaler saved: {scaler_file}")
            
            # Save metrics as JSON
            metrics_file = output_path / "model_metrics.json"
            all_metrics = {
                'best_model': self.best_model,
                'best_r2': self.best_score,
                'models': {}
            }
            
            for name, data in self.models.items():
                all_metrics['models'][name] = data['metrics']
            
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2, default=str)
            
            logger.info(f"💾 Metrics saved: {metrics_file}")
            
            # Save feature importance if available
            if hasattr(best_model_data['model'], 'feature_importances_'):
                importance = best_model_data['model'].feature_importances_
                feature_importance = dict(zip(self.feature_names, importance))
                
                # Sort by importance
                sorted_importance = dict(
                    sorted(feature_importance.items(), 
                          key=lambda x: x[1], reverse=True)[:20]
                )
                
                importance_file = output_path / "feature_importance.json"
                with open(importance_file, 'w') as f:
                    json.dump(sorted_importance, f, indent=2)
                
                logger.info(f"💾 Feature importance saved: {importance_file}")
            
            # Save all models if requested
            if save_all:
                all_models_file = output_path / "all_models.joblib"
                joblib.dump(self.models, all_models_file, compress=3)
                logger.info(f"💾 All models saved: {all_models_file}")
            
            logger.info(f"✅ All artifacts saved to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {str(e)}")
            raise
    
    def generate_sample_data(self, n_samples: int = 100, n_features: int = 20):
        """
        Generate sample data for testing if no real data is available
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
        """
        logger.info("🔧 Generating sample data...")
        
        np.random.seed(42)
        
        # Generate realistic features
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Realistic feature generation
        X = np.random.randn(n_samples, n_features)
        
        # Create realistic salary (50k-200k range)
        base_salary = 50000
        coefs = np.random.uniform(1000, 10000, n_features)
        
        y = base_salary + np.dot(X, coefs) + np.random.randn(n_samples) * 15000
        y = np.maximum(y, 30000)  # Minimum salary
        
        # Create DataFrame
        self.df = pd.DataFrame(X, columns=feature_names)
        self.df['salary'] = y
        
        self.X = X
        self.y = y
        self.feature_names = feature_names
        
        # Save sample data
        sample_path = Path("data/processed/preprocessed_data.csv")
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(sample_path, index=False)
        
        logger.info(f"✅ Sample data generated: {sample_path}")
        logger.info(f"   Samples: {n_samples}, Features: {n_features}")
        logger.info(f"   Salary range: {y.min():.0f} - {y.max():.0f} RUB")
        
        return sample_path

def print_banner():
    """Print training banner"""
    print("\n" + "="*70)
    print("HH.RU SALARY PREDICTION MODEL TRAINING")
    print("="*70)

def main():
    """Main training function"""
    print_banner()
    
    try:
        # Initialize trainer
        trainer = HHSalaryModelTrainer()
        
        # Try to load data or generate sample
        try:
            trainer.load_data()
        except FileNotFoundError:
            logger.warning("No data file found, generating sample data...")
            trainer.generate_sample_data(n_samples=1000, n_features=15)
        
        # Train models
        models = trainer.train_models(test_size=0.2, random_state=42)
        
        # Save model
        trainer.save_model(output_dir="resources", save_all=False)
        
        # Print summary
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        
        best_metrics = models[trainer.best_model]['metrics']
        
        print(f"\n🏆 Best Model: {trainer.best_model.upper()}")
        print(f"   R² Score:  {best_metrics['r2']:.4f}")
        print(f"   MAE:       {best_metrics['mae']:.2f} RUB")
        print(f"   RMSE:      {best_metrics['rmse']:.2f} RUB")
        print(f"   MAPE:      {best_metrics['mape']:.2f}%")
        
        print(f"\n📊 Dataset:")
        print(f"   Total samples: {trainer.X.shape[0]}")
        print(f"   Features:      {trainer.X.shape[1]}")
        print(f"   Train samples: {best_metrics['train_samples']}")
        print(f"   Test samples:  {best_metrics['test_samples']}")
        
        print(f"\n💾 Saved to: resources/")
        print(f"   • model_weights.joblib")
        print(f"   • scalers.joblib")
        print(f"   • model_metrics.json")
        print("="*70 + "\n")
        
        # Create a test .npy file for immediate testing
        test_data = np.random.randn(5, trainer.X.shape[1])
        test_file = Path("test_example.npy")
        np.save(test_file, test_data)
        
        print(f"✅ Test file created: {test_file}")
        print(f"   To test: python app.py {test_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
