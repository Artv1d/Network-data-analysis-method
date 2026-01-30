"""
Salary Prediction App for hh.ru data
Usage: python app.py path/to/x_data.npy
"""

import sys
import os
import numpy as np
import joblib
from pathlib import Path
import logging
import traceback
from typing import List, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SalaryPredictionApp:
    """Main application class for salary prediction"""
    
    VERSION = "1.0.0"
    
    def __init__(self, model_dir: str = "resources"):
        """
        Initialize the prediction application
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.model_metrics = None
        self._load_resources()
    
    def _load_resources(self) -> None:
        """Load model, scaler, and metrics from resources directory"""
        try:
            # Check if resources directory exists
            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"Resources directory not found: {self.model_dir}. "
                    f"Please run model_training.py first."
                )
            
            # Load model weights
            model_path = self.model_dir / "model_weights.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model_data = joblib.load(model_path)
            self.model = model_data.get('model')
            if self.model is None:
                raise ValueError("Model not found in model_weights.joblib")
            
            # Load scaler
            scaler_path = self.model_dir / "scalers.joblib"
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            scaler_data = joblib.load(scaler_path)
            self.scaler = scaler_data.get('feature_scaler')
            if self.scaler is None:
                raise ValueError("Scaler not found in scalers.joblib")
            
            # Load metrics
            metrics_path = self.model_dir / "model_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.model_metrics = json.load(f)
            
            logger.info(f"✅ Model loaded successfully (type: {type(self.model).__name__})")
            logger.info(f"✅ Scaler loaded successfully")
            if self.model_metrics:
                logger.info(f"✅ Model metrics: R²={self.model_metrics.get('r2', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load resources: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data format
        
        Args:
            data: Input numpy array
            
        Returns:
            bool: True if valid
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Input must be numpy.ndarray, got {type(data)}")
        
        if len(data.shape) != 2:
            raise ValueError(
                f"Input must be 2D array (samples × features), got shape {data.shape}"
            )
        
        # Check if dimensions match model expectations
        if hasattr(self.model, 'n_features_in_'):
            expected_features = self.model.n_features_in_
            if data.shape[1] != expected_features:
                raise ValueError(
                    f"Model expects {expected_features} features, "
                    f"got {data.shape[1]}"
                )
        
        # Check for NaN or Inf values
        if np.any(np.isnan(data)):
            raise ValueError("Input contains NaN values")
        
        if np.any(np.isinf(data)):
            raise ValueError("Input contains infinite values")
        
        return True
    
    def predict(self, input_data: np.ndarray) -> List[float]:
        """
        Make salary predictions
        
        Args:
            input_data: Input features array
            
        Returns:
            List of predicted salaries in RUB
        """
        try:
            # Validate input
            self.validate_input(input_data)
            
            logger.info(f"📊 Processing {input_data.shape[0]} samples "
                       f"with {input_data.shape[1]} features")
            
            # Scale features
            X_scaled = self.scaler.transform(input_data)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            # Ensure positive salaries (optional)
            predictions = np.maximum(predictions, 30000)  # Minimum reasonable salary
            
            # Convert to list of floats
            result = [float(p) for p in predictions]
            
            # Log statistics
            pred_array = np.array(result)
            logger.info(f"📈 Prediction statistics:")
            logger.info(f"   Min: {pred_array.min():,.2f} RUB")
            logger.info(f"   Max: {pred_array.max():,.2f} RUB")
            logger.info(f"   Mean: {pred_array.mean():,.2f} RUB")
            logger.info(f"   Std: {pred_array.std():,.2f} RUB")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    def run(self, input_path: str) -> List[float]:
        """
        Main execution method
        
        Args:
            input_path: Path to input .npy file
            
        Returns:
            List of predictions
        """
        try:
            input_file = Path(input_path)
            
            # Validate input file
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            if input_file.suffix != '.npy':
                logger.warning(f"File extension is {input_file.suffix}, expected .npy")
            
            # Load input data
            logger.info(f"📂 Loading data from: {input_file}")
            input_data = np.load(input_file, allow_pickle=False)
            
            # Make predictions
            predictions = self.predict(input_data)
            
            # Save predictions to file
            output_file = input_file.parent / f"predictions_{input_file.stem}.npy"
            np.save(output_file, predictions)
            logger.info(f"💾 Predictions saved to: {output_file}")
            
            # Save as CSV for easier viewing
            csv_file = input_file.parent / f"predictions_{input_file.stem}.csv"
            np.savetxt(csv_file, predictions, delimiter=',', fmt='%.2f', 
                      header='predicted_salary_rub')
            logger.info(f"💾 CSV version saved to: {csv_file}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Application error: {str(e)}")
            sys.exit(1)

def print_usage():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("SALARY PREDICTION APP")
    print("="*60)
    print("Usage: python app.py <path_to_npy_file>")
    print("\nExamples:")
    print("  python app.py data/test_features.npy")
    print("  python app.py /path/to/your/x_data.npy")
    print("\nRequirements:")
    print("  • Input: .npy file with 2D array (samples × features)")
    print("  • Model files in 'resources/' directory")
    print("="*60 + "\n")

def main():
    """Main entry point"""
    # Check command line arguments
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    
    # Check for help flag
    if sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)
    
    # Check for version flag
    if sys.argv[1] in ['-v', '--version']:
        print(f"Salary Prediction App v{SalaryPredictionApp.VERSION}")
        sys.exit(0)
    
    # Run the application
    try:
        app = SalaryPredictionApp()
        predictions = app.run(sys.argv[1])
        
        # Print results
        print("\n" + "="*60)
        print(f"PREDICTION RESULTS ({len(predictions)} salaries)")
        print("="*60)
        
        # Show first 5 predictions
        for i, salary in enumerate(predictions[:5], 1):
            print(f"{i:3d}. {salary:12,.2f} RUB")
        
        if len(predictions) > 5:
            print(f"... and {len(predictions) - 5} more predictions")
        
        # Show summary
        avg_salary = np.mean(predictions)
        print(f"\n📊 Summary:")
        print(f"   Total predictions: {len(predictions)}")
        print(f"   Average salary: {avg_salary:,.2f} RUB")
        print(f"   Total monthly payroll: {avg_salary * len(predictions):,.2f} RUB")
        print("="*60)
        
        return predictions
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
