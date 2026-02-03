import sys
import os
import argparse
from .pipeline.data_processor import DataProcessor


def main():
    parser = argparse.ArgumentParser(
        description='Process CSV data using Chain of Responsibility pattern'
    )
    
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to CSV file to process'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='Name of target column'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--list-columns',
        action='store_true',
        help='List all columns in CSV file'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: File '{args.csv_path}' not found")
        return 1
    
    if args.list_columns:
        try:
            df = pd.read_csv(args.csv_path, nrows=1)
            print("Columns in CSV file:")
            for i, col in enumerate(df.columns, 1):
                print(f"{i:3d}. {col}")
            return 0
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")
            return 1
    
    try:
        processor = DataProcessor(
            target_column=args.target,
            output_dir=args.output
        )
        
        context = processor.process(args.csv_path)
        
        print(f"\nOutput files saved to: {os.path.abspath(args.output)}")
        print(f"Final shapes: X={context['X'].shape}, y={context['y'].shape}")
        
        return 0
        
    except Exception as e:
        print(f"\nProcessing failed: {str(e)}")
        print("Try --list-columns to see available columns")
        return 1


if __name__ == "__main__":
    sys.exit(main())
