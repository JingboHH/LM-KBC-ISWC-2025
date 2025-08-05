#!/usr/bin/env python3
"""
Main execution script for LM-KBC 2025 models.

This script loads configuration, processes input data, and generates predictions
using the specified model implementation.

Usage:
    python main.py -c configs/self_rag_config.yaml -i data/test.jsonl -o results/predictions.jsonl
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import hashlib
import uuid

import yaml
from loguru import logger

from models.user_config import Models


def generate_unique_filename(config_file: str, input_file: str, method: str = "combined") -> str:
    """
    Generate unique filename for output files.
    
    Args:
        config_file: Path to configuration file
        input_file: Path to input file  
        method: Generation method ("timestamp", "uuid", "hash", "combined")
    
    Returns:
        Unique filename string
    """
    config_stem = Path(config_file).stem
    input_stem = Path(input_file).stem
    
    if method == "timestamp":
        # Method 1: Use timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{config_stem}_{timestamp}.jsonl"
    
    elif method == "uuid":
        # Method 2: Use UUID
        unique_id = str(uuid.uuid4())[:8]
        return f"{config_stem}_{unique_id}.jsonl"
    
    elif method == "hash":
        # Method 3: Use hash of input file and config
        content = f"{config_file}_{input_file}_{datetime.now().isoformat()}"
        hash_value = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{config_stem}_{hash_value}.jsonl"
    
    elif method == "combined":
        # Method 4: Combined information (recommended)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{config_stem}_{input_stem}_{timestamp}.jsonl"
    
    else:
        raise ValueError(f"Unknown filename generation method: {method}")


def load_input_data(input_file: str) -> list:
    """Load input data from JSONL file."""
    logger.info(f"Loading input file: {input_file}")
    
    input_rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                input_rows.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num} in {input_file}: {e}")
                raise
    
    logger.info(f"Successfully loaded {len(input_rows):,} input samples")
    return input_rows


def save_predictions(results: list, output_file: str):
    """Save prediction results to JSONL file."""
    logger.info(f"Saving {len(results):,} predictions to: {output_file}")
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    logger.info("Predictions saved successfully!")


def validate_config(config: dict) -> bool:
    """Validate configuration parameters."""
    required_fields = ["model"]
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required configuration field: {field}")
            return False
    
    # Validate model exists
    try:
        Models.get_model(config["model"])
    except Exception as e:
        logger.error(f"Invalid model specified: {config['model']} - {e}")
        return False
    
    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run LM-KBC 2025 Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with Self-RAG model
    python main.py -c configs/self_rag_config.yaml -i data/test.jsonl
    
    # Run with custom output file
    python main.py -c configs/divide_conquer_config.yaml -i data/val.jsonl -o results/my_predictions.jsonl
    
    # Use different naming method for auto-generated filenames
    python main.py -c configs/hybrid_config.yaml -i data/test.jsonl --naming_method timestamp
        """
    )
    
    parser.add_argument(
        "-c", "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "-i", "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file"
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        required=False,
        help="Path to the output JSONL file (auto-generated if not specified)"
    )
    parser.add_argument(
        "--naming_method",
        type=str,
        choices=["timestamp", "uuid", "hash", "combined"],
        default="combined",
        help="Method to generate unique filename when output_file is not specified (default: combined)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    try:
        # Validate input files exist
        if not Path(args.config_file).exists():
            logger.error(f"Configuration file not found: {args.config_file}")
            return 1
        
        if not Path(args.input_file).exists():
            logger.error(f"Input file not found: {args.input_file}")
            return 1
        
        # Load configuration
        logger.info(f"Loading configuration from: {args.config_file}")
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        if not validate_config(config):
            logger.error("Configuration validation failed")
            return 1
        
        logger.info(f"Using model: {config['model']}")
        
        # Determine output file
        output_file = args.output_file
        if not output_file:
            output_dir = Path(__file__).resolve().parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            unique_filename = generate_unique_filename(
                args.config_file, 
                args.input_file, 
                args.naming_method
            )
            output_file = output_dir / unique_filename
            logger.info(f"Auto-generated output filename: {unique_filename}")
        
        # Load input data
        input_rows = load_input_data(args.input_file)
        
        # Initialize model
        logger.info("Initializing model...")
        model_class = Models.get_model(config["model"])
        model = model_class(config)
        
        # Generate predictions
        logger.info("Starting prediction generation...")
        start_time = datetime.now()
        
        results = model.generate_predictions(input_rows)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Prediction generation completed in {duration:.2f} seconds")
        logger.info(f"Average time per sample: {duration/len(input_rows):.3f} seconds")
        
        # Validate results
        if len(results) != len(input_rows):
            logger.warning(f"Result count mismatch: expected {len(input_rows)}, got {len(results)}")
        
        # Save results
        save_predictions(results, str(output_file))
        
        logger.info("Execution completed successfully!")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit(main())