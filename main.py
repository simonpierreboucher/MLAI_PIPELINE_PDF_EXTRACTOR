import sys
import yaml  # Ensure PyYAML is installed
from pdf_crawler import PDFExtractor  # Ensure pdf_crawler.py is in the same directory or Python path
import logging
from pathlib import Path

def load_config(config_path="config.yaml"):
    """
    Load and parse the YAML configuration file.

    :param config_path: Path to the configuration file.
    :return: Dictionary containing configuration parameters.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"✘ Configuration file {config_path} not found.")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"✘ Error parsing YAML file: {exc}")
        sys.exit(1)

def setup_logging(log_file, log_level):
    """
    Setup logging configuration.

    :param log_file: Path to the log file.
    :param log_level: Logging level as a string.
    :return: Configured logger.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"✘ Invalid log level: {log_level}")
        sys.exit(1)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

def main():
    """
    Main function to initialize and run the PDFExtractor.
    """
    # Load configuration from config.yaml
    config = load_config("config.yaml")

    # Setup logging
    log_file = config.get('logging', {}).get('log_file', 'pdf_crawler.log')
    log_level = config.get('logging', {}).get('log_level', 'INFO')
    logger = setup_logging(log_file, log_level)

    # Verify input directory
    input_dir = Path(config['input_dir'])
    if not input_dir.exists():
        logger.error(f"Le dossier PDF spécifié n'existe pas : {input_dir}")
        sys.exit(1)

    # Verify API keys file
    api_keys_file = config['api_keys_file']
    if not Path(api_keys_file).exists():
        logger.error(f"Le fichier de clés API spécifié n'existe pas : {api_keys_file}")
        sys.exit(1)

    # Initialize the PDFExtractor
    extractor = PDFExtractor(config=config, logger=logger)

    # Start processing
    extractor.process_all_pdfs()

if __name__ == "__main__":
    main()
