"""
Script to run the allennlp model
"""
import logging
import sys
from pathlib import Path
import os

sys.path.append(str(Path().absolute()))
# from allennlp.commands import main
from scibert.training.commands import main

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)


if __name__ == "__main__":
    # Make sure a new predictor is imported in processes/__init__.py
    main(prog="python -m allennlp.run")
