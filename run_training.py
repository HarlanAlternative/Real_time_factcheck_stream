#!/usr/bin/env python3
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

# Import and run the training script
if __name__ == "__main__":
    import fine_tuning.train