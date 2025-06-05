# Buat file setup_logging.py
import logging
import os
from datetime import datetime

def setup_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/clarity_enhancement_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('clarity_enhancement')