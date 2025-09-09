"""
High-Performance Telegram Bot with ML Capabilities

A production-ready Telegram bot built for high concurrency (1000+ users)
with machine learning integration, PostgreSQL database, Redis caching,
and Kubernetes deployment support.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

import logging
import sys
from pathlib import Path

# Add app directory to Python path for absolute imports
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)