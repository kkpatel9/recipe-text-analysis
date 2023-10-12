import sys
import os

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Import the module from the parent directory
from sentiment_module import sentiment

# Try getting sentiments of different terms
print(sentiment.sentiment("heat"))