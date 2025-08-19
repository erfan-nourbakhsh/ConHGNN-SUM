import logging
import sys

# Create a logger named "Summarization logger"
logger = logging.getLogger("Summarization logger")

# Define the log message format with timestamp, log level, and message
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

# Create a console handler that outputs logs to stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter       # Set formatter for the console output
console_handler.setLevel(logging.INFO)     # Only output INFO level and above to console
logger.addHandler(console_handler)         # Add the console handler to the logger

# Set the logger to capture all DEBUG and higher level messages
logger.setLevel(logging.DEBUG)


def log_score(scores_all):
    """
    Logs ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) in a readable format.

    :param scores_all: dict containing ROUGE scores in the format:
        {
            'rouge-1': {'p': precision, 'r': recall, 'f': f1},
            'rouge-2': {...},
            'rouge-l': {...}
        }
    """
    # Prepare a formatted string for all ROUGE scores
    res = f"""
        ROUGE_1:
            p={scores_all['rouge-1']['p']}
            r={scores_all['rouge-1']['r']}
            f={scores_all['rouge-1']['f']}
        ROUGE_2:
            p={scores_all['rouge-2']['p']}
            r={scores_all['rouge-2']['r']}
            f={scores_all['rouge-2']['f']}
        ROUGE_L:
            p={scores_all['rouge-l']['p']}
            r={scores_all['rouge-l']['r']}
            f={scores_all['rouge-l']['f']}
    """
    # Log the formatted ROUGE scores at INFO level
    logger.info(res)
