import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Define a custom logging function
def logg(message, level='info', verbose=False):
    """
    Log a message with a specified logging level.

    Args:
        message (str): The message to log.
        level (int or None): The logging level for the message (default is None).
            If None or False, no logging is performed.
    """
    if verbose:
        if level == "warn":
            logging.warning(message)
        elif level == "info":
            logging.info(message)
