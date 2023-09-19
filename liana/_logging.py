import logging
from types import ModuleType

logging.basicConfig(level=logging.INFO, format='%(message)s')

def _logg(message, level='info', verbose=False):
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

def _check_if_installed(package_name: str, custom_error_message: str = None) -> ModuleType:
    try:
        imported_module = __import__(package_name)
        return imported_module
    except ImportError:
        if custom_error_message:
            raise ImportError(custom_error_message)
        else:
            raise ImportError(f'{package_name} is not installed. Please install it with: pip install {package_name}')
