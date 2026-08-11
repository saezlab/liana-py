import logging
from types import ModuleType

logging.basicConfig(level=logging.INFO, format='%(message)s')


def _logg(message: str, level: str | None = 'info', verbose: bool = False):
    """
    Log a message with a specified logging level.

    Parameters
    ----------
    message
        The message to log.
    level
        The logging level for the message (default is 'info'). Accepted levels
        are 'warn' or 'info', any other value will result in no logging.
    verbose
        Controls whether the message is logged or not.

    """
    if verbose:
        if level == "warn":
            logging.warning(message)
        elif level == "info":
            logging.info(message)


def _check_if_installed(package_name: str,
                        custom_error_message: str = None
                        ) -> ModuleType:
    """
    Checks whether a package is installed in the current environment.

    Parameters
    ----------
    package_name
        The name of the package.
    custom_error_message
        Custom error message to display in case the package could not be found.

    Returns
    -------
    The imported module.

    Raises
    ------
    ImportError
        If the package could not be found/imported.

    """
    try:
        imported_module = __import__(package_name)
        return imported_module
    except ImportError:
        if custom_error_message:
            raise ImportError(custom_error_message) from None
        else:
            raise ImportError(
                    f'{package_name} is not installed. Please install it with: '
                    f'pip install {package_name}'
                ) from None
