"""
Utility functions to query OmniPath.
Functions to retrieve resources from the meta-database OmniPath.
"""


def check_if_omnipath():
    """
    Function to check if available and return OmniPath
    
    Returns
    -------
    OmniPath package

    """
    try:
        import omnipath as op
    except Exception:
        raise ImportError('omnipath is not installed. Please install it with: pip install omnipath')
    return op