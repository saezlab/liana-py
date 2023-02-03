"""
Utility functions to query OmniPath.
Functions to retrieve resources from the meta-database OmniPath.
"""

from json import loads
import pandas as pd


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


# Function to explode complexes (decomplexify Resource)
def explode_proddeg(resource: pd.DataFrame,
                      GENE='gene',
                      TARGET='receptor') -> pd.DataFrame:
    """
    Function to expand resource to one row per producing/degrading enzyme

    (inspired by MEBOCOST _met_from_enzyme_dataframe_adata_ function)

    Parameters
    ----------
    resource
        Ligand-receptor resource
    SOURCE
        Name of the source (typically ligand) column
    TARGET
        Name of the target (typically receptor) column

    Returns
    -------
    A resource with exploded complexes

    """
    met_gene = []
    for i, line in resource.iterrows():
        genes = line[GENE].split('; ')
        for g in genes:
            tmp = line.copy()
            tmp[GENE] = g
            met_gene.append(tmp)
    resource = pd.DataFrame(met_gene)
    resource['gene_name'] = resource[GENE].apply(lambda x: x.split('[')[0])

    return resource
