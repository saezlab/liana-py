from pandas import read_csv
from numpy import unique
import pathlib
from pandas import DataFrame


def select_ml_resource(resource_name: str = 'consensus') -> DataFrame:
    """
    Read resource of choice from the pre-generated resources in LIANA.

    Parameters
    ----------
    resource_name
        Name of the resource to be loaded and use for ligand-receptor inference.

    Returns
    -------
    A dataframe with ``['ligand', 'receptor']`` columns

    """

    resource_name = resource_name.lower()

    #resource_path = pathlib.Path(__file__).parent.joinpath("omni_resource.csv")
    resource_path = '/home/efarr/MEBOCOST/MEBOCOST/data/mebocost_db/human/metabolite_associated_gene_reaction_HMDB_summary.tsv'
    resource = read_csv(resource_path, sep='\t')

    
    return resource


def show_ml_resources():
    """
    Show provided resources.

    Returns
    -------
    A list of resource names available via ``liana.resource.select_resource``

    """
    resource_path = pathlib.Path(__file__).parent.joinpath("omni_resource.csv")
    resource = read_csv(resource_path, index_col=False)
    return list(unique(resource.resource))
