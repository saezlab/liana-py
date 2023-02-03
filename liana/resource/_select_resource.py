from pandas import read_csv
from numpy import unique
import pathlib
from pandas import DataFrame


def select_resource(resource_name: str = 'consensus') -> DataFrame:
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

    if resource_name == 'mebocost':

        resource_path = '/home/efarr/MEBOCOST/MEBOCOST/data/mebocost_db/human/met_sen_October-25-2022_14-52-47.tsv'
        resource = read_csv(resource_path, sep='\t')
    
        #resource = resource[resource['resource'] == resource_name]

        resource = resource[['HMDB_ID', 'Gene_name']]
        resource = resource.rename(columns={'HMDB_ID': 'ligand',
                                            'Gene_name': 'receptor'})

    else:

        resource_name = resource_name.lower()

        resource_path = pathlib.Path(__file__).parent.joinpath("omni_resource.csv")
        resource = read_csv(resource_path, index_col=False)

        resource = resource[resource['resource'] == resource_name]

        resource = resource[['source_genesymbol', 'target_genesymbol']]
        resource = resource.rename(columns={'source_genesymbol': 'ligand',
                                            'target_genesymbol': 'receptor'})

    
    return resource


def show_resources():
    """
    Show provided resources.

    Returns
    -------
    A list of resource names available via ``liana.resource.select_resource``

    """
    resource_path = pathlib.Path(__file__).parent.joinpath("omni_resource.csv")
    resource = read_csv(resource_path, index_col=False)
    return list(unique(resource.resource))
