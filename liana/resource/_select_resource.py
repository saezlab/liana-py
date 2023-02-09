from pandas import read_csv
from numpy import unique
import pathlib
from pandas import DataFrame


def select_resource(resource_name: str = 'metalinksdb') -> DataFrame:
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
    if resource_name == 'metalinksdb':
        
        resource_path = '/home/efarr/Documents/Startover/interactions_small.csv'
        resource = read_csv(resource_path, sep=',')

        resource = resource[['HMDB', 'symbol']]
        resource = resource.rename(columns={'HMDB': 'ligand',
                                            'symbol': 'receptor'})

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
