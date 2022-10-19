from pandas import read_csv
import pathlib
from pandas import DataFrame


# Read resource from CSV
def select_resource(resource_name: str = 'consensus') -> DataFrame:
    """
    Function to read resource of choice from the pre-generated resources in LIANA.

    Parameters
    ----------
    resource_name
        Name of the resource to be loaded and use for ligand-receptor inference.

    Returns
    -------
    A dataframe with ``['ligand', 'receptor']`` columns

    """

    resource_name = resource_name.lower()

    resource_path = pathlib.Path(__file__).parent.joinpath("omni_resource.csv")
    resource = read_csv(resource_path, index_col=False)

    resource = resource[resource['resource'] == resource_name]

    resource = resource[['source_genesymbol', 'target_genesymbol']]
    resource = resource.rename(columns={'source_genesymbol': 'ligand',
                                        'target_genesymbol': 'receptor'})

    return resource
