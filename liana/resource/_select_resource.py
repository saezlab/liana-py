from pandas import read_csv
from numpy import unique
import pathlib
from pandas import DataFrame


def select_resource(resource_name: str) -> DataFrame:
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
        
        resource_path = '/home/efarr/Documents/GitHub/metalinks/metalinksDB/MR_ai_cgno.csv'
        resource = read_csv(resource_path, sep='\t')

        resource = resource[['hmdb_id', 'symbol', 'name']]
        resource = resource.rename(columns={'hmdb_id': 'ligand',
                                            'symbol': 'receptor', 
                                            'name': 'ligand_name'})
        
    elif resource_name == 'NCI60':

        resource_path = '/home/efarr/Documents/GitHub/metalinks/metalinksDB/MR_NCI60.csv'
        resource = read_csv(resource_path, sep='\t')

        resource = resource[['hmdb_id', 'symbol']]
        resource = resource.rename(columns={'hmdb_id': 'ligand',
                                            'symbol': 'receptor'})
        
    elif resource_name == 'CCLE':

        resource_path = '/home/efarr/Documents/GitHub/metalinks/metalinksDB/MR_CCLE.csv'
        resource = read_csv(resource_path, sep='\t')

        resource = resource[['hmdb_id', 'symbol']]
        resource = resource.rename(columns={'hmdb_id': 'ligand',
                                            'symbol': 'receptor'})
        
    elif resource_name == 'kidney':

        resource_path = '/home/efarr/Documents/GitHub/metalinks/metalinksDB/MR_500500900_Kidney-pred.csv'
        resource = read_csv(resource_path, sep=',')

        resource = resource[['hmdb_id', 'symbol', 'name']]
        resource = resource.rename(columns={'hmdb_id': 'ligand',
                                            'symbol': 'receptor', 
                                            'name': 'ligand_name'})

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
