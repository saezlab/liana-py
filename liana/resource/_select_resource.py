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


def select_ml_resource(met_est_resource_name: str = 'metalinksdb') -> DataFrame:
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

    met_est_resource_name = met_est_resource_name.lower()
        
    if met_est_resource_name == 'oceandb':

        resource_path = '~/Documents/Database_old/recon3D_full/proddeg_ocean.csv'
        met_est_resource = read_csv(resource_path, sep=',')
        
    elif met_est_resource_name == 'metalinksdb':    

        resource_path = '~/Documents/GitHub/metalinks/metalinksDB/PD_hmdb_recon_cut.csv'
        met_est_resource = read_csv(resource_path, sep=',')

    elif met_est_resource_name == 'nci60':

        resource_path = '~/Documents/GitHub/metalinks/metalinksDB/PD_NCI60.csv'
        met_est_resource = read_csv(resource_path, sep='\t')

    elif met_est_resource_name == 'ccle':

        resource_path = '~/Documents/GitHub/metalinks/metalinksDB/PD_CCLE.csv'
        met_est_resource = read_csv(resource_path, sep='\t')

    elif met_est_resource_name == 'ocean':

        resource_path = '~/Documents/GitHub/metalinks/metalinksDB/PD_OCEAN.csv'
        met_est_resource = read_csv(resource_path, sep='\t')

    elif met_est_resource_name == 'kidney':

        resource_path = '~/Documents/GitHub/metalinks/metalinksDB/PD_Kidney.csv'
        met_est_resource = read_csv(resource_path, sep=',')

    elif met_est_resource_name == 'transport':

        resource_path = '~/Documents/GitHub/metalinks/metalinksDB/PD_t.csv'
        met_est_resource = read_csv(resource_path, sep=',')
        
    
    #resource_path = pathlib.Path(__file__).parent.joinpath("omni_resource.csv")

    return met_est_resource


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
