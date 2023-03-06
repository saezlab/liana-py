from pandas import read_csv
from numpy import unique
import pathlib
from pandas import DataFrame


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

        resource_path = '~/Documents/GitHub/metalinks/metalinksDB/PD_hmdb_recon.csv'
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
