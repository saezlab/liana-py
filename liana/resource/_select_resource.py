from pandas import read_csv
from numpy import unique
import pathlib
from pandas import DataFrame
from ._neo4j_controller import Neo4jController


def select_resource(resource_name: str, 
                    cellular_locations_list= None, 
                    tissue_locations_list=None, 
                    biospecimen_locations_list= None, 
                    database_cutoff= None, 
                    experiment_cutoff= None,
                    prediction_cutoff= None,
                    combined_cutoff= None,
                    ) -> DataFrame:

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
        
        resource_path = resource_path = pathlib.Path(__file__).parent.joinpath('MR_ai_cgno.csv') # will be removed, just for testing
        resource = read_csv(resource_path, sep='\t')

        resource = resource[['hmdb_id', 'symbol', 'name']]
        resource = resource.rename(columns={'hmdb_id': 'ligand',
                                            'symbol': 'receptor', 
                                            'name': 'ligand_name'})
        
    elif resource_name == 'neo4j':

        n4j = Neo4jController(
            'bolt://localhost:7687',
            'neo4j',
            '12345678'
        )

        subgraph = n4j.get_subgraph(
            cellular_locations_list,
            tissue_locations_list,
            biospecimen_locations_list,
            database_cutoff,
            experiment_cutoff,
            prediction_cutoff,
            combined_cutoff
        )
        # Prepare the data for the dataframe
        data = []
        for record in subgraph:
            data.append({
                'ligand': record['HMDB'],
                'receptor': record['Symbol'],
                'ligand_name': record['MetName']
                
            })

        # Create the dataframe
        resource = DataFrame(data)

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


def select_metabolite_sets(metsets_name: str = 'metalinksdb') -> DataFrame:
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

    metsets_name = metsets_name.lower()
        
    if metsets_name == 'metalinksdb':    

        resource_path = pathlib.Path(__file__).parent.joinpath('PD_hmdb_recon_cut.csv') # will be removed, just for testing
        metsets = read_csv(resource_path, sep=',')

    elif metsets_name == 'transport':

        resource_path = pathlib.Path(__file__).parent.joinpath('PD_t.csv') # will be removed, just for testing
        metsets = read_csv(resource_path, sep=',')

    return metsets


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
