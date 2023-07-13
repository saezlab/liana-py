from pandas import read_csv
from numpy import unique
import pathlib
from pandas import DataFrame
from ._neo4j_controller import Neo4jController


def select_resource(resource_name: str,
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
    
    resource_name = resource_name.lower()

    resource_path = pathlib.Path(__file__).parent.joinpath("omni_resource.csv")
    resource = read_csv(resource_path, index_col=False)

    resource = resource[resource['resource'] == resource_name]

    resource = resource[['source_genesymbol', 'target_genesymbol']]
    resource = resource.rename(columns={'source_genesymbol': 'ligand',
                                        'target_genesymbol': 'receptor'})

    return resource


def select_metalinks() -> DataFrame: # change to the same as select_resource if there are several pre-cooked DBs to merge
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
    resource_path = resource_path = pathlib.Path(__file__).parent.joinpath('MR_ai_cgno.csv') # will be removed, just for testing
    resource = read_csv(resource_path, sep='\t')

    resource = resource[['hmdb_id', 'symbol', 'name']]
    resource = resource.rename(columns={'hmdb_id': 'ligand',
                                        'symbol': 'receptor', 
                                        'name': 'ligand_name'})
    
    return resource


def query_metalinks(cellular_locations: list = ['Extracellular'],
                    tissue_locations: list = ['Adrenal Cortex', 'Brain', 'Epidermis', 'Fibroblasts', 'Kidney', 'Neuron',
                                            'Placenta', 'Skeletal Muscle', 'Spleen', 'Testis', 'Adipose Tissue',
                                            'Intestine' ,'Liver', 'Lung', 'Pancreas', 'Platelet', 'Prostate',
                                            'Thyroid Gland', 'Adrenal Gland' ,'Adrenal Medulla', 'Bladder', 'Heart',
                                            'Leukocyte' ,'Ovary', 'Eye Lens', 'All Tissues', 'Erythrocyte',
                                            'Smooth Muscle', 'Semen', 'Hair' ,'Gall Bladder', 'Retina' ,'Basal Ganglia',
                                            'Blood'], 
                    biospecimen_locations: list = ['Blood', 'Cerebrospinal Fluid (CSF)' ,'Feces' ,'Saliva' ,'Urine',
                                                    'Amniotic Fluid', 'Sweat' ,'Breast Milk', 'Cellular Cytoplasm', 'Bile',
                                                    'Semen', 'Breath'],
                    database_cutoff: int = 200,
                    experiment_cutoff: int = 300,
                    prediction_cutoff: int = 700,
                    combined_cutoff: int = 900,
                    ) -> DataFrame:
    """
    Read resource of choice from the pre-generated resources in LIANA.

    Parameters
    ----------
    resource_name : str
        Name of the resource to be loaded and use for ligand-receptor inference.
    cellular_locations_list : list, optional
        List of cellular locations to be used for filtering the resource, by default None
    tissue_locations_list : list, optional
        List of tissue locations to be used for filtering the resource, by default None
    biospecimen_locations_list : list, optional
        List of biospecimen locations to be used for filtering the resource, by default None
    database_cutoff : float, optional
        Database cutoff to be used for filtering the resource, by default None
    experiment_cutoff : float, optional
        Experiment cutoff to be used for filtering the resource, by default None
    prediction_cutoff : float, optional
        Prediction cutoff to be used for filtering the resource, by default None
    combined_cutoff : float, optional
        Combined cutoff to be used for filtering the resource, by default None      
    
    """

    

    n4j = Neo4jController(
        'bolt://localhost:7687',
        'neo4j',
        '12345678'
    )

    subgraph = n4j.get_subgraph(
        cellular_locations,
        tissue_locations,
        biospecimen_locations,
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
