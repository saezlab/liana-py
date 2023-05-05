import numpy as np
from pandas import DataFrame, Index

# TODO merge

def filter_ml_resource(resource: DataFrame, met_ids: Index, var_names: Index) -> DataFrame:
    """
    Filter interactions for which vars are not present.

    Note that here I remove any interaction that /w genes that are not found
    in the dataset. Note that this is not necessarily the case in liana-r.
    There, I assign the expression of those with missing subunits to 0, while
    those without any subunit present are implicitly filtered.

    Parameters
    ---------
    resource
        Resource with 'ligand' and 'receptor' columns
    met_ids
        Relevant metabolites - i.e. the metabolites that were estimated by the met_est method
    var_names
        Relevant variables - i.e. the variables to be used downstream

    Returns
    ------
    A filtered resource dataframe
    """
    
    # Remove those without any subunit
    resource = resource[(np.isin(resource.ligand, met_ids)) & (np.isin(resource.receptor, var_names))]
 
    return resource