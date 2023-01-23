import numpy as np


def _assign_entity_weights(resource, net, entity_key='receptor', source='source', target='target', weight='weight'):
    # only keep relevant columns
    net = net[[source, target, weight]]
    
    # process ligand-receptor resource
    # assign receptor complex as entity
    entity_resource = resource[[entity_key]].drop_duplicates().set_index(entity_key)
    entity_resource['subunit'] =  entity_resource.index
    # explode complexes, keeping the complex as a key
    entity_resource = entity_resource.apply(lambda x: x.str.split('_')).explode(['subunit'])
    
    # join weights to subunits
    entity_resource = entity_resource.reset_index()
    entity_resource = entity_resource.merge(net, left_on='subunit', right_on=target)
    
    # check for sign and set consistency
    # count expected subunits separated by _
    entity_resource = entity_resource.assign(subunit_expected = entity_resource[entity_key].str.count('_')+1)
    # count subunits by receptor complex & source
    entity_resource['subunit_count'] = entity_resource.groupby([source, entity_key])[[weight]].transform('count')
    # check if all subunits are present
    entity_resource = entity_resource.assign(subunit_complete = lambda x: x['subunit_expected'] == x['subunit_count'])
    # assign flag to sign-coherent subunits
    entity_resource['sing_coherent'] = entity_resource.groupby([source, entity_key])[[weight]].transform(lambda x: np.all(x > 0) | np.all(x < 0))
    
    # keep only relevant targets
    entity_resource = entity_resource[entity_resource['subunit_complete']] # keep only complete complexes
    entity_resource = entity_resource[entity_resource['sing_coherent']] # keep only sign-coherent complexes
    
    # get mean weight per complex & source
    entity_resource = entity_resource.groupby([source, entity_key])[[weight]].mean().reset_index()
    
    return entity_resource


# function that returns the mean only if all values are sign coherent
def _sign_coherent_mean(x):
    if np.all(x > 0) | np.all(x < 0):
        return np.mean(x)
    else:
        return np.nan
    
    
    
def generate_lr_geneset(resource,
                        net, 
                        ligand='ligand',
                        receptor='receptor',
                        lr_separator='&',
                        source='source', 
                        target='target',
                        weight='weight'):
    """
    Generate a ligand-receptor gene set from a resource and a network.
    
    Parameters
    ----------
    resource : pandas.DataFrame
        Ligand-receptor resource.
    net : pandas.DataFrame
        Prior knowledge network in bipartite or decoupler format.
    ligand : str, optional
        Name of the ligand column in the resource, by default 'ligand'
    receptor : str, optional
        Name of the receptor column in the resource, by default 'receptor'
    lr_separator : str, optional
        Separator to use when joining ligand and receptor, by default '&'
    source : str, optional
        Name of the source column in the network, by default 'source'
    target : str, optional 
        Name of the target column in the network, by default 'target'
    weight : str, optional
    
    Returns
    -------
    Returns ligand-receptor geneset resource as a pandas.DataFrame with the following columns:
    - interaction: ligand-receptor interaction
    - weight: mean weight of the interaction
    - source: source of the interaction
    """
    
    # supp keys
    ligand_weight = ligand + '_' + weight
    receptor_weight = receptor + '_' + weight
    ligand_source = ligand + '_' + source
    receptor_source = receptor + '_' + source
    
    
    # assign weights to each entity
    ligand_weights = _assign_entity_weights(resource, net, entity_key=ligand)
    ligand_weights.rename(columns={weight: ligand_weight, source:ligand_source}, inplace=True)
    receptor_weights = _assign_entity_weights(resource, net, entity_key=receptor)
    receptor_weights.rename(columns={weight: receptor_weight, source: receptor_source}, inplace=True)
    
    # join weights to the the ligand-receptor resource
    resource = resource.merge(ligand_weights, on=ligand, how='inner')
    resource = resource.merge(receptor_weights, on=receptor, how='inner')
    
    # keep only coherent ligand and receptor sources
    resource = resource[resource[ligand_source] == resource[receptor_source]]
    # mean of sign-coherent ligand-receptor weights
    resource.loc[:, weight] = resource.apply(lambda x: _sign_coherent_mean(np.array([x[ligand_weight], x[receptor_weight]])), axis=1)
    
    # unite ligand-receptor columns
    resource = resource.assign(interaction = lambda x: x[ligand] + lr_separator + x[receptor])
    
    # keep only relevant columns
    resource = resource[[ligand_source, 'interaction', weight]].rename(columns={ligand_source: source})
    
    # drop nan weights
    resource = resource.dropna()
    
    return resource
