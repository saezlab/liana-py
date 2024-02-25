import numpy as np
from liana._constants import PrimaryColumns as P, DefaultValues as V
from liana._docs import d

@d.dedent
def generate_lr_geneset(resource,
                        net,
                        ligand_key=P.ligand,
                        receptor_key=P.receptor,
                        lr_sep=V.lr_sep,
                        source='source',
                        target='target',
                        weight='weight'
                        ):
    """
    Generate a ligand-receptor gene set from a resource and a network.

    Specifically, it works with weighted bipartite networks, where the weight represents the importance of the genes
    to a given geneset. The function will assign a weight to each ligand-receptor interaction, based on the mean.
    It does so by first assigning a weight to each ligand-receptor subunit, checking for sign coherence and completeness
    of the ligand-receptor complex.

    Parameters
    ----------
    resource:
        A pandas dataframe with [`ligand`, `receptor`] columns.
    net
        Prior knowledge network in bipartite or decoupler format.
    ligand : str, optional
        Name of the ligand column in the resource
    receptor : str, optional
        Name of the receptor column in the resource
    %(lr_sep)s
    source : str, optional
        Name of the source column in the network.
    weight : str, optional
        Name of the weight column in the network. If None, all weights are set to 1.

    Returns
    -------
    Returns ligand-receptor geneset resource as a pandas.DataFrame with the following columns:
    - interaction: ligand-receptor interaction
    - weight: mean weight of the interaction
    - source: source of the interaction
    """
    # TODO: Fix this if else, it's not very elegant
    if weight is None:
        weight = 'weight'
        net[weight] = 1

        drop_weight = True
    else:
        drop_weight = False

    # supp keys
    ligand_weight = ligand_key + '_' + weight
    receptor_weight = receptor_key + '_' + weight
    ligand_source = ligand_key + '_' + source
    receptor_source = receptor_key + '_' + source

    # assign weights to each entity
    ligand_weights = _assign_entity_weights(resource, net, source=source, target=target, entity_key=ligand_key)
    ligand_weights.rename(columns={weight: ligand_weight, source:ligand_source}, inplace=True)
    receptor_weights = _assign_entity_weights(resource, net, source=source, target=target, entity_key=receptor_key)
    receptor_weights.rename(columns={weight: receptor_weight, source: receptor_source}, inplace=True)

    # join weights to the the ligand-receptor resource
    resource = resource.merge(ligand_weights, on=ligand_key, how='inner')
    resource = resource.merge(receptor_weights, on=receptor_key, how='inner')

    # keep only coherent ligand and receptor sources
    resource = resource[resource[ligand_source] == resource[receptor_source]]
    # mean of sign-coherent ligand-receptor weights
    resource.loc[:, weight] = resource.apply(lambda x: _sign_coherent_mean(np.array([x[ligand_weight], x[receptor_weight]])), axis=1)

    # unite ligand-receptor columns
    resource = resource.assign(interaction = lambda x: x[ligand_key] + lr_sep + x[receptor_key])

    # keep only relevant columns
    resource = resource[[ligand_source, 'interaction', weight]].rename(columns={ligand_source: source})

    # drop nan weights
    resource = resource.dropna()

    if drop_weight:
        resource.drop(columns=['weight'], inplace=True)

    return resource

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

def _sign_coherent_mean(x):
    if np.all(x > 0) | np.all(x < 0):
        return np.mean(x)
    else:
        return np.nan
