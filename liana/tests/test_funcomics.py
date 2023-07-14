from liana.resource import select_resource
from liana.funcomics import generate_lr_geneset

import decoupler as dc


def test_generate_lr_resource():
    """Test generate_lr_resource."""
    # load data
    net = dc.get_progeny(top=1000, organism='human') # reduce top to 1000 for testing
    resource = select_resource('consensus')
    
    lr_net = generate_lr_geneset(resource, net)
    
    # check if the result has the right columns
    assert set(lr_net.columns) == set(['interaction', 'weight', 'source'])
    
    # check if the result has the right number of rows
    assert lr_net.shape[0] == 177
    
    # check if the result has the right number of unique interactions
    assert lr_net['interaction'].nunique() == 158
    
    # check if the result has the right number of unique sources
    assert lr_net['source'].nunique() == 14
    
    # check the weight of a specific interaction
    assert lr_net[lr_net['interaction'] == 'LAMB3^ITGAV_ITGB8']['weight'].values[0] == 3.6229854822158813
    
    
def test_generate_nondefault_lr_resource():
    """Test generate_lr_resource."""
    # load data
    net = dc.get_progeny(top=1000, organism='human')
    net.drop(columns=['weight'], inplace=True)
    net.rename(columns={'source': 'tf', 'target': 'genesymbol'}, inplace=True)
    
    resource = select_resource('consensus')
    
    lr_net = generate_lr_geneset(resource, net, source='tf', weight=None, target='genesymbol')
    assert lr_net.shape[0] == 295
    assert max(lr_net['weight']) == 1
