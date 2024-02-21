from liana.resource import select_resource
from liana.resource._resource_utils import generate_lr_geneset

import decoupler as dc


def test_generate_lr_resource():
    """Test generate_lr_resource."""
    net = dc.get_progeny(top=1000, organism='human')
    resource = select_resource('consensus')
    lr_net = generate_lr_geneset(resource, net)
    assert set(lr_net.columns) == set(['interaction', 'weight', 'source'])
    assert lr_net.shape[0] == 170
    assert lr_net['interaction'].nunique() == 153
    assert lr_net['source'].nunique() == 14
    assert lr_net[lr_net['interaction'] == 'LAMB3^ITGAV_ITGB8']['weight'].values[0] == 3.6229854822158813


def test_generate_nondefault_lr_resource():
    """Test generate_lr_resource."""
    # load data
    net = dc.get_progeny(top=1000, organism='human')
    net.drop(columns=['weight'], inplace=True)
    net.rename(columns={'source': 'tf', 'target': 'genesymbol'}, inplace=True)

    resource = select_resource('consensus')

    lr_net = generate_lr_geneset(resource, net, source='tf', weight=None, target='genesymbol')
    assert lr_net.shape[0] == 285
    assert 'weight' not in lr_net.columns
