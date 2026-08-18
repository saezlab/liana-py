import decoupler as dc
import numpy as np

from liana.resource import select_resource
from liana.resource._resource_utils import generate_lr_geneset


def test_generate_lr_resource():
    """Test generate_lr_resource."""
    # load data
    net = dc.op.progeny(top=1000, organism='human', thr_padj=1)
    resource = select_resource('consensus')
    lr_net = generate_lr_geneset(resource, net)
    assert set(lr_net.columns) == {'interaction', 'weight', 'source'}
    assert lr_net.shape[0] == 170
    assert lr_net['interaction'].nunique() == 153
    assert lr_net['source'].nunique() == 14
    assert np.isclose(lr_net[lr_net['interaction'] == 'LAMB3^ITGAV_ITGB8']['weight'].values[0], 3.62299, atol=1e-5)


def test_generate_nondefault_lr_resource():
    """Test generate_lr_resource."""
    # load data
    net = dc.op.progeny(top=1000, organism='human')
    net.drop(columns=['weight'], inplace=True)
    net.rename(columns={'source': 'tf', 'target': 'genesymbol'}, inplace=True)

    resource = select_resource('consensus')

    lr_net = generate_lr_geneset(resource, net, source='tf', weight=None, target='genesymbol')
    assert lr_net.shape[0] == 250
    assert 'weight' not in lr_net.columns
