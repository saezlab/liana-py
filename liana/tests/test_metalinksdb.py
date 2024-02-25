import pandas as pd
from liana.resource.get_metalinks import get_metalinks, get_metalinks_values, describe_metalinks
import pathlib

# TODO: replace with link, None should use the link, optionally use path
import os
filepath = pathlib.Path(__file__).parent.absolute()
os.chdir(filepath)

def test_get_metalinks():
    # set path to here
    result = get_metalinks(tissue_location='Brain',
                           hmdb_ids='HMDB0000073',
                           uniprot_ids='P14416'
                           )
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 4)
    assert 'Dopamine' in result['metabolite'].values
    assert 'HMDB0000073' in result['hmdb'].values
    assert 'P14416' in result['uniprot'].values
    assert 'DRD2' in result['gene_symbol'].values


def test_get_metalinks_values():
    # Call the function with a test db_path, table_name, and column_name
    result = get_metalinks_values('disease', 'disease')

    # Check that the result is a list
    assert isinstance(result, list)

    assert 'Obesity' in result
    assert 'Schizophrenia' in result
    assert len(result) == 530


def test_describe_metalinks():
    out = describe_metalinks(return_output=True)
    assert 'metabolites' in out
    assert 'proteins' in out
    assert 'Primary Key: 1' in out
