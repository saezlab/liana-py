from pandas import read_csv
from numpy import unique
import pathlib
from pandas import DataFrame

from liana._logging import _logg
from liana._constants import DefaultValues as V

def select_resource(resource_name: str = V.resource_name) -> DataFrame:
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

    if resource_name not in resource['resource'].unique():
        raise ValueError(f"Resource {resource_name} not found. "
                         f"Please choose from {resource['resource'].unique()}")

    resource = resource[resource['resource'] == resource_name]

    resource = resource[['source_genesymbol', 'target_genesymbol']]
    resource = resource.rename(columns={'source_genesymbol': 'ligand',
                                        'target_genesymbol': 'receptor'}
                               )

    return resource


def show_resources():
    """
    Show available resources.

    Returns
    -------
    A list of resource names available via ``liana.resource.select_resource``

    """
    resource_path = pathlib.Path(__file__).parent.joinpath("omni_resource.csv")
    resource = read_csv(resource_path, index_col=False)
    return list(unique(resource['resource']))


def _handle_resource(interactions=None, resource=None, resource_name=None, x_name='ligand', y_name='receptor', verbose=True):
    if interactions is None:
        if resource is None:
            if resource_name is None:
                raise ValueError("If 'interactions' and 'resource' are both None, 'resource_name' must be provided.")
            else:
                _logg(f"Using resource `{resource_name}`.", verbose=verbose)
                resource = select_resource(resource_name)
        else:
            if verbose:
                print("Using provided `resource`.")
            if not isinstance(resource, DataFrame) or x_name not in resource.columns or y_name not in resource.columns:
                raise ValueError("If 'interactions' is None, 'resource' must be a valid DataFrame "
                                 "with columns '{}' and '{}'.".format(x_name, y_name))
            resource = resource.copy()
    else:
        _logg("Using provided `interactions`.", verbose=verbose)
        if not isinstance(interactions, list) or any(len(item) != 2 for item in interactions):
            raise ValueError("'interactions' should be a list of tuples in the format [(x1, y1), (x2, y2), ...].")
        resource = DataFrame(interactions, columns=[x_name, y_name])

    return resource
