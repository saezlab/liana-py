from liana._constants import PrimaryColumns as P, DefaultValues as V
from numpy import union1d


def _join_stats(source, target, dedict, resource):
    """
    Joins and renames source-ligand and target-receptor stats to the ligand-receptor resource

    Parameters
    ----------
    source
        Source/Sender cell type
    target
        Target/Receiver cell type
    dedict
        dictionary
    resource
        Ligand-receptor Resource

    Returns
    -------
    Ligand-Receptor stats

    """
    source_stats = dedict[source].copy()
    source_stats.columns = source_stats.columns.map(
        lambda x: 'ligand_' + str(x))
    source_stats = source_stats.rename(
        columns={'ligand_names': 'ligand', 'ligand_label': 'source'})

    target_stats = dedict[target].copy()
    target_stats.columns = target_stats.columns.map(
        lambda x: 'receptor_' + str(x))
    target_stats = target_stats.rename(
        columns={'receptor_names': 'receptor', 'receptor_label': 'target'})

    bound = resource.merge(source_stats).merge(target_stats)

    return bound

def _get_props(X_mask):
    return X_mask.getnnz(axis=0) / X_mask.shape[0]

def _get_groupby_subset(groupby_pairs):
    if groupby_pairs is not V.groupby_pairs:
        if not (P.source in groupby_pairs.columns) | (P.target in groupby_pairs.columns):
            raise AssertionError(f"{P.source} and {P.target} must be in groupby_pairs")
        groupby_subset = union1d(groupby_pairs[P.source].unique(),
                                 groupby_pairs[P.target].unique())

    else:
        groupby_subset = None
    return groupby_subset
