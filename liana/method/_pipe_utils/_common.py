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