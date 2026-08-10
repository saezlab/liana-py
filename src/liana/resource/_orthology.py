import os
import urllib.request
from itertools import product

import numpy as np
import pandas as pd

from liana._logging import _logg

_HCOP_BASE = "https://storage.googleapis.com/public-download-files/hcop"


def _replace_subunits(lst, my_dict, one_to_many):
    result = []
    for x in lst:
        if x in my_dict:
            value = my_dict[x]

            if not isinstance(value, list):
                value = [value]

            if len(value) > one_to_many:
                result.append(np.nan)
            else:
                result.append(value)
        else:
            result.append(np.nan)
    return result


def _generate_orthologs(data, column, map_dict, one_to_many):
    df = data[[column]].drop_duplicates().set_index(column)

    df["subunits"] = df.index.str.split("_")
    df["subunits"] = df["subunits"].apply(
        _replace_subunits,
        args=(
            map_dict,
            one_to_many,
        ),
    )
    df = df["subunits"].explode().reset_index()

    grouped = (
        df.groupby(column).filter(lambda x: x["subunits"].notna().all()).groupby(column)
    )

    # Generate all possible subunit combinations within each group
    complexes = []
    for name, group in grouped:
        if group["subunits"].isnull().all():
            continue
        subunit_lists = [list(x) for x in group["subunits"]]
        complex_combinations = list(product(*subunit_lists))
        for complex in complex_combinations:
            complexes.append((name, "_".join(complex)))

    # Create output DataFrame
    col_names = ["orthology_source", "orthology_target"]
    result = pd.DataFrame(complexes, columns=col_names).set_index("orthology_source")

    return result


def translate_column(
    resource: pd.DataFrame,
    map_df: pd.DataFrame,
    column: str,
    replace: bool = True,
    one_to_many: int = 1,
    ) -> pd.DataFrame:
    """
    Generate orthologs for a given column in a DataFrame.

    Parameters
    ----------
    resource
        Input DataFrame.
    map_df
        DataFrame with orthology mappings, where the first column is the source and the second column is the target for mapping.
    column
        Column name to translate.
    replace
        Whether to replace the original column with the translated values. Default is True.
        If False, it will create a new column with the prefix "orthology_".
    one_to_many
        Maximum number of orthologs allowed per gene. Default is 1.

    Details
    -------
    This function generates orthologs for a given column in a DataFrame.
    It handles complex names by splitting them into subunits and generating all possible combinations of orthologs.
    It assumes that subunits are separated by an underscore ("_").

    Returns
    -------
    Resulting DataFrame with translated column.

    Raises
    ------
    ValueError
        If the `mapping_df` does not contain 'source' and 'target' columns or `one_to_many` is not an integer

    """
    if not isinstance(one_to_many, int):
        raise ValueError("`one_to_many` should be a positive integer!")
    if ['source', 'target'] != map_df.columns.tolist():
        raise ValueError("The `map_df` DataFrame must have two columns named 'source' and 'target'!")

    # get orthologs
    map_df = map_df.set_index("source")
    map_dict = map_df.groupby(level=0)["target"].apply(list).to_dict()
    map_data = _generate_orthologs(resource, column, map_dict, one_to_many)

    # join orthologs
    resource = resource.merge(map_data,
                              left_on=column,
                              right_index=True,
                              how="left")

    # replace orthologs
    if replace:
        resource[column] = resource["orthology_target"]
    else:
        resource[f"orthology_{column}"] = resource.apply(
            lambda x: x["orthology_target"]
            if not pd.isnull(x["orthology_target"])
            else x[column],
            axis=1,
        )
    resource = resource.drop(columns=["orthology_target"])

    resource = resource.dropna(subset=[column])
    return resource


# function that loops over columns and applies translate_column
def translate_resource(
        resource: pd.DataFrame,
        map_df: pd.DataFrame,
        columns: list[str] = None,
        **kwargs
        ) -> pd.DataFrame:
    """
    Generate orthologs for multiple columns in a DataFrame.

    Parameters
    ----------
    resource
        Input DataFrame.
    map_df
        DataFrame with orthology mappings, where the first column is the source and the second column is the target for mapping.
    columns
        List of column names to translate.
    **kwargs
        Additional arguments for `liana.utils.translate_column`.

    Returns
    -------
    Resulting DataFrame with translated columns.

    """
    if columns is None:
        columns = ['ligand', 'receptor']

    for column in columns:
        resource = translate_column(resource, map_df, column, **kwargs)

    return resource


def get_hcop_orthologs(target_organism="mouse",
                       url=None,
                       filename=None,
                       min_evidence=3,
                       columns=None
                       ):
    """
    Download the HCOP orthology file and filter it by minimum evidence.

    Parameters
    ----------
    target_organism : str
        Target organism for orthology mapping. Default is ``"mouse"``.
        Supported values: ``anole_lizard``, ``c.elegans``, ``cat``, ``cattle``,
        ``chicken``, ``chimpanzee``, ``dog``, ``fruitfly``, ``horse``, ``macaque``,
        ``mouse``, ``opossum``, ``pig``, ``platypus``, ``rat``, ``s.cerevisiae``,
        ``s.pombe``, ``xenopus``, ``zebrafish``.
        The target-organism column in the returned DataFrame follows the pattern
        ``{target_organism}_symbol`` (e.g. ``mouse_symbol``, ``rat_symbol``).
    url : str, optional
        Override the download URL. If ``None`` (default), the URL is constructed
        from ``target_organism`` using the HGNC Google Cloud Storage bucket.
    filename : str, optional
        Local filename to save the downloaded file. Derived from the URL if ``None``.
    min_evidence : int
        Minimum number of orthology resources that must support an interaction.
    columns : list, optional
        Columns to keep in the final DataFrame. If ``None``, all columns are kept.

    Returns
    -------
    mapping
        DataFrame with the HCOP mapping.

    Details
    -------
    HCOP is a composite database combining data from various orthology resources.
    It provides a comprehensive set of human orthologs across many species.

    If you use this function, please reference the original HCOP papers:
    - Eyre, T.A., Wright, M.W., Lush, M.J. and Bruford, E.A., 2007. HCOP: a searchable database of human orthology predictions. Briefings in bioinformatics, 8(1), pp.2-5.
    - Yates, B., Gray, K.A., Jones, T.E. and Bruford, E.A., 2021. Updates to HCOP: the HGNC comparison of orthology predictions tool. Briefings in Bioinformatics, 22(6), p.bbab155.

    For more information, please visit the HCOP website: https://www.genenames.org/tools/hcop/

    """
    if url is None:
        url = f"{_HCOP_BASE}/human_{target_organism}_hcop_fifteen_column.txt.gz"
    # check if exists
    if filename is None:
        filename = os.path.basename(url.split("/")[-1])
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    else:
        _logg(f"File {filename} already exists. Skipping download.", level="info")

    mapping = pd.read_csv(filename, sep="\t")
    mapping['evidence'] = mapping['support'].apply(lambda x: len(x.split(",")))
    mapping = mapping[mapping['evidence'] >= min_evidence]

    if columns is not None:
        mapping = mapping[columns]

    return mapping
