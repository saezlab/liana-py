from itertools import product
import urllib.request
import os

import numpy as np
import pandas as pd


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
    resource,
    map_df,
    column,
    replace=True,
    one_to_many=1,
    ):
    """
    Generate orthologs for a given column in a DataFrame.

    Parameters
    ----------
    resource : pandas.DataFrame
        Input DataFrame.
    map_df : pandas.DataFrame
        DataFrame with orthology mappings, where the first column is the source and the second column is the target for mapping.
    column : str
        Column name to translate.
    replace : bool, optional
        Whether to replace the original column with the translated values. Default is True.
        If False, it will create a new column with the prefix "orthology_".
    one_to_many : int, optional
        Maximum number of orthologs allowed per gene. Default is 1.

    Returns
    -------
    Resulting DataFrame with translated column.

    """
    if not isinstance(one_to_many, int):
        raise ValueError("`one_to_many` should be a positive integer!")

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
        resource = resource.drop(columns=["orthology_target"])
    else:
        resource[column] = resource.apply(
            lambda x: x["orthology_target"]
            if not pd.isnull(x["orthology_target"])
            else x[column],
            axis=1,
        )
        resource.rename(columns={"orthology_target": f"orthology_{column}"}, inplace=True)

    resource = resource.dropna(subset=[column])
    return resource


# function that loops over columns and applies translate_column
def translate_resource(resource, map_df, columns=['ligand', 'receptor'], **kwargs):
    """
    Generate orthologs for multiple columns in a DataFrame.

    Parameters
    ----------
    resource : pandas.DataFrame
        Input DataFrame.
    map_df : pandas.DataFrame
        DataFrame with orthology mappings, where the first column is the source and the second column is the target for mapping.
    columns : list
        List of column names to translate.
    **kwargs
        Additional arguments for `liana.utils.translate_column`.

    Returns
    -------
    Resulting DataFrame with translated columns.

    """
    for column in columns:
        resource = translate_column(resource, map_df, column, **kwargs)

    return resource


def get_hcop_orthologs(url="https://ftp.ebi.ac.uk/pub/databases/genenames/hcop/human_mouse_hcop_fifteen_column.txt.gz",
                       filename="human_mouse_hcop_fifteen_column.txt.gz",
                       min_evidence=3,
                       columns = ['human_symbol', 'mouse_symbol']
                       ):
    """
    Simple function to download the HCOP file from the EBI FTP server and filter it by minimum evidence.

    Parameters
     ----------
    url : str
        URL of the HCOP file. See https://ftp.ebi.ac.uk/pub/databases/genenames/hcop/ for bulk download options besides human and mouse.
    filename : str
        Name of the file to save the HCOP file.
    min_evidence : int
        Minimum number of evidences to keep the interaction.
    columns : list
        Columns to keep in the final DataFrame. If None, it will keep the default columns.

    Returns
    -------
    mapping : pd.DataFrame
        DataFrame with the HCOP mapping.

    Details
    -------
    HCOP is a composite database combining data from various orthology resources.
    It provides a comprehensive set of orthologs among human, mouse, and rat, among many other species.

    If you use this function, please reference the original HCOP papers:
    - Eyre, T.A., Wright, M.W., Lush, M.J. and Bruford, E.A., 2007. HCOP: a searchable database of human orthology predictions. Briefings in bioinformatics, 8(1), pp.2-5.
    - Yates, B., Gray, K.A., Jones, T.E. and Bruford, E.A., 2021. Updates to HCOP: the HGNC comparison of orthology predictions tool. Briefings in Bioinformatics, 22(6), p.bbab155.

    For more information, please visit the HCOP website: https://www.genenames.org/tools/hcop/,
    or alternatively check the bulk download FTP links page: https://ftp.ebi.ac.uk/pub/databases/genenames/hcop/
    """
    # check if exists
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"File {filename} already exists. Skipping download.")
    mapping = pd.read_csv(filename, sep="\t")
    mapping['evidence'] = mapping['support'].apply(lambda x: len(x.split(",")))
    mapping = mapping[mapping['evidence'] >= min_evidence]

    if columns is not None:
        mapping = mapping[columns]

    return mapping
