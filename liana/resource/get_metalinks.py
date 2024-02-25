import os
import sqlite3
import pandas as pd
from typing import Optional, List
from liana._logging import _logg, _check_if_installed

def _download_metalinksdb(verbose=True):
    """
    Ensures the Metalinksdb is downloaded and available for use.
    If the Metalinks database is not present in the current working directory, it downloads it.

    Returns
    -------
    str
        The path to the downloaded database file.
    """
    requests = _check_if_installed("requests")

    METALINKS_URL = "https://figshare.com/ndownloader/files/44624707?private_link=4744950f8768d5c8f68c"

    # Define the local filename to save the downloaded database
    db_file_name = 'metalinksdb.db'
    db_path = os.path.join(os.getcwd(), db_file_name)

    # Check if the database file already exists
    if not os.path.exists(db_path):
        _logg("Downloading database...", verbose=verbose)
        response = requests.get(METALINKS_URL)
        with open(db_path, 'wb') as f:
            f.write(response.content)
        _logg(f"Database downloaded and saved to {db_path}.", verbose=verbose)

    return db_path


def get_metalinks(db_path: Optional[str] = None,
                  cell_location: Optional[List[str]] = None,
                  tissue_location: Optional[List[str]] = None,
                  biospecimen_location: Optional[List[str]] = None,
                  disease: Optional[List[str]] = None,
                  pathway: Optional[List[str]] = None,
                  hmdb_ids: Optional[List[str]] = None,
                  uniprot_ids: Optional[List[str]] = None,
                  source: Optional[List[str]] = None
                  ):
    """
    Fetches edges of metabolite-proteins with specified annotations, applying filters if they are not None.
    Allows filtering by lists of hmdb and uniprot IDs and uses source for filtering without including it in the result,
    avoids duplicate column names, and returns the results as a pandas DataFrame.
    Filters are applied using INNER JOINs and WHERE clauses - i.e. the results are the intersection of the filters.

    Parameters
    ----------
    db_path
        Path to the SQLite database file. If None, the database will be downloaded to the current working directory.
    cell_location
        Desired metabolite cell locations.
    tissue_location
        Desired metabolite tissue locations.
    biospecimen_location
        Desired metabolite biospecimen locations.
    disease
        Desired metabolite diseases.
    pathway
        Desired metabolite pathways.
    hmdb_ids
        Desired HMDB IDs.
    uniprot_ids
        Desired UniProt IDs.
    source
        Desired sources for filtering.

    Returns
    ----------

    A pandas DataFrame containing the query results without the source column.
    """

    if db_path is None:
        db_path = _download_metalinksdb()
    conn = sqlite3.connect(db_path)

    # Adjusted SELECT statement to exclude the source column
    base_query = """
    SELECT m.metabolite AS metabolite, m.hmdb as hmdb, p.uniprot AS uniprot, p.gene_symbol as gene_symbol
    FROM edges e
    JOIN metabolites m ON e.hmdb = m.hmdb
    JOIN proteins p ON e.uniprot = p.uniprot
    """

    # Source table join is conditional based on source being provided
    if source:
        base_query += "JOIN source s ON e.hmdb = s.hmdb AND e.uniprot = s.uniprot "

    def _to_list(x):
        if isinstance(x, str):
            return [x]
        return x

    cell_location = _to_list(cell_location)
    tissue_location = _to_list(tissue_location)
    biospecimen_location = _to_list(biospecimen_location)
    disease = _to_list(disease)
    pathway = _to_list(pathway)
    hmdb_ids = _to_list(hmdb_ids)
    uniprot_ids = _to_list(uniprot_ids)
    source = _to_list(source)

    annotations_filters = {
        'cell_location': cell_location,
        'tissue_location': tissue_location,
        'biospecimen_location': biospecimen_location,
        'disease': disease,
        'pathway': pathway
    }

    join_clauses = []
    where_clauses = []
    for annotation_table, values in annotations_filters.items():
        if values is not None:
            join_clause = f"INNER JOIN {annotation_table} ON m.hmdb = {annotation_table}.hmdb"
            join_clauses.append(join_clause)

            values_str = ", ".join([f"'{value}'" for value in values])
            where_clause = f"{annotation_table}.{annotation_table} IN ({values_str})"
            where_clauses.append(where_clause)

    if hmdb_ids:
        hmdb_str = ", ".join([f"'{i}'" for i in hmdb_ids])
        where_clauses.append(f"m.hmdb IN ({hmdb_str})")

    if uniprot_ids:
        uniprot_str = ", ".join([f"'{i}'" for i in uniprot_ids])
        where_clauses.append(f"p.uniprot IN ({uniprot_str})")

    if source:
        source_str = ", ".join([f"'{src}'" for src in source])
        where_clauses.append(f"s.source IN ({source_str})")

    full_query = base_query
    if join_clauses:
        full_query += " " + " ".join(join_clauses)
    if where_clauses:
        full_query += " WHERE " + " AND ".join(where_clauses)

    df = pd.read_sql_query(full_query, conn)
    conn.close()

    return df


def get_metalinks_values(table_name, column_name, db_path: Optional[str] = None):
    """
    Fetches distinct values from a specified column in a specified table.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file. If None, the database will be downloaded to the current working directory.
    table_name : str
        Name of the table from which to fetch distinct values.
    column_name : str
        Name of the column from which to fetch distinct values.

    Returns
    -------
    list
        A list of distinct values from the specified column.
    """
    if db_path is None:
        db_path = _download_metalinksdb()
    conn = sqlite3.connect(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"SELECT DISTINCT {column_name} FROM {table_name};"
    cursor.execute(query)
    distinct_values = cursor.fetchall()
    conn.close()
    return [value[0] for value in distinct_values]


def describe_metalinks(db_path: Optional[str] = None, return_output: bool = False):
    """
    Prints the schema information and foreign key details for all tables in the specified SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file. If None, the database will be downloaded to the current working directory.
    """
    if db_path is None:
        db_path = _download_metalinksdb()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    output = ""
    for table in tables:
        table_name = table[0]
        output += f"Schema of table: {table_name}\n{'=' * len(f'Schema of table: {table_name}')}\n"

        cursor.execute(f"PRAGMA table_info({table_name});")
        schema_info = cursor.fetchall()
        for column in schema_info:
            cid, name, ctype, _, _, pk = column
            output += f"Column ID: {cid}, Name: {name}, Type: {ctype}, Primary Key: {pk}\n"

        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        fk_info = cursor.fetchall()
        if fk_info:
            output += "\nForeign Keys:\n"
            for fk in fk_info:
                id, seq, table, from_col, to_col, _, _, _ = fk
                output += f"ID: {id}, Seq: {seq}, Table: {table}, From: {from_col}, To: {to_col}\n"
        else:
            output += "\nNo Foreign Keys.\n"
        output += "-" * 40 + "\n"

    if return_output:
        return output
    else:
        print(output)
