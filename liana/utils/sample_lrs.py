import numpy as np
import pandas as pd
import string


# Function to generate a liana_res sample/example
def sample_lrs():
    """Create sample method output for testing metrics in this task."""
    row_num = 200
    rng = np.random.default_rng(seed=1)

    label_vector = list(string.ascii_uppercase[0:10])
    entity_vector = list(string.ascii_lowercase[0:10])

    df = pd.DataFrame(rng.random((row_num, 1)), columns=["magnitude"])
    df["source"] = rng.choice(label_vector, row_num)
    df["target"] = rng.choice(label_vector, row_num)
    df["ligand_complex"] = rng.choice(entity_vector, row_num)
    df["receptor_complex"] = rng.choice(entity_vector, row_num)
    df["specificity"] = rng.random((200, 1))

    # deduplicate
    df = df.loc[~df.duplicated()]

    return df
