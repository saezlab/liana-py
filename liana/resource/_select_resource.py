from pandas import read_csv


# Placeholder function to read in resource
def select_resource(resource_name='consensus'):
    resource = read_csv(f"{resource_name}.csv", index_col=False)  ### TO BE CHANGED

    resource = resource[['source_genesymbol', 'target_genesymbol']]
    resource = resource.rename(columns={'source_genesymbol': 'ligand',
                                        'target_genesymbol': 'receptor'})

    return resource


# Function to Decomplexify Resource
def _explode_complexes(resource):
    resource['interaction'] = resource['ligand'] + '|' + resource['receptor']
    resource = (resource.set_index('interaction')
                .apply(lambda x: x.str.split('_'))
                .explode(['receptor'])
                .explode('ligand')
                .reset_index()
                )
    resource[['ligand_complex', 'receptor_complex']] = resource[
        'interaction'].str.split('|', expand=True)

    return resource
