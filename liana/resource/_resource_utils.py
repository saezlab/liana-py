"""
Utility functions to query OmniPath.
Functions to retrieve resources from the meta-database OmniPath.
"""


def check_if_omnipath():
    """
    Function to check if available and return OmniPath
    
    Returns
    -------
    OmniPath package

    """
    try:
        import omnipath as op
    except Exception:
        raise ImportError('omnipath is not installed. Please install it with: pip install omnipath')
    return op


# Function to explode complexes (decomplexify Resource)

# """Functions to obtain additional OmniPath resources"""
# def obtain_extra_resource(databases,
#                           blocklist,
#                           allowlist
#                           ):
#     omnipath = check_if_omnipath()
#
#     # Obtain resource
#     add = omnipath.interactions.PostTranslational.get(databases=databases,
#                                                       genesymbols=True,
#                                                       entity_types=['protein', 'complex'],
#                                                       fields={"extra_attrs"}
#                                                       )
#     add = add[~add[['source', 'target']].duplicated()]
#
#     block_keys = blocklist.keys()
#     allow_keys = allowlist.keys()  # union of relevant checks
#     union_keys = block_keys ^ allow_keys
#
#     # explode relevant attributes and join them to database
#     explode_attrs = add['extra_attrs'].apply(_json_intersect_serialize, union_keys=union_keys)
#     add = pd.concat([add, explode_attrs], axis=1).drop('extra_attrs', axis=1)
#
#     # Convert blocklist to mask & remove unwanted rows
#     for k in block_keys:
#         add[k + '_msk'] = [
#             any([block in att for block in blocklist[k]]) if type(att) is not float else True
#             for att in add[k]]
#         # iter
#         add = add[~add[k + '_msk']]
#
#     # Convert allowlist to mask & keep only relevant rows
#     for k in allow_keys:
#         add[k + '_msk'] = [
#             any([allow in att for allow in allowlist[k]]) if type(att) is not float else False
#             for att in add[k]]
#         add = add[add[k + '_msk']]
#
#     return add
#
#
# # Function to format extra_attributes
# def _json_intersect_serialize(att, union_keys):
#     att = loads(att)
#     att = {k: att[k] for k in union_keys if k in att.keys()}
#     return pd.Series(att)
