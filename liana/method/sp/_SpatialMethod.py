from pandas import DataFrame


# TODO MethodMeta class to generalizable, SpatialMethod & SingleCellMethod should inherit and extend
class SpatialMethod:
    """
    A SpatialMethod Class
    """

    def __init__(self,
                 method_name: str,
                 key_cols: list,  # note that this is defined here but not in Method
                 reference: str
                 ):
        """
        Parameters
        ----------
        method_name
            Name of the Method
        key_cols
            columns which make every interaction unique (i.e. PK).
        reference
            Publication reference in Harvard style
        """
        self.method_name = method_name
        self.key_cols = key_cols
        self.reference = reference

    def describe(self):
        """Briefly described the method"""
        print(f"{ self.method_name } does XYZ")

    def reference(self):
        """Prints out reference in Harvard format"""
        print(self.reference)

    def get_meta(self):
        """Returns method metadata as pandas row"""
        meta = DataFrame([{"Method Name": self.method_name,
                           "Reference": self.reference
                           }])
        return meta
