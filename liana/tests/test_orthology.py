from liana.resource import translate_column, translate_resource, get_hcop_orthologs, select_resource

import pandas as pd

def test_complex_cases():
    map_df = pd.DataFrame(
        {"source":
            ["CSF2RA", "IFNL3", "IFNL3", "IFNLR1", "IL10RB", "HCST", "CD8A", "CD8B", "IL4"],
        "target":
            ["Csf2ra", "Ifnl3", "Ifnl2", "Ifnlr1", "Il10rb", "Hcst", "Cd8a", "Cd8b1", "Il4"]
            }
    )
    df = pd.DataFrame(
        {
            "symbol": [
                "CSF2RA_CSF2RB",  # one to many
                "IFNL3_IFNLR1_IL10RB",  # 3 subunits
                "HCST_KLRK1",  # one subunit missing
                "CD8A_CD8B",  # 1 to 1
                "IL4",  # 1 to 1 simple protein
            ]
        }
    )

    default = translate_column(
        df,
        map_df=map_df,
        column="symbol",
    )
    assert all(default["symbol"] == ["Cd8a_Cd8b1", "Il4"])

    to_many = translate_column(
        df,
        map_df=map_df,
        column="symbol",
        replace=True,
        one_to_many=2,
    )
    expected = [
        "Cd8a_Cd8b1",
        "Ifnl2_Ifnlr1_Il10rb",
        "Ifnl3_Ifnlr1_Il10rb",
        "Il4",
    ]

    assert to_many.shape == (4, 1)
    assert all(to_many["symbol"].isin(expected))

    keep_missing = translate_column(
        df,
        map_df=map_df,
        column="symbol",
        replace=False,
        one_to_many=2,
    )
    untranslated = keep_missing["symbol"].isin(["HCST_KLRK1"])
    assert untranslated.any()


def test_translate_resource():
    resource = select_resource()
    map_df = get_hcop_orthologs(columns=['human_symbol', 'mouse_symbol'], min_evidence=3)
    map_df = map_df.rename(columns={"human_symbol": "source", "mouse_symbol": "target"})

    translated = translate_resource(resource, map_df, one_to_many=1)
    assert translated.shape[0] > 3000
    translated2 = translate_resource(resource, map_df, one_to_many=5, replace=False)
    assert translated2.shape[0] > translated.shape[0]


def test_get_hcop():
    mapping = get_hcop_orthologs(columns=None, min_evidence=0)
    assert mapping.shape[0] > 1000
    assert mapping.shape[1] == 16 # 15 columns + added evidence column
