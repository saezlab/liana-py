# LIANA: a LIgand-receptor ANalysis frAmework <img src="https://github.com/saezlab/liana-py/blob/main/docs/source/logo.png?raw=true" align="right" height="125">

## Install LIANA
```
pip install git+https://github.com/saezlab/liana-py
```

## Documentation
url:

## Tutorials
- [Basic Usage]()
- [Resource Customization]()


## Methods

The methods implemented in this repository are:

- [CellPhoneDBv2](https://github.com/Teichlab/cellphonedb)
- [NATMI](https://github.com/forrest-lab/NATMI)
- [Connectome](https://github.com/msraredon/Connectome)
- [SingleCellSignalR](https://github.com/SCA-IRCM/SingleCellSignalR)
- *1-vs-rest* expression LogFC score
- `rank_aggregate` of the predictions calculated with the
[RobustRankAggregate](https://academic.oup.com/bioinformatics/article/28/4/573/213339) method


TO DO:
- [CellChat](https://github.com/sqjin/CellChat)
- Geometric mean + perms
- 

## Ligand-Receptor Resources


The following CCC resources are accessible via this pipeline:

- Consensus ($)
- CellCall
- CellChatDB
- CellPhoneDB
- Ramilowski2015
- Baccin2019
- LRdb
- Kiroauc2010
- ICELLNET
- iTALK
- EMBRACE
- HPMR
- Guide2Pharma
- ConnectomeDB2020
- CellTalkDB

($) LIANA's default resource was generated from the Consensus of several expert-curated resources
, then filtered to additional quality control steps including literature support, complex re-union/consensus,
and localisation.


