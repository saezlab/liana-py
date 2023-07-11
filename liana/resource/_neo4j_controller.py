from neo4j import GraphDatabase, unit_of_work

class Neo4jController:
    
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(uri, auth=(user, pwd))
        except Exception as e:
            print('Failed to create the driver', e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def get_subgraph(
    self,
    cellular_locations,
    tissue_locations,
    biospecimen_locations,
    database_cutoff,
    experiment_cutoff, 
    prediction_cutoff,
    combined_cutoff
):
        with self.__driver.session() as session:
            result = session.read_transaction(
                self.__get_subgraph,
                cellular_locations,
                tissue_locations,
                biospecimen_locations,
                database_cutoff,
                experiment_cutoff, 
                prediction_cutoff,
                combined_cutoff
            )
            return result

    @staticmethod
    @unit_of_work(timeout=1000)
    def __get_subgraph(
        tx,
        cellular_locations,
        tissue_locations,
        biospecimen_locations,
        database_cutoff,
        experiment_cutoff,
        prediction_cutoff,
        combined_cutoff
    ):

        result = tx.run(
            """MATCH (m)-[a]->(p:Protein)
            WHERE type(a) IN [ "StitchMetaboliteReceptor", "NeuronchatMetaboliteReceptor", "CellphoneMetaboliteReceptor"] 
            AND (($database_cutoff <= a.database) 
                OR ($experiment_cutoff <= a.experiment) 
                OR ($prediction_cutoff <= a.prediction) 
                OR ($combined_cutoff <= a.combined) 
                OR (type(a) IN [ "NeuronchatMetaboliteReceptor", "CellphoneMetaboliteReceptor"]))
            AND ((a.database >= 150 ) 
                OR (a.experiment >= 150) 
                OR (a.prediction >= 150 ) 
                OR (type(a) IN [ "NeuronchatMetaboliteReceptor", "CellphoneMetaboliteReceptor"]))
            AND a.mode IN ["activation", "inhibition", "binding"]
            AND ANY(value IN m.cellular_locations WHERE value IN $cellular_locations)
            AND (ANY(value IN m.tissue_locations WHERE value IN $tissue_locations) 
                OR ANY(value IN m.biospecimen_locations WHERE value IN $biospecimen_locations))
            RETURN m.id as HMDB,
                m.name as MetName,
                p.symbol as Symbol
                """,
            database_cutoff=database_cutoff,
            experiment_cutoff=experiment_cutoff,
            prediction_cutoff=prediction_cutoff,
            combined_cutoff=combined_cutoff,
            cellular_locations=cellular_locations,
            tissue_locations=tissue_locations,
            biospecimen_locations=biospecimen_locations 
            
        )

        return result.data()
        
        


 