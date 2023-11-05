import pickle
    
# Load relationships from a pickle file
def unpickle_relationships(filename="relationships.pkl"):
    with open(filename, "rb") as file:
        return pickle.load(file)
    
# Convert and store relationships into a pickle file
def pickle_relationships(relationships, filename="relationships.pkl"):
    # Convert Record objects to dictionaries
    dict_relationships = [dict(record) for record in relationships]
    
    with open(filename, "wb") as file:
        pickle.dump(dict_relationships, file)

uri = "bolt://localhost:7687"
user = "neo4j"
password = "1dbstntk"

from neo4j import GraphDatabase

class Neo4jService(object):

    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()
        
    def add_nodes_legis(self, tx, nodes):
        query = (
            "UNWIND $nodes AS node "
            "CREATE (n:Legislator {name: node.name, bioguide: node.bioguide})" 
        )
        tx.run(query, nodes=nodes)
    
    def add_nodes_ticker(self, tx ,nodes):
        query = (
            "UNWIND $nodes AS node "
            "CREATE (n:Ticker {ticker: node.ticker, name: node.name})" 
        )
        tx.run(query, nodes=nodes)
    
    def add_nodes_committee(self, tx, nodes):
        query = (
            "UNWIND $nodes AS node "
            "CREATE (n:Committee {ticker: node.committee, name: node.name})" 
        )
        tx.run(query, nodes=nodes)

    def add_nodes_naics(self, tx, nodes):
        query = (
            "UNWIND $nodes AS node "
            "CREATE (n:NAICS {ticker: node.naics, desc: node.desc})" 
        )
        tx.run(query, nodes=nodes)    

    def add_nodes_bills(self, tx, nodes):
        query = (
            "UNWIND $nodes AS node "
            "CREATE (n:Bill {id: node.id, short_title: node.short_title, official_title: node.official_title, summary: node.summary_text})" 
        )
        tx.run(query, nodes=nodes) 

    def add_buy_sell_relationships_with_dates(self, tx, relationships):
        query = (
            "UNWIND $relationships AS rel "
            "MATCH (l:Legislator {bioguide: rel.legis_bioguide}), (t:Ticker {ticker: rel.ticker}) "
            "CREATE (l)-[:BUY_SELL {start_date: rel.start_date, end_date: rel.end_date}]->(t)"
        )
        tx.run(query, relationships=relationships)

def delete_relationships(tx, legislator, ticker):
    query = f"""
    MATCH (l:Legislator {{name: "{legislator}"}})-[r:BUY_SELL]->(s:Ticker {{ticker: "{ticker}"}})
    DELETE r;
    """
    tx.run(query)

def fetch_relationships(tx, legislator, ticker):
    query = """
    MATCH (l:Legislator {name: $legislator})-[r:BUY_SELL]->(s:Ticker {ticker: $ticker})
    RETURN l.name, r.start_date, r.end_date, s.name, s.ticker;
    """
    return list(tx.run(query, legislator=legislator, ticker=ticker))

def restore_relationships(tx, relationships):
    for record in relationships:
        query = """
        MATCH (l:Legislator {name: $l_name}), (s:Ticker {name: $s_name, ticker: $s_ticker})
        MERGE (l)-[r:BUY_SELL {start_date: $r_start_date, end_date: $r_end_date}]->(s)
        RETURN r;
        """
        tx.run(query, 
               l_name=record['l.name'],
               s_name=record['s.name'],
               s_ticker=record['s.ticker'],
               r_start_date=record['r.start_date'],
               r_end_date=record['r.end_date'])
