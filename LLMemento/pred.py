def make_pred(agent, legislator, ticker):
    fpath = prep(legislator, ticker) # this deletes and returns pickle path

    # perform task
    from prompt import gen_task_prompt, gen_task_prompt_wisdom
    task = gen_task_prompt(legislator, ticker)
    task_wisdom = gen_task_prompt_wisdom(task)
    answer = agent(task_wisdom)

    # result of task performed
    log = answer['intermediate_steps']
    log_str = stringfy_log(log)
    pred = answer['output']
    if pred not in ['EXIST', 'NOT-EXIST']:
        raise ValueError(f"pred should be either 'EXIST' or 'NOT-EXIST' but got {pred}") if pred not in ['EXIST', 'NOT-EXIST'] else pred
    eval = 'Inaccurate' if pred == 'NOT-EXIST' else 'Accurate'
    
    # after the task performed
    restore(fpath) # restore the relationships
    return pred, log_str, eval, task_wisdom


def prep(legislator, ticker):
    from utils import GraphDatabase
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "1dbstntk"
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def fetch_relationships(tx, legislator, ticker):
        query = """
        MATCH (l:Legislator {name: $legislator})-[r:BUY_SELL]->(s:Ticker {ticker: $ticker})
        RETURN l.name, r.start_date, r.end_date, s.name, s.ticker;
        """
        return list(tx.run(query, legislator=legislator, ticker=ticker))

    with driver.session() as session:
        relationships = session.read_transaction(fetch_relationships, legislator=legislator, ticker=ticker)
    print(relationships)
    assert len(relationships) != 0

    import pickle 
    # Convert and store relationships into a pickle file
    def pickle_relationships(relationships, filename):
        # Convert Record objects to dictionaries
        dict_relationships = [dict(record) for record in relationships]
        
        with open(filename, "wb") as file:
            pickle.dump(dict_relationships, file)
    filename=f'relationships_{legislator}_{ticker}.pkl'
    pickle_relationships(relationships=relationships, filename=filename) 

    # Delete relationships

    def delete_relationships(tx, legislator, ticker):
        query = """
        MATCH (l:Legislator {name: $legislator})-[r:BUY_SELL]->(s:Ticker {ticker: $ticker})
        DELETE r;
        """
        tx.run(query, legislator=legislator, ticker=ticker)

    with driver.session() as session:
        session.write_transaction(delete_relationships, legislator=legislator, ticker=ticker)

    # after delete, check it's deleted
    with driver.session() as session:
        relationships = session.read_transaction(fetch_relationships, legislator=legislator, ticker=ticker)
    print(relationships)
    assert len(relationships) == 0

    return filename

def restore(filename):
    from utils import GraphDatabase
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "1dbstntk"
    driver = GraphDatabase.driver(uri, auth=(user, password))

    import pickle
    # Open the file in binary read mode
    with open(filename, 'rb') as file:
        # Unpickle the file content
        relationships = pickle.load(file)

    # Now, relationships variable contains the unpickled data
    print(relationships)
    assert len(relationships) != 0

    # restore 
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
    with driver.session() as session:
        session.write_transaction(restore_relationships, relationships)

def stringfy_log(log):
    log_str = ''
    for l in log:
        log_str += str(l)
        log_str += '\n'
    # print(log_str)
    return log_str


# if __name__ == "__main__":
#     # prep(legislator='Ron Wyden', ticker='AMAT')
#     # restore(filename='relationships_Ron Wyden_AMAT.pkl')
#     pass
# restore(filename='relationships_Ron Wyden_AMAT.pkl')