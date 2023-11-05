def gen_task_prompt(Legis, Ticker):
    task = f"""
You are granted access to a graph database that encompasses interactions and activities related to the U.S. Congress, its Legislators, and publicly traded companies. 
Your mission is to deduce the probability of a financial transaction between a designated U.S. Legislator, {Legis}, and a specific publicly traded company, {Ticker}.

Available Nodes:

- Legislator: Represents members of the U.S. Congress. Essential attributes include their 'name' and an identifier called 'bioguide'.
- Ticker: Symbolizes publicly traded companies on stock exchanges. Each company is uniquely identified by its 'ticker' symbol, a shorthand used on stock trading platforms, and its 'name', the official corporate designation.
- Committee: Stands for the specialized committees within the U.S. Congress, each of which focuses on specific domains like finance, defense, health, or education. Crucial properties include 'name' and 'id'.
- NAICS: Corresponds to the North American Industry Classification System, which categorizes businesses and industries. Pertinent attributes are 'ticker' and 'desc', which describe the sector or industry where 
a publicly traded company belongs.
- Bill: Denotes legislative bills processed within the U.S. Congress. Essential attributes include 'summary', 'id', 'official_title', and 'short_title'.
Available Relationships:

- BUY_SELL: Indicates a financial transaction or stock trading activity between a U.S. Legislator and a publicly traded company, with date specifics. (Legislator -> BUY_SELL -> Ticker)
- COMMITTEE_ASSIGNMENT: Relays the assignment of a U.S. Legislator to a specific Committee within Congress for a set tenure. (Legislator -> COMMITTEE_ASSIGNMENT -> Committee)
- LOBBY_ON: Chronicles the endeavors where a publicly traded company, via its Ticker, lobbied a particular Bill. (Ticker -> LOBBY_ON -> Bill)
- BELONGS_TO: Designates the industry or sector, as per the NAICS system, where a publicly traded company operates. (Ticker -> BELONGS_TO -> NAICS)
- ASSIGNED_TO: Signals the allocation of a Bill to a distinct Committee within Congress. (Bill -> Assigned_to -> Committee)

[TASK]
Using the information, scrutinize the nodes and relationships to estimate the likelihood of a "BUY_SELL" link between {Legis} and {Ticker}. You should give your guess (prediction) about a EXIST or NOT-EXIST of the transaction between {Legis} and {Ticker} as a final answer. 

[Note]
What to be Careful About: 

1. For the purposes of this task, any existing direct "BUY_SELL" relationships between {Legis} and {Ticker} have been intentionally deleted from graph database. Your goal is to infer the likelihood of such a relationship based on indirect data points.
Therefore, make sure that 
    - 1. You don't need to check whether there exist a direct "BUY_SELL" relationship between {Legis} and {Ticker} because it's already and intentionally deleted if exists.
    - 2. You should "guess" a EXIST or NOT-EXIST of the transaction between {Legis} and {Ticker} as a final answer.

2. When you fetch the bill node, do not fetch the `summary` attribute of the node because it's too long and this will deplete your Token limits to conduct the task successfully. Instead, you should fetch the `official_title` attribute of the node to understand the bill's content.
For example, if you run this query,  it will deplete your Token limits immediately.

MATCH (:Ticker {{name: 'Applied Materials Inc AMAT'}})-[:LOBBY_ON]->(b:Bill)
RETURN b

so instead, do:
MATCH (:Ticker {{name: 'Applied Materials Inc AMAT'}})-[:LOBBY_ON]->(b:Bill)
RETURN b.id, b.name, b.date

3. Avoid using plain language instructions as part of your graph database queries. 
For instance, you should not input text like "5. Find the bills that were assigned to the committee with id 'COM123':" into a tool designed for "GraphDB Query" execution. Instead, your input should consist solely of the Cypher query syntax, as shown below:

Correct Query Example:

MATCH (c:Committee {{id: 'COM123'}})<-[:ASSIGNED_TO]-(b:Bill)
RETURN b.official_title
"""
    return task

def gen_task_prompt_wisdom(task):
    prompt = f"""
    {task}

[Harnessing the Memory Vector Database]

You have a Memory Vector Database that encapsulates key strategies derived from your history of analyzing legislator-company stock transactions. This tool operates on a principle of condition-action pairs:

    Condition: A specific scenario or set of circumstances from past tasks.
    Action: The strategic response that was effective in that scenario.

When faced with a new task, the database can be leveraged as follows:

Describe the Condition: Input a description of the current task's conditions into the database. Then the database uses vector similarity to find past conditions that closely match your description.
It then presents you with a list of condition-action pairs that were effective in similar past scenarios. This process helps you to quickly identify the most relevant strategies for the current task, drawing on successful experiences to inform your decision-making.

The following is a list of condition-action pairs that you may be able to find in the database:

    [Example 1]
        Condition: Scenarios where a legislator has multiple overlapping NAICS codes with the tickers they've engaged in transactions with.
        Action: In such scenarios, confidently assert the existence of a relationship.

    [Example 2]
        Condition: Situations where there are no overlapping NAICS codes between a ticker and the legislator's transactions.
        Action: In these instances, scrutinize any bill titles lobbied by the ticker that are pertinent to the committees the legislator belongs to.

In addition, make sure you start your chain-of-action by using this "Memory" tool by describing your task so that you can more faster and more accurately solve the task.  
"""
    return prompt