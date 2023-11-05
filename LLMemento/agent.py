from langchain.chat_models import ChatOpenAI

from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph

OPENAI_API_KEY = "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"

# HYPERPARAMETER
model_for_cypher = 'gpt-4'
model_for_agent = 'gpt-4'

# local
url = "bolt://localhost:7687"
username = "neo4j"
password = "1dbstntk"

graph = Neo4jGraph(
    url=url, username=username, password=password
)

# define the Graph Chain
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0, 
               openai_api_key=OPENAI_API_KEY,
               model=model_for_cypher,), 
    graph=graph,
    verbose=True,
    return_intermediate_steps=True
)

model_name = "text-embedding-ada-002"
OPENAI_API_KEY = "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"
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

k= 3 # hyperparameter
def refer_to_experience(data_landscape, k=k):

    from langchain.chains import RetrievalQA
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    import pinecone

    model_name = "text-embedding-ada-002"
    OPENAI_API_KEY = "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"
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

    # PINECONE_API_KEY = "978bf2d9-c8d3-449f-bdfb-5adeb77a6e97"
    PINECONE_API_KEY = "85623b4c-6d07-4e82-a03c-9b06beb27d88"
    PINECONE_ENV = "gcp-starter"
    from langchain.embeddings.openai import OpenAIEmbeddings

    # Initialize Pinecone
    index_name = "decision-tree"
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pinecone.Index(index_name)

    text_field = "condition"
    # text_field = "strategy" # for the first round of run, use the data_landscape
    vectorstore = Pinecone(index, embed.embed_query, text_field)

    # query = "legislator is assigned to a committee that has been lobbied on by the company"
    retdoc = vectorstore.similarity_search(
        data_landscape,  # our search query
        k=k  # return 3 most relevant docs 
    )


        # Post-process the results to replace 'page_content' with 'condition'
    processed_results = []
    for doc in retdoc:
        new_doc = {'condition': doc.page_content}
        new_doc['action'] = doc.metadata['action']
        processed_results.append(new_doc)

    return processed_results


llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, streaming=False, model_name=model_for_agent)

from pydantic import BaseModel, Field


# Define tools
tools = [
    Tool(
        name="GraphDB Query",
        func=chain.run,
        description="Useful to query about the following data: 1) Stocks traded by legislators, 2) Companies' lobbying on specific bills - where Companies are represented as Ticker, 3) Bill assignments to Committees, and 4) Committee memberships of legislators."
    ),
]

class ReferToExpInput(BaseModel):
    data_landscape: str = Field()


tools.append(
    Tool.from_function(
        func=refer_to_experience,
        name="Memory",
        description="""
        Useful for when you want to refer to your previous experience of solving the link prediction task over the pairs legislator-company (ticker). 
        This tool will return the condition and action pair you've developed from the previous experience.

        Input should be description of your condition you're facing now - then the tool will return the condition and action pair you've developed from the previous experience.
        """,
        args_schema=ReferToExpInput
    )
)

# Initialize agent
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True, 
    handle_errors=True, 
    handle_parsing_errors=True,
    return_intermediate_steps=True
)
