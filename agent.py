#from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PubMedLoader

from langchain.prompts import ChatPromptTemplate


from langchain.agents import AgentType, initialize_agent, Tool

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

#llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.3, max_retries=5, timeout=60)


#  creating function for interacting with pubmed api for Retrieve PubMed abstracts
def pubmed_papers(query,max_doc=3):
    
    print(f"🔍 Searching PubMed for: '{query}'...")

    loader = PubMedLoader(query=query, load_max_docs=max_doc)
    docs = loader.load() #it will load the query and exectu it and fetech the results from pubmed
    print(f"✅ Retrieved {len(docs)} papers from PubMed.")
    # Convert docs → text
    combined_text = "\n\n".join(
        f"Title: {d.metadata.get('Title','')}\nAbstract: {d.page_content}"
        for d in docs
    )

    return combined_text


#for agent 
# Define tools for agents
tools = [
    Tool(name="PubMedSearch", func=pubmed_papers, description=("Search PubMed for relevant papers. "
            "Input: a short query string describing the topic. "
            "Optional second arg: max_doc (int). "
            "Return: a concatenated string of abstracts of the relivant paper.")),
    # Tool(name="ExperimentDesigner", func=design_experiment, description="Design experiments"),
    # ... other tools
]

#system prompt which tells agent what to do, like whats your role and stuff 
researcher_prompt = ChatPromptTemplate.from_template("""
You are a scientific writer. BEFORE designing an experiment, you MUST search the literature using the PubMedSearch tool.
1) First call the PubMedSearch tool with the user's query (use up to 3 papers).
2) Summarize all the retrieved papers as one summary.
3) After summarizing the literature, propose an experiment using the following  output structure (strict):
hypothesis, variables, Methodology, expected outcomes.
Each section MUST include numeric values, durations, and procedures.
Do NOT summarize papers in the final protocol.
""")
#agent intializer 
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, #good with api, where they dont have to think that much
#agent=AgentType.OPENAI_FUNCTIONS, # this will help model to select the specfic tool
    verbose = True, #this will explain every step like how agent is 
#    handle_parsing_errors=True,
    agent_kwargs={"system_message": researcher_prompt}
)

#calling an agent
query="design an experiment to test whether the stress changes the natural balance between safe, enclosed spaces and exposed, rewarding spaces in mice."
result = agent.run(query)
print("\nFINAL ANSWER:\n", result)


