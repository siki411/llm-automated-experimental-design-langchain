import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv()

docs = []
root_dir=f'data/'
#doc = os.listdir(root_dir)
#print(f"document names: {doc}")

#loading pdf file 
for file in os.listdir(root_dir):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(root_dir, file))
        docs.extend(loader.load()) #making langchain document object and storing the the content of file in docs

#chunking all the file data  and typcially it used to prepare data to load into vector database
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs) # here we are chunking so later we can reterive the chunks
#print(f"Created {len(chunks)} text chunks")



#lets create embeddings
embeddings=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=1536) # (the bigger the dimension means the bigger paragraph will be converted onces as embedding= more context) less dimension means less context capturing for the vector and offourse the cost is less

#lets create chroma for saving embedding for later reterivals 

vector_store = Chroma(
    embedding_function=embeddings, # this will be embedding technique 
    persist_directory='my_chroma_db', # that will be database directory name in which it will create database
    collection_name='sample' # this will be the table name in database
)
#add vector (it will generate embeddings using openai and then store them  in chroma db=vector_store())
vector_store.add_documents(chunks)

# now i will build a reteriver for reterving chunks from chroma db 
reterivers= vector_store.as_retriever(search_kwargs={"k": 14})

prompt = ChatPromptTemplate.from_template("""
You are an expert research assistant specializing in behavioral neuroscience.

Your task is to DESIGN an experiment ONLY using the information provided in the context.

Rules:
- Use ONLY the provided context as your knowledge source
- Do NOT use external knowledge
- If the context does not contain sufficient information to design the experiment, say:
"I don't know based on the provided documents."
- Do NOT hallucinate methods, tasks, or results

Context:
{context}

Research Question:
{input}

Instructions:
Using the context, design an experiment with the following structure:

1. Objective
2. Hypothesis (explicitly grounded in the context)
3. Subjects (species, grouping if mentioned)
4. Experimental Design
   - Independent variables
   - Dependent variables
   - Control conditions
5. Behavioral Tasks or Measures (ONLY if present in context)
6. Data Collection
7. Expected Outcomes (ONLY if inferable from context)
8. Limitations based on missing information in the context

If any section cannot be filled from the context, explicitly state that it cannot be determined.

Answer:
""")

#creating an llm instance so it can be invoked
llm = ChatOpenAI( model="gpt-4o-mini", temperature=0)
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

retrieval_chain = create_retrieval_chain(
    retriever=reterivers,
    combine_docs_chain=document_chain
)

query = "design an experiment in rats to test how alchole intake affects attention-related behaviors."
docs = reterivers.get_relevant_documents(query)
#for i, d in enumerate(docs):
#    print(f"\n--- Retrieved chunk {i+1} ---\n")
#    print(d.page_content[:500])

response = retrieval_chain.invoke({
    "input": query
})

print(response["answer"])