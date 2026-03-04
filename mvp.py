#from langchain.chat_models import ChatOpenAI // this is old one 
#from langchain_openai import ChatOpenAI   # this is new one as old is deperaction
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain

#llm = ChatOpenAI(model="gpt-4o-mini")

from dotenv import load_dotenv
load_dotenv() 
#llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.3)

#llm = ChatOpenAI(model="gpt-4o-mini")

researcher_prompt = ChatPromptTemplate.from_template("""
You are a creative scientific researcher. 
Given the topic below, propose a detailed experimental design:
- Include: hypothesis, variables, methodology, expected outcomes.
Topic: {topic}
""")

researcher_chain = LLMChain(prompt=researcher_prompt, llm=llm, output_parser=StrOutputParser())


analyst_prompt = ChatPromptTemplate.from_template("""
You are a data analyst collaborating with a researcher. 
Here is the proposed experiment:
{experiment_design}
Suggest:
1. Data collection plan
2. Statistical analysis
3. Metrics for evaluation
4. Potential confounds and controls
""")

analyst_chain = LLMChain(prompt=analyst_prompt, llm=llm, output_parser=StrOutputParser())


reviewer_prompt = ChatPromptTemplate.from_template("""
You are a peer reviewer evaluating the experiment and analysis below.
Provide a detailed critique:
- Identify flaws, biases, missing controls, or unclear assumptions.
- Suggest how to improve it.

Experiment + Analysis:
{combined_output}
""")

reviewer_chain = LLMChain(prompt=reviewer_prompt, llm=llm, output_parser=StrOutputParser())
communicator_prompt = ChatPromptTemplate.from_template("""
You are a scientific writer. Summarize the reasoning process into a structured report:
Sections:
1. Research Question
2. Experiment Design
3. Analysis Plan
4. Reviewer Critique
5. Final Summary

Content:
{all_sections}
""")

communicator_chain = LLMChain(prompt=communicator_prompt, llm=llm, output_parser=StrOutputParser())

def collabchain_pipeline(topic):
    exp_design = researcher_chain.run({"topic": topic})
    analysis = analyst_chain.run({"experiment_design": exp_design})
    combined = exp_design + "\n\n" + analysis
    review = reviewer_chain.run({"combined_output": combined})
    final_report = communicator_chain.run({"all_sections": combined + "\n\nReviewer Comments:\n" + review})
    return final_report


topic = "design an to test rodents that are used as a model for Parkinson’s disease move more slowly or less often than healthy animal?."
output = collabchain_pipeline(topic)
print(output)
