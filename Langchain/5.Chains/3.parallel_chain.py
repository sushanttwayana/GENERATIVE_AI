from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from transformers import pipeline
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.runnables import RunnableParallel ### can be used to execute multiple parallel chains at the same time

load_dotenv()

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=128,
    temperature=0.3,
)

# Wrap the HF pipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Now pass wrapped LLM
model1 = ChatHuggingFace(llm=llm)

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# model = ChatGroq(model="openai/gpt-oss-120b")
model2 = ChatGroq(model="llama-3.3-70b-versatile")

### Creating the prompts

prompt1 = PromptTemplate(
    template = "Generate short and concise notes from the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short questions answers from the following text \n {text} ",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

 
parser = StrOutputParser()

### Creating the chain

parallel_chain = RunnableParallel({  
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 |parser
    })


text = """
LangChain is an open-source framework that simplifies building applications powered by large language models (LLMs).

It provides modular building blocks like chains for sequencing LLM calls with prompts and tools, agents for dynamic decision-making, 
and memory systems to maintain conversation context across interactions. Retrieval modules enable integration with vector databases for RAG (Retrieval-Augmented Generation) systems.

LangChain streamlines workflows by offering seamless integrations with models like Groq, OpenAI, and external data sources, boosting developer productivity through reusable components and debugging tools.
This makes it ideal for creating chatbots, document analyzers, and AI agents.
"""


merge_chain = prompt3 | model2 |parser

chain = parallel_chain | merge_chain

result = chain.invoke({'text': text})

print(result)


### print the chain graph
chain.get_graph().print_ascii()