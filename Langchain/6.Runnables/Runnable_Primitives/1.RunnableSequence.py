from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

prompt1 = PromptTemplate(
    template="write a joke on the {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({'topic': 'Langchain'}))