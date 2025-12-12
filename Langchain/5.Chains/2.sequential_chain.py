from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary for the given text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model |parser | prompt2

result = chain.invoke({"topic": "Nepal"})
print(result)

### print the chain graph
chain.get_graph().print_ascii()