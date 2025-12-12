from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

prompt = PromptTemplate(
    template = "generate 5 intresting facts about {topic}",
    input_variables=["topic"],
    )

parser = StrOutputParser()

chain = prompt | model | parser

chain.invoke({'topic':'football'})

### print the chain graph
chain.get_graph().print_ascii()