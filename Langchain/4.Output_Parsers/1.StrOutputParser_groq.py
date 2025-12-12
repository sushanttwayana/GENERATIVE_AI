from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

## 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed proper report on the provided {topic}",
    input_variables= ["topic"]
)

## 2nd priimpt -> summary
template2 = PromptTemplate(
    template="write a 5 line summary on the following text. /n {text}",
    input_variables= ["text"]
)

## Creating output parser
parser = StrOutputParser()

first_chain = template1 | model | parser 
second_chain = template2 | model | parser


report = first_chain.invoke({"topic": "Football"})
summary = second_chain.invoke({"text": report})

print("Report:\n", report)
print("\nSummary:\n", summary)
 