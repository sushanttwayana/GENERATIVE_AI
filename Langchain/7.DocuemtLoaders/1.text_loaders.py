from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

prompt = PromptTemplate(
    template="Write a summary for the following poem -\n {poem}",
    input_variables= ['poem']
)

parser = StrOutputParser()



# print(type(docs))
# print(type(docs[0]))

# print(len(docs))

# print(docs[0])

# print(docs.metadata)


# print(docs.page_content)

chain = prompt | model | parser

print(chain.invoke({"poem": docs[0].page_content}))