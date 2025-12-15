from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

prompt = PromptTemplate(
    template= "Answer the following question \n {question} from the following text - \n  {text}",
    input_variables=['question', 'text']
)

parser = StrOutputParser()
 
url = "https://docs.langchain.com/oss/python/langchain/overview" 
 
loader = WebBaseLoader(url) 

docs = loader.load()


chain = prompt | model | parser


print(chain.invoke({"question": "What is langchain?", "text":docs[0].page_content}))
# print(len(docs))

# print(docs[0].page_content)