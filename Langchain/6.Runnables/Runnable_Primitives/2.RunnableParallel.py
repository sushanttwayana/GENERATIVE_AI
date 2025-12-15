from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

prompt1 = PromptTemplate(
    template='generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='generate a simple linkedin post on {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({    
    'tweet': RunnableSequence(prompt1, model, parser),
    'linekdin': RunnableSequence(prompt2, model, parser)
    }
)

result = parallel_chain.invoke({"topic": "langchain"})

print(result)
print("=================================================================")
print(result['tweet'])
print(result['linekdin'])