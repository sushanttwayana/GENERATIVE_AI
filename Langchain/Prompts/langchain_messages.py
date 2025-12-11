from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

messages = [
    SystemMessage(content="You are a helpful assistant handling the user queries properly."),
    HumanMessage(content="Tell me about LangChain")
]

result = model.invoke(messages)

messages.append(AIMessage(content= result.content))

print(messages)