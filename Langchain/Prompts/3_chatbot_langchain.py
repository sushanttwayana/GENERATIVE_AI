from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import os 

load_dotenv()
# model = ChatOpenAI()

chat_history = [    
                SystemMessage(content="You are a helpful assistant handling the user queries properly.")
            ]

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

while True:
    user_input = input('You:')
    
    chat_history.append(HumanMessage(content=user_input))
    
    if user_input == 'exit':
        break
    
    # result = model.invoke(user_input)
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)
    
print(chat_history)