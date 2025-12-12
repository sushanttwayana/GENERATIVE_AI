from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

## Chat Template
chat_template= ChatPromptTemplate([
    ('system', "you are a helpful customer support agent"),
    MessagesPlaceholder(variable_name='chat_history'), ## all previous memory histroy of the user
    ('human', '{query}')
])

## load chat history
chat_history = []

with open('chatbot_history.txt') as f:
    chat_history.extend(f.readlines())
    
print(chat_history)


## Create a prompt
prompt = chat_template.invoke({"chat_history":chat_history, 'query': HumanMessage(content= "Where is my refund??")})

print(prompt)