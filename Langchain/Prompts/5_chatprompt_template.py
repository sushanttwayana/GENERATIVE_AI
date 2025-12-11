from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

# chat_template = ChatPromptTemplate.from_messages([
#     ("system", "You are an {domain} expert."),
#     ("human", "Explain in simple terms about {topic}.")
# ])
chat_template = ChatPromptTemplate([
    ("system", "You are an {domain} expert."),
    ("human", "Explain in simple terms about {topic}.")
])

prompt = chat_template.invoke({'domain': 'football', 'topic': 'Cristiano Ronaldo'})

print(prompt)