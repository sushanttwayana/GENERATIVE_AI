from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# 🔹 Create a Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)

# 🔹 Define the input
topic = input('Enter a topic')

# 🔹 Format the prompt manually using PromptTemplate
formatted_prompt = prompt.format(topic=topic)

# 🔹 Call the LLM directly
blog_title = llm.predict(formatted_prompt)

# 🔹 Print the output
print("Generated Blog Title:", blog_title)
