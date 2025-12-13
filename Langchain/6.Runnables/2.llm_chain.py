# llmchain.py
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 🔹 Load the LLM (GPT-3.5)
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# 🔹 Create a Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],  # Defines what input is needed
    template="Suggest a catchy blog title about {topic}."
)

# 🔹 Create an LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# 🔹 Run the chain with a specific topic
topic = input('Enter a topic')
output = chain.run(topic)

print("Generated Blog Title:", output)
