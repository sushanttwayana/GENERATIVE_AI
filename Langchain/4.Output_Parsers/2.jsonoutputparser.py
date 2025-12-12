from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

parser = JsonOutputParser()

template = PromptTemplate(
    # template="give me the name, age and city of a fictional character \n, {format_instruction}", ## addtiional info about the expected output structure 
    template="give me 5 facts about {topic} \n, {format_instruction}", ## addtiional info about the expected output structure 
    # input_variables=[],
    input_variables=["topic"],
    partial_variables= {'format_instruction': parser.get_format_instructions()} ## the parser gets the instructions load ahead of the runtime
)

#### --------------without using chains

# prompt = template.format()

# # print(prompt)

# result = model.invoke(prompt)
# # print(result)

# final_result = parser.parse(result.content)

# print(final_result)
# print(type(final_result))

# print(final_result["name"])


#### ------------- using chains

chain = template | model | parser

# result = chain.invoke({})
result = chain.invoke({"topic": "Nepal"})

print(result)