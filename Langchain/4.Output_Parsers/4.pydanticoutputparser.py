from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser, JsonOutputParser  # Uncomment if needed
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

class Person(BaseModel):
    
    name : str = Field(description="name of the person")
    age: int = Field(gt=18, description="age of the person")
    city: str = Field(description="name of the city person belogs to")
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({'place': "Nepali"})

print(prompt)
print("===============================================================")
result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)

## ---------------------------- Using chain 

# chain = template | model | parser

# final_result = chain.invoke({'place': "Japan"})

# print(final_result)
