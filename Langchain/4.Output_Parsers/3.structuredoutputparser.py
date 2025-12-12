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

### Schema to guide LLM
class FactsModel(BaseModel):
    fact_1: str = Field(description="Fact 1 about the topic")
    fact_2: str = Field(description="Fact 2 about the topic")
    fact_3: str = Field(description="Fact 3 about the topic")

parser = PydanticOutputParser(pydantic_object=FactsModel)


# 2. Build prompt with parser instructions
template = PromptTemplate(
    template=(
        "Provide exactly 3 factual statements about the topic '{topic}'.\n\n"
        "{format_instructions}"
    ),
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({"topic": "black hole"})

# 3. Invoke model and parse output using Pydantic parser
# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)

### with chains (example)
chain = template | model | parser
final_result_chain = chain.invoke({"topic": "black hole"})
print(final_result_chain)