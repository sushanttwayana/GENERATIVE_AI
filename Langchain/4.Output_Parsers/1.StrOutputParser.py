from dotenv import load_dotenv
from transformers import pipeline
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=128,
    temperature=0.8,
)

# Wrap the HF pipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Now pass wrapped LLM
model = ChatHuggingFace(llm=llm)

## 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed proper report on the provided {topic}",
    input_variables= ["topic"]
)

## 2nd priimpt -> summary
template2 = PromptTemplate(
    template="write a 5 line summary on the following text. /n {text}",
    input_variables= ["topic"]
)

# prompt1 = template1.invoke({'topic': "Model Context Protocol"})

# result = model.invoke(prompt1)

# prompt2 = template2.invoke({"text": result.content})

# result1 = model.invoke(prompt2)

# print(result1.content)

## Creating output parser
parser = StrOutputParser()

first_chain = template1 | model | parser 
second_chain = template2 | model | parser


report = first_chain.invoke({"topic": "Football"})
summary = second_chain.invoke({"text": report})

print("Report:\n", report)
print("\nSummary:\n", summary)