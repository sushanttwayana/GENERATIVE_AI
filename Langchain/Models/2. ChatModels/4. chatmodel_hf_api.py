from dotenv import load_dotenv
from transformers import pipeline
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline

load_dotenv()

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=128,
    temperature=0.3,
)

# Wrap the HF pipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Now pass wrapped LLM
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Nepal?")
print(result.content)

# ChatHuggingFace expects a LangChain-compatible LLM wrapper.

# HuggingFacePipeline adapts the raw Transformers pipeline into the correct interface.

# After wrapping it, ChatHuggingFace works normally.