from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
# from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

# parser1 = StrOutputParser()

class Feedback(BaseModel):
    
    sentiment: Literal["positive", "negative"] = Field(description="Provide the proper sentiment of the feedback of the user")

parser = PydanticOutputParser(pydantic_object=Feedback)


#### Classsification
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following text into 'positive' or 'negative'.\n"
        "{format_instruction}\n\n"
        "Text: {feedback}"
    ),
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

classifier_chain = prompt1 | model | parser

# result = classifier_chain.invoke({"feedback": " This is a very good smartphone. I loved it very much !!!"})

# print(result) 


prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"]
)

# branch_chain = RunnableParallel(
#     (condition1, chain1),
#     (condition2, chain2),
#     default chain
# )


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt2 | model | StrOutputParser()),
    (lambda x:x.sentiment == "negative", prompt3 | model | StrOutputParser()),
    RunnableLambda(lambda x: "Could not find the sentiment")
)

full_chain = classifier_chain | branch_chain

### +=============== Testing the result ====================================
print("Testing with negative feedback:")
result = full_chain.invoke({"feedback": "This is a terrible phone."})
print(result)

print("\n" + "="*50 + "\n")

print("Testing with positive feedback:")
result2 = full_chain.invoke({"feedback": "This is a very good smartphone. I loved it very much !!!"})
print(result2)

### print the chain graph
full_chain.get_graph().print_ascii()

# ## graph ======================
# #     +-------------+      
# #     | PromptInput |
# #     +-------------+
# #             *
# #             *
# #             *
# #    +----------------+
# #    | PromptTemplate |
# #    +----------------+
# #             *
# #             *
# #             *
# #       +----------+
# #       | ChatGroq |
# #       +----------+
# #             *
# #             *
# #             *
# # +----------------------+
# # | PydanticOutputParser |
# # +----------------------+
# #             *
# #             *
# #             *
# #        +--------+
# #        | Branch |
# #        +--------+
# #             *
# #             *
# #             *
# #     +--------------+
# #     | BranchOutput |
#     +--------------+