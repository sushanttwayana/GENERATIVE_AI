from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

### ============== simple RunnableLambda==========================
## word counter
def word_counter(text):
    return len(text.split())

prompt1 = PromptTemplate(
    template="write an brief overview on the {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

# report_gen_chain = RunnableSequence(prompt1, model, parser)
report_gen_chain = prompt1 | model | parser

# RunnableBranch(
#     (condition, runnable),
#     (condition, runnable),
#     (condition, runnable),
#     default
# )


branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic': "World War II"}))