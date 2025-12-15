from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b")

### ============== simple RunnableLambda==========================
## word counter
def word_counter(text):
    return len(text.split())

# runnable_word_counter = RunnableLambda(word_counter)

# print(runnable_word_counter.invoke("How there how are you?"))


prompt1 = PromptTemplate(
    template="write a joke on the {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter)
})

# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'word_count': RunnableLambda(lambda x: len(x.split()))
# })

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': "AI"})

print(result)

final_result = """{} \n Word count - {}""".format(result['joke'], result['word_count'])

print(final_result)