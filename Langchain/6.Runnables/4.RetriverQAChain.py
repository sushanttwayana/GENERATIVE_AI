from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 🔹 Load the PDF
loader = PyPDFLoader("example.pdf")
documents = loader.load()

# 🔹 Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 🔹 Convert text into embeddings & store in FAISS
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# 🔹 Initialize the LLM
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# 🔹 Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 🔹 Ask a question
query = "What are the key takeaways from the document?"
answer = qa_chain.run(query)

# 🔹 Print the Answer
print("Answer:", answer)
