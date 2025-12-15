from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Sushant Twayana Resume.pdf')

docs = loader.load()

text = "hello what is your name??"

text_splitter = CharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0,
    separator=' '
)

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
# )


# result = splitter.split_text(text)
result = text_splitter.split_documents(docs)

print(result[0])
print(result[1])