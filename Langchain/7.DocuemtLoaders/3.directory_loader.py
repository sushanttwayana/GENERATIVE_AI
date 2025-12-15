from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


### Using load()
loader = DirectoryLoader(
    path="books",
    glob ="*.pdf",
    loader_cls = PyPDFLoader
)

docs = loader.load()

print(len(docs[0].page_content))
print(len(docs[0].metadata))


docs = loader.lazy_load()

for document in docs:
    print(document.metadata)