from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)

# from semantic_chunker_langchain.chunker import SemanticChunker, SimpleSemanticChunker
# from semantic_chunker_langchain.extractors.pdf import extract_pdf
# from semantic_chunker_langchain.outputs.formatter import write_to_txt

# # Extract
# docs = extract("sample.pdf")

# # Using SemanticChunker
# chunker = SemanticChunker(model_name="gpt-3.5-turbo")
# chunks = chunker.split_documents(docs)

# # Save to file
# write_to_txt(chunks, "output.txt")

# # Using SimpleSemanticChunker
# simple_chunker = SimpleSemanticChunker(model_name="gpt-3.5-turbo")
# simple_chunks = simple_chunker.split_documents(docs)