from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

# embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

embedding = OllamaEmbeddings(model="gemma:2b")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.",
    """Cristiano Ronaldo dos Santos Aveiro is a Portuguese professional footballer widely regarded 
    as one of the greatest players in history. Born on 5 February 1985 in Funchal, Madeira, 
    Ronaldo is known for his athleticism, goal-scoring ability, and legendary longevity. 
    He played for Sporting CP, Manchester United, Real Madrid, Juventus, and Al-Nassr, 
    winning numerous league titles and Champions League trophies. At Real Madrid, he became 
    the club’s all-time top scorer with more than 450 goals and won four Champions League titles. 
    He also led Portugal to victory in Euro 2016 and the 2019 Nations League. His work ethic, 
    aerial dominance, and leadership define his playing style, making him one of the most influential 
    athletes in modern sports history."""
]

query = 'tell me about Ronaldo'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
print(cosine_similarity([query_embedding], doc_embeddings))

scores = cosine_similarity([query_embedding], doc_embeddings)[0] # conversion from 2D list to single list

print(list(enumerate(scores)))

print(sorted(list(enumerate(scores)),key=lambda x:x[1]))

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1] # sorting on the basis of the 2nd item of the list  and retriving the last one

print(query)
print(documents[index])
print("similarity score is:", score)



