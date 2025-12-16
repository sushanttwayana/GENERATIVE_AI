#### Complete chatbot for youtube video 
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

load_dotenv()

# Set up the Groq API key (assuming it's set in the environment)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# LLM setup
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

# Prompt template
prompt = PromptTemplate(
    template="""You are an expert assistant that answers questions about this YouTube video transcript. 

CONTEXT FROM TRANSCRIPT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer using ONLY the information from the transcript above
2. If the transcript doesn't contain relevant information, say "I don't have enough information from the video to answer that"
3. Keep your answer clear and helpful
4. Reference specific parts of the transcript when possible
5. If asked for your opinion, redirect to what the transcript states

Your answer:""",
    input_variables=['context', 'question']
)

# Function to extract YouTube video ID from various URL formats
def extract_youtube_id(url_or_id):
    """
    Extract YouTube video ID from various URL formats or return the ID if already given.
    
    Supported formats:
    - Direct ID: dQw4w9WgXcQ
    - Standard URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
    - Short URL: https://youtu.be/dQw4w9WgXcQ
    - Embed URL: https://www.youtube.com/embed/dQw4w9WgXcQ
    - Mobile URL: https://m.youtube.com/watch?v=dQw4w9WgXcQ
    - With additional parameters: https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120s
    """
    
    # If it's already an 11-character YouTube ID, return it
    if len(url_or_id) == 11 and all(c.isalnum() or c in "-_" for c in url_or_id):
        return url_or_id
    
    # List of regex patterns to extract YouTube ID
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/e\/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11}))',
        r'(?:youtube\.com\/.*[?&]v=([a-zA-Z0-9_-]{11}))',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    
    # Try parsing as URL and extracting v parameter
    try:
        parsed_url = urlparse(url_or_id)
        if parsed_url.hostname and ('youtube.com' in parsed_url.hostname or 'youtu.be' in parsed_url.hostname):
            # For youtu.be URLs
            if 'youtu.be' in parsed_url.hostname:
                video_id = parsed_url.path.lstrip('/')
                if video_id and len(video_id) == 11 and all(c.isalnum() or c in "-_" for c in video_id):
                    return video_id
            
            # For standard youtube.com URLs
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params:
                video_id = query_params['v'][0]
                if len(video_id) == 11 and all(c.isalnum() or c in "-_" for c in video_id):
                    return video_id
    except:
        pass
    
    return None

# Function to fetch and process transcript
def fetch_transcript(video_id):
    try:
        
        # 1. Create an instance of the API
        api = YouTubeTranscriptApi()
        # 2. Use the .fetch() method on the instance
        fetched_transcript = api.fetch(video_id, languages=["en"])
        # 3. Convert the result to a list of dictionaries
        transcript_list = fetched_transcript.to_raw_data()
        
        # Your existing code to process the list
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
            
        # transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        # transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript, None
    
    except TranscriptsDisabled:
        return None, "Captions are disabled for this video."
    except NoTranscriptFound:
        return None, "No English transcript found (manual or auto)."
    except VideoUnavailable:
        return None, "Video is unavailable."
    except CouldNotRetrieveTranscript as e:
        return None, f"Transcript exists but could not be retrieved: {e}"
    except Exception as e:
        return None, f"Error fetching transcript: {str(e)}"

# Function to build vector store from transcript
def build_vector_store(transcript):
    if not transcript:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.create_documents([transcript])
    embeddings = OllamaEmbeddings(model="gemma:2b")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# Format docs for context
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Chatbot logic with state management
def chatbot(message, history, state):
    if state is None:
        state = {"stage": "ask_id", "video_id": None, "retriever": None}
    
    if state["stage"] == "ask_id":
        # Extract YouTube ID from URL or use the input as ID
        video_id = extract_youtube_id(message.strip())
        
        if video_id:
            transcript, error = fetch_transcript(video_id)
            if error:
                history = history + [{"role": "user", "content": message},
                                   {"role": "assistant", "content": f"{error} Please paste a valid YouTube URL or video ID to proceed."}]
                return "", history, state
            
            vector_store = build_vector_store(transcript)
            if vector_store:
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                state = {"stage": "chat", "video_id": video_id, "retriever": retriever}
                response = f"Great! I've loaded the transcript for video ID: {video_id}. What would you like to ask about it?"
                history = history + [{"role": "user", "content": message},
                                   {"role": "assistant", "content": response}]
                return "", history, state
            else:
                response = "Failed to build vector store. Please try another YouTube URL or video ID."
                history = history + [{"role": "user", "content": message},
                                   {"role": "assistant", "content": response}]
                return "", history, state
        else:
            response = "Please paste a valid YouTube URL or video ID to proceed.\n\nExamples:\n• Full URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ\n• Short URL: https://youtu.be/dQw4w9WgXcQ\n• Just the ID: dQw4w9WgXcQ"
            history = history + [{"role": "user", "content": message},
                               {"role": "assistant", "content": response}]
            return "", history, state
    
    elif state["stage"] == "chat":
        # Build the chain on the fly if needed
        parallel_chain = RunnableParallel({
            'context': state["retriever"] | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        main_chain = parallel_chain | prompt | llm | StrOutputParser()
        try:
            response = main_chain.invoke(message)
        except Exception as e:
            response = f"Error processing question: {str(e)}"
        history = history + [{"role": "user", "content": message},
                           {"role": "assistant", "content": response}]
        return "", history, state

# Gradio interface
with gr.Blocks(title="YouTube Transcript Chatbot") as demo:
    gr.Markdown("# YouTube Transcript RAG Chatbot")
    gr.Markdown("Enter a YouTube URL or video ID to start chatting about its transcript.")
    gr.Markdown("""
    **Accepted formats:**
    - Full URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
    - Short URL: `https://youtu.be/dQw4w9WgXcQ`
    - Embed URL: `https://www.youtube.com/embed/dQw4w9WgXcQ`
    - Just the ID: `dQw4w9WgXcQ`
    """)
    
    chatbot_component = gr.Chatbot(label="Chat", height=500)
    message = gr.Textbox(label="Your Message", placeholder="Paste YouTube URL or video ID, then ask questions about the video...")
    state = gr.State(None)  # Hidden state for tracking stage, video_id, retriever
    
    message.submit(chatbot, inputs=[message, chatbot_component, state], outputs=[message, chatbot_component, state])

# Launch the app
if __name__ == "__main__":
    demo.launch()