# Generative AI with LangChain
A comprehensive repository showcasing end-to-end implementations of Generative AI applications using LangChain framework. This repository covers everything from foundational LLM concepts to advanced RAG systems and AI agents.

## 🎯 Overview
This repository serves as a comprehensive learning resource and reference implementation for building production-ready Generative AI applications. It covers the complete LangChain ecosystem, from basic LLM interactions to complex AI agent systems, providing hands-on examples for each concept.

#### Perfect for:

- AI/ML Engineers looking to master LangChain
- Developers building RAG-based applications
- Teams implementing AI agents and autonomous systems
- Anyone interested in production-grade GenAI solutions

## 📁 Repository Structure

<img width="541" height="358" alt="image" src="https://github.com/user-attachments/assets/32d48c36-a056-4ff9-97a5-93f186fd678a" />

## ✨ Features
#### 🎨 Core Capabilities

- Multiple LLM Integrations: OpenAI, Groq, Ollama, and more
- Advanced RAG Systems: Complete document processing pipelines
- AI Agents: Tool-using autonomous agents with decision-making
- Production-Ready Code: Well-structured, documented implementations
- LCEL Support: Modern LangChain Expression Language patterns
- Vector Database Integration: ChromaDB, FAISS, and more
- Real-World Examples: YouTube chatbot, document Q&A systems

#### 🚀 Highlights

- ✅ Comprehensive coverage of LangChain ecosystem
- ✅ Hands-on Jupyter notebooks for each concept
- ✅ Production-grade code structure
- ✅ Multiple use cases and implementations
- ✅ Best practices and design patterns
- ✅ Ready-to-deploy examples

#### 🔧 Prerequisites
Before you begin, ensure you have the following installed:

* Python 3.8 or higher
* pip (Python package manager)
* Virtual environment tool (venv or conda)
* API Keys for:

    - OpenAI (optional)
    - Groq (optional)
    - Other LLM providers as needed

## 📦 Installation

1. Install Dependencies
   ```bash
    pip install langchain langchain-community langchain-openai
    pip install chromadb faiss-cpu
    pip install python-dotenv
    pip install jupyter notebook
    pip install openai groq
    
    # Additional dependencies
    pip install pypdf unstructured tiktoken
    pip install youtube-transcript-api
   ```

2. Set Up Environment Variables
Create a .env file in the root directory:
   ```bash
    OPENAI_API_KEY=your_openai_api_key_here
    GROQ_API_KEY=your_groq_api_key_here
    LANGCHAIN_API_KEY=your_langchain_api_key_here
    LANGCHAIN_TRACING_V2=true
   ```

# 📚 Module Breakdown

This section provides a structured walkthrough of each LangChain module included in the repository. Each module builds progressively, from foundational concepts to advanced production-ready systems.

---

## 1️⃣ Models
**Location:** `LangChain/1_Models/`

Learn how to work with different types of language models.

### Covered Topics
- LLMs: Traditional completion-based models  
- Chat Models: Conversational models with role-based messages  
- Embedding Models: Vector representations for semantic similarity  

### Key Concepts
- Model initialization and configuration  
- Streaming responses  
- Token management  
- Multi-model comparisons  

---

## 2️⃣ Prompts
**Location:** `LangChain/2_Prompts/`

Master prompt engineering and reusable template design.

### Covered Topics
- Prompt templates with variables  
- Few-shot learning examples  
- Chat prompt templates  
- System, user, and assistant messages  

### Key Concepts
- Dynamic prompt generation  
- Prompt composition  
- Template chaining  
- Prompt optimization techniques  

---

## 3️⃣ Structured Output
**Location:** `LangChain/3_Structured_Output/`

Generate schema-based, type-safe outputs from LLMs.

### Covered Topics
- Pydantic models for output validation  
- JSON schema enforcement  
- Type-safe data extraction  
- Structured data from unstructured text  

### Key Concepts
- Output schema definition  
- Data validation  
- Type hints and annotations  
- Error handling  

---

## 4️⃣ Output Parsers
**Location:** `LangChain/4_Output_Parsers/`

Parse and validate LLM-generated responses.

### Covered Topics
- String parsers  
- JSON parsers  
- List parsers  
- Custom parser implementations  

### Key Concepts
- Response formatting  
- Error recovery  
- Retry logic  
- Parser composition  

---

## 5️⃣ Chains
**Location:** `LangChain/5_Chains/`

Build sequential and conditional processing pipelines.

### Covered Topics
- Simple chains  
- Sequential chains  
- Router chains  
- Conditional chains  

### Key Concepts
- Chain composition  
- Data flow management  
- Error propagation  
- Chain debugging  

---

## 6️⃣ Runnables (LCEL)
**Location:** `LangChain/6_Runnables/`

Explore the modern **LangChain Expression Language (LCEL)**.

### Covered Topics
- Pipe (`|`) operator usage  
- Parallel execution  
- Conditional routing  
- Stream processing  

### Key Concepts
- LCEL syntax  
- Composability  
- Async operations  
- Performance optimization  

---

## 7️⃣ Document Loaders
**Location:** `LangChain/7_Document_Loaders/`

Load documents from multiple data sources.

### Covered Topics
- PDF files  
- Text files  
- Web pages  
- APIs and databases  

### Key Concepts
- Multi-format support  
- Metadata extraction  
- Batch loading  
- Custom loaders  

---

## 8️⃣ Text Splitters
**Location:** `LangChain/8_Text_Splitter/`

Split documents into optimized chunks for retrieval.

### Covered Topics
- Character-based splitting  
- Recursive splitting  
- Token-aware splitting  
- Semantic chunking  

### Key Concepts
- Chunk size optimization  
- Overlap strategies  
- Context preservation  
- Performance tuning  

---

## 9️⃣ Vector Databases
**Location:** `LangChain/9_Vector_Database/`

Store and retrieve vector embeddings efficiently.

### Covered Topics
- ChromaDB integration  
- FAISS implementation  
- Pinecone setup  
- Vector similarity search  

### Key Concepts
- Embedding storage  
- Similarity metrics  
- Index optimization  
- Hybrid search  

---

## 🔟 Retrievers
**Location:** `LangChain/10_Retrievers/`

Advanced document retrieval strategies.

### Covered Topics
- Vector store retrievers  
- Multi-query retrievers  
- Contextual compression  
- Ensemble retrievers  

### Key Concepts
- Retrieval strategies  
- Reranking  
- MMR (Maximal Marginal Relevance)  
- Performance optimization  

---

## 1️⃣1️⃣ RAG (Retrieval-Augmented Generation)
**Location:** `LangChain/11_RAG/`

End-to-end RAG system implementations.

### Covered Topics
- Basic RAG pipelines  
- Document Q&A systems  
- YouTube Chatbot (video transcript Q&A)  
- Context-aware responses  

### Key Projects
- `rag_using_langchain.ipynb` – Complete RAG tutorial  
- `youtube_chatbot.py` – Production-ready YouTube Q&A bot  

### Key Concepts
- End-to-end RAG architecture  
- Context retrieval  
- Answer generation  
- Citation and source tracking  

---

## 1️⃣2️⃣ Tools
**Location:** `LangChain/12_Tools/`

Extend LLM capabilities using external tools.

### Covered Topics
- Calculator tools  
- Search tools  
- API integration tools  
- Custom tool creation  

### Key Concepts
- Tool definition  
- Tool selection logic  
- Error handling  
- Tool composition  

---

## 1️⃣3️⃣ AI Agents
**Location:** `LangChain/13_AI_AGENT_LangChain/`

Build autonomous, tool-using AI agents.

### Covered Topics
- ReAct agents  
- Tool-using agents  
- Multi-step reasoning  
- Agent executors  

### Key Concepts
- Agent types  
- Tool binding  
- Memory management  
- Agent orchestration  

---

# 🎯 Key Projects

## 1️⃣ YouTube Chatbot
**Location:** `LangChain/11_RAG/youtube_chatbot.py`

A production-ready chatbot that:
- Fetches YouTube video transcripts  
- Processes and indexes content  
- Answers questions about video content  
- Provides source citations  

### Features
- Automatic transcript extraction  
- Semantic search over video content  
- Context-aware responses  
- Streamlit UI integration  

---

## 2️⃣ RAG Pipeline
**Location:** `LangChain/11_RAG/rag_using_langchain.ipynb`

Complete implementation covering:
- Document loading and preprocessing  
- Embedding generation  
- Vector store setup  
- Retrieval and generation  

---

## 3️⃣ AI Agents
**Location:** `LangChain/13_AI_AGENT_LangChain/`

Autonomous agents featuring:
- Tool selection and usage  
- Multi-step reasoning  
- Self-correction capabilities  
- Task planning and execution  

---

# 🛠️ Technologies Used

- **LangChain** – Core framework  
- **OpenAI API** – GPT models  
- **Groq** – High-speed LLM inference  
- **ChromaDB** – Vector database  
- **FAISS** – Similarity search  
- **Python** – Programming language  
- **Jupyter Notebook** – Interactive experimentation  
- **Pydantic** – Data validation  
- **python-dotenv** – Environment variable management  

