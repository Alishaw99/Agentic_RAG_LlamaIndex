import json
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.llms.groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
api_key = os.getenv('API_KEY')
documents = SimpleDirectoryReader("data").load_data()

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm =Groq(model="mixtral-8x7b-32768", api_key='API_KEY')

index = VectorStoreIndex.from_documents(
    documents,
)
query_engine = index.as_query_engine()
response = query_engine.query("What did the magie do growing up?")
print(response)

# import os
# import json
# from llama_index.core.tools import QueryEngineTool, ToolMetadata
# from llama_index.core.agent import ReActAgent
# from llama_index.llms.ollama import Ollama
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.groq import Groq
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()
# api_key = os.getenv('OPENAI_API_KEY')

# # Verify that the API key is loaded
# if not api_key:
#     raise ValueError("API key not found. Please ensure it is set in the .env file.")

# # Load documents from the specified directory
# documents = SimpleDirectoryReader("data").load_data()

# # Initialize embedding model settings
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# Settings.embed_model = embed_model

# # Initialize LLM settings
# llm = Groq(model="mixtral-8x7b-32768", api_key=api_key)
# Settings.llm = llm

# # Create index from documents
# index = VectorStoreIndex.from_documents(documents)

# # Create query engine from index
# query_engine = index.as_query_engine()

# # Execute query
# response = query_engine.query("What did the magie do growing up?")
# print(response)
