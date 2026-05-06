from langchain_huggingface import HuggingFaceEmbeddings
import openai
from typing import Literal
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from utils.logger import get_logger
from langchain_core.vectorstores import FAISS


logger = get_logger(__name__)

def get_embedding_model(model_provider:Literal['huggingface', 'openai'] = 'huggingface'):
    """Factory function to get the appropriate embedding model based on configuration."""
    # For simplicity, we are using HuggingFaceEmbeddings here. You can extend this to support multiple providers.
    if model_provider == 'huggingface':
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif model_provider == 'openai':
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        raise ValueError("Unsupported model provider")
    
def create_vector_store(chunks,embedding_model, collection_name:str = "default_collection"):
    """Create a vector store using the specified embedding model."""
    # For simplicity, we are using FAISS here. You can extend this to support 
    # multiple vector store implementations.
    logger.info(f"Creating vector store with collection name: {collection_name}")
    try:
        store = FAISS.from_texts(chunks, embedding_model)
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise
    return store  # Changed from return None to return store


def retrieve(store,query,k=2):
    """Retrieve relevant documents from the vector store based on the query."""
    logger.info(f"Retrieving top {k} results for query: {query}")
    try:
        results = store.similarity_search(query, k=k)
        logger.debug(f"Retrieved results: {results}")
        return [doc.page_content for doc in results]
    except Exception as e:
        logger.error(f"Error retrieving from vector store: {e}")
        return []
    
def retrieve_with_score(store,query,k=2):
    """Retrieve relevant documents from the vector store based on the query along with their similarity scores."""
    logger.info(f"Retrieving top {k} results with scores for query: {query}")
    try:
        results = store.similarity_search_with_score(query, k=k)
        logger.debug(f"Retrieved results with scores: {results}")
        return [{
            'text':doc.page_content,
            'score':score
        } for doc, score in results]
    except Exception as e:
        logger.error(f"Error retrieving from vector store: {e}")
        return []