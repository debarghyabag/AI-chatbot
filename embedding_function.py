from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

def get_embedding_function():
    # Uncomment the embedding model you want to use
    
    # For Bedrock Embeddings (AWS)
   
    # For Ollama Embeddings (Local)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    return embeddings
