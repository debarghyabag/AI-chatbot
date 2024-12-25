import argparse
from langchain.vectorstores.chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    print("Welcome! You can start the conversation. Type 'exit' to end the conversation.")
    
    while True:
        # Get user input
        user_input = input("You: ")

        # Exit condition for the conversation
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Process the query and get the response
        query_rag(user_input)

def query_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)

    # Extract relevant document content
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Prepare the prompt for the LLM
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Query the Ollama model (or your chosen LLM model)
    model = Ollama(model="phi3.5")  # You can change the model name as needed
    response_text = model.invoke(prompt)

    # Collect document sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # Format the response with sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
