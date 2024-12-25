import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "docs"


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    # If the reset flag is provided, clear the database
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create or update the Chroma database
    documents = load_documents()
    chunks = split_documents(documents)
    
    # Uncomment the following line to visualize chunks
    visualize_chunks(chunks)
    
    add_to_chroma(chunks)


def load_documents():
    """Load all PDFs from the DATA_PATH directory."""
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Each chunk will be 800 characters
        chunk_overlap=80,  # Each chunk overlaps by 80 characters with the next
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    """Generate unique chunk IDs based on document source and page."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the chunk index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Generate a unique ID for this chunk
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add the chunk ID to the metadata
        chunk.metadata["id"] = chunk_id

    return chunks


def visualize_chunks(chunks):
    """Print chunks and their metadata for visualization."""
    print("\n✨ Visualizing Chunks")
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown Source")
        page = chunk.metadata.get("page", "Unknown Page")
        chunk_id = chunk.metadata.get("id", "Unknown ID")
        print(f"Source: {source}")
        print(f"Page: {page}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Content: {chunk.page_content}\n{'-' * 80}")


def clear_database():
    """Clear the Chroma database by deleting the CHROMA_PATH directory."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Database cleared.")


if __name__ == "__main__":
    main()
