import os
import torch
import chromadb
import shutil
from paths import VECTOR_DB_DIR
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import load_publication


def initialize_db(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "publications",
    delete_existing: bool = False,
) -> chromadb.Collection:
    if os.path.exists(persist_directory) and delete_existing:
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_directory)

    try:
        collection = client.get_collection(name=collection_name)
        print(f"Retrieved existing collection: {collection_name}")
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "hnsw:batch_size": 10000},
        )
        print(f"Created new collection: {collection_name}")

    print(f"ChromaDB initialized with persistent storage at: {persist_directory}")

    return collection


def get_db_collection(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "publications",
) -> chromadb.Collection:
    return chromadb.PersistentClient(path=persist_directory).get_collection(
        name=collection_name
    )


def chunk_publication(
    publication: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(publication)


def embed_documents(documents: list[str]) -> list[list[float]]:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    embeddings = model.embed_documents(documents)
    return embeddings


def insert_publications(collection: chromadb.Collection, publications: list[str]):
    next_id = collection.count()
    for publication in publications:
        chunked_publication = chunk_publication(publication)
        embeddings = embed_documents(chunked_publication)
        ids = list(range(next_id, next_id + len(chunked_publication)))
        ids = [f"document_{id}" for id in ids]
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunked_publication,
        )
        next_id += len(chunked_publication)


def main():
    collection = initialize_db(
        persist_directory=VECTOR_DB_DIR,
        collection_name="publications",
        delete_existing=True,
    )
    publications = [load_publication('fumar')]
    insert_publications(collection, publications)
    print(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    main()
