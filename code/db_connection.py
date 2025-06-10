import os
import logging
from dotenv import load_dotenv
from paths import OUTPUTS_DIR
from db_start import get_db_collection, embed_documents

logger = logging.getLogger()

def setup_logging():
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "rag_assistant.log"))
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
collection = get_db_collection(collection_name="publications")

def retrieve_relevant_documents(query: str, n_results: int = 5, threshold: float = 0.3) -> list[str]:
    logging.info(f"Retrieving relevant documents for query: {query}")
    relevant_results = {"ids": [], "documents": [], "distances": []}
    logging.info("Embedding query...")
    query_embedding = embed_documents([query])[0]
    logging.info("Querying collection...")
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results, include=["documents", "distances"])
    logging.info("Filtering results...")
    keep_item = [distance < threshold for distance in results["distances"][0]]
    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])
    return relevant_results["documents"]

