from venv import logger
from langchain_milvus import Milvus
from langchain_community.embeddings import GPT4AllEmbeddings
import constants

class Database():
    def __init__(self, db_name = "milvus_demo.db", collection_name = "milvus_demo"):
        """Create a vector database with a single text."""
        self.db_name = db_name
        self.collection_name = collection_name
        self.embeddings = GPT4AllEmbeddings(model_name=constants.EMBEDDING_MODEL) # type: ignore
        

    def create_database(self):
        self.vector_db = Milvus.from_texts(
            texts=[" "],
            embedding=self.embeddings,
            connection_args={
                "uri": self.db_name,
                "collection_name": self.collection_name,},
            drop_old=True,  # Drop the old Milvus collection if it exists
    )   

    def load_database(self):
        self.vector_db = Milvus.from_texts(
            texts=[" "],
            embedding=self.embeddings,
            connection_args={
                "uri": self.db_name,
                "collection_name": self.collection_name,},
            drop_old=False,  # Keep the old Milvus collection if it exists
        )

    def store_documents(self, document):
        """Store a document in the vector database."""
        self.vector_db.add_documents(document)

    def query_document(self, query):
        """Query a document in the vector database."""
        logger.info(f"Querying document with query: {query}")
        docs = self.vector_db.similarity_search_with_score(query, k=constants.NO_DOCS_PER_QUERY)
        logger.info(f"Found {len(docs)} documents.")
        return docs
