from langchain_milvus import Milvus
from langchain_community.embeddings import GPT4AllEmbeddings
import constants

class Database():
    def __init__(self, db_name = "milvus_demo.db", collection_name = "milvus_demo"):
        """Create a vector database with a single text."""
        self.embeddings = GPT4AllEmbeddings(model_name=constants.EMBEDDING_MODEL) # type: ignore
        self.vectorstore = Milvus.from_texts(
            texts=[" "],
            embedding=self.embeddings,
            connection_args={
                "uri": db_name,
                "collection_name": collection_name,},
            drop_old=True,  # Drop the old Milvus collection if it exists
    )   

    def store_document(self, document):
        """Store a document in the vector database."""
        self.vectorstore.add_documents(document)

    def query_document(self, query):
        """Query a document in the vector database."""
        docs = self.vectorstore.similarity_search(query, k=constants.NO_DOCS_PER_QUERY)
        return docs
