from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import faiss
from sympy import N

class Retriever:
    def create_vector_db(self):
        """Create a vector database with a single text."""
        texts = ["FAISS is a library for efficient similarity search and clustering of dense vectors."]
        db = FAISS.from_texts(texts, GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")) # type: ignore
        return db

    def load_index(self, db):
        """Load the index from disk."""
        db.load_local("faiss_index")
        return db
    
    def store_documents(self, db, doc_location):
        """Load documents from a directory into a vector database."""
        loader = PyPDFLoader(doc_location)
        document = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        docs = text_splitter.split_documents(document)

        db.add_documents(docs)

        print(db.index.ntotal)

        # save the index to disk
        db.save_local("faiss_index")

        return db


    def query_document(self, db, query):
        """Query a document in the vector database."""
        docs = db.similarity_search(query, k=3)
        return docs

if __name__ == "__main__":
    retriever = Retriever()
    db = retriever.create_vector_db()
    print("loading documents in vector DB...")
    retriever.store_documents(db, "/Users/mahadevgaonkar/Documents/Guides/idrac9-4-00-00-00-ug-new-en-us.pdf")
    print("done.")
    # db.load_local("faiss_index")
    print("qureying documents...")
    query = "How to get sensor data from IPMI?"
    docs = retriever.query_document(db, query)
    print("done.")
    print(docs[0].page_content)

