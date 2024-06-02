from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
import constants

class DocumentLoader():
    """Storer class for storing data."""
    def load_web_document(self, url):
        """Load a document from a URL."""
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=constants.CHUNK_SIZE, 
                                                       chunk_overlap=constants.CHUNK_OVERLAP)
        docs = text_splitter.split_documents(document)

        return docs
    
    def load_pdf_document(self, pdf_location):
        """Load a document from a PDF."""
        loader = PyPDFLoader(pdf_location)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=constants.CHUNK_SIZE, 
                                                       chunk_overlap=constants.CHUNK_OVERLAP)
        docs = text_splitter.split_documents(document)

        return docs