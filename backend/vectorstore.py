from unittest import loader
from loguru import logger
from backend.database import Database
import constants
from backend.loader import DocumentLoader
from configuration import Configuration, VectorStoreStatus

class VectorStore():
    """Class for storing vectors."""
    vector_db_initialized = False

    def __init__(self, loader: DocumentLoader, config: Configuration) -> None:
        self.loader = loader
        self.config = config
        self.database = Database()
        if config.get_vector_store_config().get_vector_store_status() == VectorStoreStatus.READY:
            self.vector_db_initialized = True   
        
    def init_vectorstore(self, doc_location=constants.DOCS_LOCATION):
        # webpage = constants.DOC_URL
       
        # for i in range(1, 6):
        #     url = webpage + str(i)
        #     docs = self.loader.load_web_document(url)
        #     self.chat.db.vectorstore.add_documents(docs)
        
        # webpage = "https://gutenberg.org/cache/epub/1661/pg1661.txt"
        # docs = self.loader.load_web_document(webpage)
        # self.chat.db.vectorstore.add_documents(docs)

        self.config.load_config()
        logger.debug(self.config.get_config())
        logger.debug(self.config.get_vector_store_config().get_vector_store_status())
        logger.debug(VectorStoreStatus.READY.value)

        self.vector_db_initialized = self.config.get_vector_store_config().get_vector_store_status() == VectorStoreStatus.READY.value
        if not self.vector_db_initialized:
            logger.info("vector store not initialized, initializing...")
            self.database.create_database()
            docs = self.loader.load_documents_from_directory(doc_location)
            logger.info(f"Adding {len(docs)} documents to vector store")
            self.database.store_documents(docs)
            logger.info("Done.")
            self.config.get_vector_store_config().set_vector_store_status(VectorStoreStatus.READY).save_config()  
        else:
            self.database.load_database()
            logger.info("Vector store already initialized.")