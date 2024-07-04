from json import load
from math import log

import pip
from backend.loader import DocumentLoader
from backend.pipeline import Pipeline
from backend.vectorstore import VectorStore
from configuration import Configuration
from frontend.chat import ChatGUI
from loguru import logger

def main():
    config =Configuration()
    try:
        config.load_config()
        logger.info("Configuration loaded.")
    except Exception as e:
        logger.critical(f"Error loading configuration: {e}")
        logger.info("creating a new configuration file")
        config.create_new_config()
        logger.info("Configuration file created.")

    loader = DocumentLoader(config)
    vector_store = VectorStore(loader, config)
    pipeline = Pipeline()
    gui = ChatGUI(pipeline, vector_store)

    gui.run()

if __name__ == "__main__":
    main()

