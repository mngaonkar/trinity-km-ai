import json
from enum import Enum
from logging import config


from sympy import im
import constants

class VectorStoreStatus(Enum):
    READY = 1
    NOT_INTIALIZED = 2
    ERROR = 3

class Configuration(object):
    """Class for storing application configuration."""
    def __init__(self, config_file=constants.DEFAULT_CONFIF_FILE):
        self.config_file = config_file
        self_config = None

    def load_config(self):
        self._config = json.load(open(self.config_file))

    def save_increamental_config(self, key, value):
        self._config[key] = value
        json.dump(self._config, open(self.config_file, "w"))

    def get_vector_store_config(self):
        return VectorStoreConfig(self)
    
    def get_config(self):
        return self._config
    
    def create_new_config(self):
        self._config = {"vector_store": {"status": VectorStoreStatus.NOT_INTIALIZED.value}}
        json.dump(self._config, open(self.config_file, "w"))

class VectorStoreConfig():
    """Class for storing vector store configuration."""
    def __init__(self, config_obj: Configuration) -> None:
        self._config_obj = config_obj

    def get_vector_store_status(self):
        return self._config_obj._config["vector_store"]["status"]
    
    def set_vector_store_status(self, status: VectorStoreStatus):
        self._config_obj._config["vector_store"]["status"] = status.value
        return self

    def get_config(self):
        return self._config_obj._config["vector_store"]
    
    def save_config(self):
        self._config_obj.save_increamental_config("vector_store", self._config_obj._config["vector_store"])
    
