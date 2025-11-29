import os
from neo4j import GraphDatabase

class MemgraphDriver:
    _instance = None
    _driver = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemgraphDriver, cls).__new__(cls)
            uri = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
            auth = (os.getenv("MEMGRAPH_USER", ""), os.getenv("MEMGRAPH_PASSWORD", ""))
            cls._driver = GraphDatabase.driver(uri, auth=auth)
        return cls._instance

    @property
    def driver(self):
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
