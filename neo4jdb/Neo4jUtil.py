from py2neo import Graph, NodeMatcher, RelationshipMatcher

from config_loader import config_loader


class NeoUtil:

    def __init__(self):
        cfg = config_loader.get_config()['neo4j']
        uri = cfg['uri']
        username = cfg['username']
        password = cfg['password']
        self.graph = Graph(uri=uri, auth=(username, password))
        self.matcher = NodeMatcher(self.graph)
        self.relation_matcher = RelationshipMatcher(self.graph)

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(NeoUtil, cls).__new__(cls, *args, **kwargs)
        return cls._instance


if __name__ == '__main__':
    neo4j = NeoUtil()








