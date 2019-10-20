from py2neo import Graph, NodeMatcher, RelationshipMatcher

from config_loader import config_loader


class NeoUtil:

    def __init__(self, uri=None, username=None, password=None):
        cfg = config_loader.get_config()['neo4j']
        if uri is None:
            uri = cfg['uri']
        if username is None:
            username = cfg['username']
        if password is None:
            password = cfg['password']
        self.graph = Graph(uri=uri, auth=(username, password))
        self.matcher = NodeMatcher(self.graph)
        self.relation_matcher = RelationshipMatcher(self.graph)

    # def __new__(cls, *args, **kwargs):
    #     if not hasattr(cls, '_instance'):
    #         cls._instance = super(NeoUtil, cls).__new__(cls, *args, **kwargs)
    #     return cls._instance


neo_util = NeoUtil()

if __name__ == '__main__':
    neo4j = NeoUtil()








