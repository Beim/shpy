from py2neo import Graph, NodeMatcher, RelationshipMatcher

from config_loader import config_loader


class NeoUtil:

    def __init__(self, uri, username=None, password=None):
        if username is None or password is None:
            cfg = config_loader.get_config()['neo4j']
            username = cfg['username']
            password = cfg['password']

        self.uri = uri
        self.username = username
        self.password = password
        self.graph = Graph(uri=uri, auth=(username, password))
        self.matcher = NodeMatcher(self.graph)
        self.relation_matcher = RelationshipMatcher(self.graph)

    def reconnect(self):
        self.graph = Graph(uri=self.uri, auth=(self.username, self.password))
        self.matcher = NodeMatcher(self.graph)
        self.relation_matcher = RelationshipMatcher(self.graph)

    # def __new__(cls, *args, **kwargs):
    #     if not hasattr(cls, '_instance'):
    #         cls._instance = super(NeoUtil, cls).__new__(cls, *args, **kwargs)
    #     return cls._instance


# neo_util = NeoUtil()

if __name__ == '__main__':
    neo4j = NeoUtil()








