from neo4jdb.Neo4jUtil import NeoUtil
from py2neo.data import Node, Relationship



class KsDbService:

    neo4j = NeoUtil()

    def __new__(cls, *args, **kwargs):
        """
        单例
        :param args:
        :param kwargs:
        :return:
        """
        if not hasattr(cls, '_instance'):
            cls._instance = super(KsDbService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def get_related_relationships(self, keyword):
        """
        获得与keyword 相关的relationship (triple)
        relationship 的document 属性类型为list，存储了两端节点所有属性的关键字和边的关键字
        :param keyword:
        :return: list
        """
        res = self.neo4j.relation_matcher.match().where("'%s' in _.document" % keyword)
        return list(res)

    def get_related_relationships_count(self, keyword=None, r_type=None):
        """
        获得与keyword 相关的，类型为r_type 的relationship 的数量
        若未指定keyword，则无该限制
        若未指定r_type，则无该限制
        relationship 的document 属性类型为list，存储了两端节点所有属性的关键字和边的关键字
        :param keyword: string
        :param r_type: type of Relationship
        :return: number
        """
        res = self.neo4j.relation_matcher.match(r_type=r_type)
        if keyword is not None:
            res = res.where("'%s' in _.document" % keyword)
        return len(res)

    def get_relationship_type_count(self):
        """
        获取relationship 的type 种类数量
        :return: list
        """
        res = list(self.neo4j.graph.run("match (n)-[r]->(m) return type(r) as type, count(*) as count"))
        return len(res)

    def get_relationship_types(self):
        """
        获取relationship 的所有type
        :return: list
        """
        res = list(self.neo4j.graph.run("match (n)-[r]->(m) return type(r) as type, count(*) as count"))
        types = []
        for item in res:
            types.append(item['type'])
        return types

    def _mark_relationship(self, rel):
        """
        标记边，两端节点和边的关键词标记到rel['document']
        :param rel:
        :return:
        """
        from keywordSearch.KeywordSearchEngine import KeywordSearchEngine
        print(rel)
        start_prop = dict(rel.start_node)
        end_prop = dict(rel.end_node)
        document = set()
        document.update(list(rel.start_node.labels))
        document.update(list(rel.end_node.labels))
        document.update(start_prop['名'])
        document.update(end_prop['名'])
        for prop in [start_prop, end_prop]:
            for k in prop:
                v = prop[k]
                if type(v) is list:
                    v = ' '.join(v)
                document.update(KeywordSearchEngine._cut_words(v))
        rel['document'] = list(document)
        self.neo4j.graph.push(rel)

    def mark_rels(self):
        rels = self.neo4j.relation_matcher.match()
        idx = 0
        for rel in rels:
            idx += 1
            self._mark_relationship(rel)
        print(idx)


if __name__ == '__main__':
    service = KsDbService()
    rel = service.neo4j.relation_matcher[0]
    service.mark_rels()
    pass
