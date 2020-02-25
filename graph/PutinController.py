from neo4jdb.Neo4jUtil import NeoUtil
from py2neo.data import Node, Relationship
from neo4jdb.ExSubgraph import ExSubgraph
from graph.SCMatcher import SCMatcher
from graph.GraphService import GraphService
from Utils import deprecated
import jieba
from neo4j.exceptions import ServiceUnavailable

KEY_NODE = 'node'
KEY_REL = 'rel'

class NodeLabelCache:
    """


    Attributes:
        neo_util: Neo4jUtil
        node_label_map: {
            [label_name]: {
                hit: int,
                nodes: set()
            },
            ...
        }
    """

    HIT = 'hit'
    NODES = 'nodes'
    MIN_HIT = -100

    step = 2000
    node_label_map = {}
    updated_labels = set()

    def __init__(self, neo_util):
        self.neo_util = neo_util

    def fetch_node_sets(self, labels):
        """
        从缓存获取labels 对应的nodes
        :param labels:
        :return:
        """
        node_sets = set()
        for label in labels:
            node_sets.update(self._fetch_node_set(label))
        return node_sets

    def _fetch_node_set(self, label):
        """
        从缓存获取nodes，若不存在与缓存则从db 获取
        :param label:
        :return: set
        """
        if label not in self.node_label_map:
            self._load_node_set_from_db(label)
        node_set = self.node_label_map[label][self.NODES]
        self.node_label_map[label][self.HIT] += 1
        return node_set

    def _clear_cache(self):
        """
        清除缓存
        所有label 的hit -= 1
        hit < MIN_HIT 的label 清除，即最近-MIN_HIT次未命中
        :return: None
        """
        for label in self.node_label_map:
            node_label_obj = self.node_label_map[label]
            node_label_obj[self.HIT] -= 1
            if node_label_obj[self.HIT] < self.MIN_HIT:
                del self.node_label_map[label]

    def _update_cached_node_set(self, label, updated_nodes: dict):
        """
        更新label 对应的node 缓存（在添加了新节点以后执行）
        :param label:
        :param updated_nodes: 更新的节点
        :return: None
        """
        if label not in self.node_label_map:
            return
        node_label_obj = self.node_label_map[label]
        # count 记录当前缓存了该标签的节点数量，更新缓存的时候根据id顺序跳过这些节点，只更新后面添加的节点（但是这样会导致只更新到新加节点，没有更新到改变的已有节点）
        count = len(node_label_obj[self.NODES])
        new_added_node_set = self._read_node_set_from_db_idx(label, count)
        node_label_obj[self.NODES].update(new_added_node_set)
        # 重新获取已有（但有更新）的节点，解决前面跳过count 个而漏掉更新节点的问题
        if label not in updated_nodes:
            return
        updated_node_id_list = list(updated_nodes[label])
        for identity in updated_node_id_list:
            up_to_date_node = self.neo_util.matcher.get(identity)
            node_label_obj[self.NODES].add(up_to_date_node)

    def add_updated_labels(self, labels):
        """
        添加更新过的label
        :param label:
        :return:
        """
        self.updated_labels.update(labels)

    def _clear_updated_label(self):
        self.updated_labels.clear()

    def update_cache(self, updated_nodes):
        """
        更新缓存，根据已记录的更新的标签
        :return:
        """
        for label in self.updated_labels:
            self._update_cached_node_set(label, updated_nodes)
        self._clear_updated_label()
        try:
            self._clear_cache()
        except Exception as e:
            print(e)

    def _read_node_set_from_db_idx(self, label, idx=0):
        """
        从neo4jdb 读入对应标签的所有node，以idx 作为起始
        :param label:
        :param idx:
        :return: set
        """
        node_set = set()
        step = self.step
        cursor = self.neo_util.matcher.match(label)
        total = len(cursor)
        while idx < total:
            r = cursor.limit(step).skip(idx)
            node_set.update(r)
            idx += step
        return node_set

    def _load_node_set_from_db(self, label):
        """
        从neo4jdb 读入对应标签的所有node
        :param label:
        :return: None
        """
        node_set = set()
        step = self.step
        cursor = self.neo_util.matcher.match(label)
        total = len(cursor)
        idx = 0
        while idx < total:
            r = cursor.limit(step).skip(idx)
            node_set.update(r)
            idx += step
        self.node_label_map[label] = {
            self.HIT: 0,
            self.NODES: node_set
        }


class PutinController:
    """
    添加entity

    """

    # entity 必须的属性
    LABEL = 'label'
    # entity 必须的属性
    NAME = "name"

    def __init__(self):
        self.neo_util_map = {}
        self.node_label_cache_map = {}

    def post(self, gid: int, uri: str, entity_arr: list):
        """
        添加entity 到db
        :param gid: 1
        :param uri: bolt://localhost:7687
        :param entity_arr:
        :return:
        """
        if gid not in self.neo_util_map:
            self.neo_util_map[gid] = NeoUtil(uri)
            self.node_label_cache_map[gid] = NodeLabelCache(self.neo_util_map[gid])
        gq = self.trans_entity_arr_to_graph(gid, entity_arr)
        gd_nodes = self.node_label_cache(gid).fetch_node_sets(gq.subgraph.labels)
        sc_matcher = SCMatcher(gd_nodes, gq, rs=0, K=3, st=0.8, neo_util=self.neo_util_map[gid])
        match = sc_matcher.run()
        updated_nodes = self.get_updated_nodes(match)
        # 更新图
        ng = self.update_data_graph(gid, gq, match)

        self.neo_util(gid).reconnect()
        # 补充已有节点的属性
        print('graph.push...')
        try:
            self.neo_util(gid).graph.push(ng.subgraph)
        except ServiceUnavailable:
            self.neo_util(gid).reconnect()
            self.neo_util(gid).graph.push(ng.subgraph)

        # 添加新属性与关系
        print('graph.create...')
        try:
            self.neo_util(gid).graph.create(ng.subgraph)
        except ServiceUnavailable:
            self.neo_util(gid).reconnect()
            self.neo_util(gid).graph.create(ng.subgraph)

        # 更新缓存
        print('update_cache...')
        self.node_label_cache(gid).update_cache(updated_nodes)

    def neo_util(self, gid: int) -> NeoUtil:
        return self.neo_util_map[gid]

    def node_label_cache(self, gid) -> NodeLabelCache:
        return self.node_label_cache_map[gid]

    @deprecated
    def _print_match(self, match):
        """
        用于debug，输出匹配节点
        :param match:
        :return:
        """
        for qn in match:
            dn = match[qn]
            print('q ', qn.labels, dict(qn))
            if dn is None:
                print('None\n')
            else:
                print('d[%s] ' % dn.identity, dn.labels, dict(dn), '\n')

    def get_updated_nodes(self, match):
        """
        根据匹配表，获取更新的已有节点
        :param match:
        :return: {'label_xxx': {101, 102, ...}} 标签：节点id集合
        """
        updated_nodes = {}
        for node_q in match:
            node_d = match[node_q]
            if node_d is None:
                continue
            labels = node_d.labels
            for label in labels:
                if label not in updated_nodes:
                    updated_nodes[label] = set()
                updated_nodes[label].add(node_d.identity)
        return updated_nodes


    def update_data_graph(self, gid: int, gq, match):
        """
        根据匹配关系更新图数据库，返回改动后的数据子图
        :param gq: ExSubgraph 查询子图
        :param match: dict<py2neo.data.Node, py2neo.data.Node> 匹配关系
        :return: ExSubgraph
        """
        ng = ExSubgraph()
        # 处理node
        for node_q in gq.subgraph.nodes:
            node_d = match[node_q]
            if node_d is None:
                match[node_q] = node_q
                ng.add_node(node_q)
                self.node_label_cache(gid).add_updated_labels(node_q.labels)
            else:
                prop_q = dict(node_q)
                for k in prop_q:
                    # 若node_q 的属性是list，则进行合并
                    if type(prop_q[k]) is list:
                        if type(node_d[k]) is list:
                            s = set(node_d[k])
                        else:
                            s = set([node_d[k]])
                        s.update(prop_q[k])
                        node_d[k] = list(s)
                    # 若node_q 的属性是single value，则覆盖
                    else:
                        node_d[k] = prop_q[k]
                ng.add_node(node_d)
                self.node_label_cache(gid).add_updated_labels(node_d.labels)
        # 处理relationship
        for rel in gq.subgraph.relationships:
            start_node_q = rel.start_node
            end_node_q = rel.end_node
            start_node_d = match[start_node_q]
            end_node_d = match[end_node_q]
            rel_d = type(rel)(start_node_d, end_node_d)
            ng.add_relationship(rel_d)
        return ng

    @deprecated
    def _mark_rel(self, rel):
        """
        标记边，两端节点和边的关键词标记到rel['document']
        :param rel: py2neo.data.Relationship
        :return: None
        """
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
                document.update(self._cut_words(v))
        r_document = list(document)
        # 重复添加一次，提高属性的权重
        r_document.extend(list(rel.start_node.labels))
        r_document.extend(list(rel.start_node.labels))
        r_document.extend(start_prop['名'])
        r_document.extend(end_prop['名'])
        r_document.extend(self._cut_words(' '.join(start_prop['名'])))
        r_document.extend(self._cut_words(' '.join(end_prop['名'])))
        rel['document'] = r_document

    @staticmethod
    def _cut_words(s):
        """
        分词
        :param s: str
        :return: list<str>
        """
        seg_list = jieba.cut(s)
        words = []
        for seg in seg_list:
            if seg.strip() != '':
                words.append(seg.strip())
        return words

    def trans_neo4jdb_to_nodes(self, labels):
        """
        取出neo4jdb 中所有相关label 的节点
        :param labels: list<str> 相关标签
        :return: list<py2neo.data.Node>
        """
        node_set = set()
        neo4j = self.neoUtil
        step = 2000
        for label in labels:
            cursor = neo4j.matcher.match(label)
            total = len(cursor)
            idx = 0
            while idx < total:
                r = cursor.limit(step).skip(idx)
                node_set.update(r)
                idx += step
        return node_set

    @deprecated
    def trans_neo4jdb_to_graph(self, labels):
        """
        取出所有相关label 的节点与关系，构成子图
        :param labels: list<str> 相关标签
        :return: ExSubgraph
        """
        def label_clause(v, labels):
            clause = ' or '.join(map(lambda label: '%s:%s' % (v, label), labels))
            return '(%s)' % clause
        neo4j = self.neoUtil
        esg = ExSubgraph()

        where_clause = 'WHERE %s' % (label_clause('a', labels))
        query = 'MATCH (a) %s RETURN a' % where_clause
        cursor = neo4j.graph.run(query)
        for record in cursor:
            n = record['a']
            esg.add_node(n)

        where_clause = 'WHERE %s and %s' % (label_clause('a', labels), label_clause('b', labels))
        query = 'MATCH (a)-[r]-(b) %s RETURN r' % where_clause
        cursor = neo4j.graph.run(query)
        for record in cursor:
            rel = record['r']
            esg.add_relationship(rel)
        return esg

    def _create_node(self, entity, schema_controller, exsubgraph):
        """
        创建新node
        若有未确定label 的node，则返回None
        {
            node: Node(),
            rel: {
                rel_name1: Node(),
                ...
            }
        }
        :param entity:
        :param schema_controller:
        :param exsubgraph:
        :return: dict
        """
        class_uri = entity[self.LABEL]
        new_node = Node(class_uri)
        new_rel = {}
        for key in entity:
            if key is self.LABEL:
                pass
            prop_uri = key
            if schema_controller.is_datatype_property(class_uri, prop_uri):
                prop_val = entity[key]
                new_node[prop_uri] = prop_val
            elif schema_controller.is_object_property(class_uri, prop_uri):
                prop_val = entity[key]
                prop_ranges = schema_controller.get_prop_range(class_uri, prop_uri)
                if len(prop_ranges) == 1:
                    prop_label = prop_ranges[0]
                    temp_prop_node = Node(prop_label)
                    temp_prop_node[self.NAME] = prop_val
                    # 若已存在相同label 与属性 的node 则使用该node
                    new_rel[prop_uri] = exsubgraph.get_node_if_exist(temp_prop_node,
                                                                     check_label=True, return_new_node=True)
                elif len(prop_ranges) > 1:  # 有多个range
                    for prop_label in prop_ranges:
                        temp_prop_node = Node(prop_label)
                        temp_prop_node[self.NAME] = prop_val
                        # 若以存在相同label 与属性 的node 则使用该node
                        old_node = exsubgraph.get_node_if_exist(temp_prop_node,
                                                                check_label=True, return_new_node=False)
                        if old_node is not None:
                            new_rel[prop_uri] = old_node

                    pass
                else:  # 未指定range
                    temp_prop_node = Node()
                    temp_prop_node[self.NAME] = prop_val
                    old_node = exsubgraph.get_node_if_exist(temp_prop_node,
                                                            check_label=False, return_new_node=False)
                    if old_node is not None:
                        new_rel[prop_uri] = old_node
        return {
            KEY_NODE: exsubgraph.get_node_if_exist(new_node, check_label=True, return_new_node=True),
            KEY_REL: new_rel
    }

    def trans_entity_arr_to_graph(self, gid: int, entity_arr: list):
        """
        将entity 数组转换为ExSubgraph
        :param entityArr:
        :return: ExSubgraph
        """
        class_relative_uri_list = []
        for entity in entity_arr:
            class_relative_uri_list.append(entity[self.LABEL])
        schema_controller = SchemaController()
        schema_controller.add_class_list(gid, class_relative_uri_list)
        subgraph = ExSubgraph()
        while True:
            has_change = False
            used_entity_arr = []
            for entity in entity_arr:
                node_obj = self._create_node(entity, schema_controller, subgraph)
                if node_obj is None:
                    continue
                has_change = True
                used_entity_arr.append(entity)
                # 添加node
                subgraph.add_node(node_obj[KEY_NODE])
                # 添加relationship
                for rel_name in node_obj[KEY_REL]:
                    related_node = node_obj[KEY_REL][rel_name]
                    new_rel = Relationship(node_obj[KEY_NODE], rel_name, related_node)
                    subgraph.add_relationship(new_rel)
            for entity in used_entity_arr:
                entity_arr.remove(entity)
            if not has_change: 
                break
        return subgraph


class SchemaController:
    """
    schema 格式:
    {
        "Movie": {
            "objectProperty": {},
            "datatypeProperty": {
                "hasDirector": {
                    "domain": [
                        "Movie"
                    ],
                    "range": [
                        "string"
                    ]
                },
                "hasActor": {
                    "domain": [
                        "Movie"
                    ],
                    "range": [
                        "string"
                    ]
                }
            }
        },
        ...
    }
    """
    schema = {}

    def add_class_list(self, gid: int, class_relative_uri_list: list):
        class_info = GraphService.get_class_info(gid, class_relative_uri_list)
        for key in class_info:
            self.schema[key] = class_info[key]

    def has_property(self, class_uri, prop_uri):
        """
        schema 中是否定义了class 对应的prop
        :param class_uri:
        :param prop_uri:
        :return: boolean
        """
        return (prop_uri in self.schema[class_uri][GraphService.DATATYPE_PROPERTY]) or (prop_uri in self.schema[class_uri][GraphService.OBJECT_PROPERTY])

    def is_datatype_property(self, class_uri, prop_uri):
        """
        class 的prop 是否是datatype prop
        :param class_uri:
        :param prop_uri:
        :return: boolean
        """
        return prop_uri in self.schema[class_uri][GraphService.DATATYPE_PROPERTY]

    def is_object_property(self, class_uri, prop_uri):
        """
        class 的prop 是否是object prop
        :param class_uri:
        :param prop_uri:
        :return:
        """
        return prop_uri in self.schema[class_uri][GraphService.OBJECT_PROPERTY]

    def get_prop_range(self, class_uri, prop_uri):
        """
        获取class 的prop 对用的range
        :param class_uri:
        :param prop_uri:
        :return: list
        """
        prop_type = None
        if self.is_datatype_property(class_uri, prop_uri):
            prop_type = GraphService.DATATYPE_PROPERTY
        elif self.is_object_property(class_uri, prop_uri):
            prop_type = GraphService.OBJECT_PROPERTY
        return self.schema[class_uri][prop_type][prop_uri][GraphService.RANGE]







if __name__ == '__main__':
    p = PutinController()
    entity_arr = [
        {
            "label": "Movie",
            "name": "fulian4",
            "hasActor": "xiaoluobote",
            "hasDirector": "fd"
        },
        {
            "label": "Person",
            "name": "fd",
        },
        {
            "label": "Movie",
            "name": "xunlonggaoshou3",
            "hasActor": "fd"
        }
    ]
    # entity_arr = [
    #     {
    #         "label": "Movie",
    #         "name": ["xunlonggaoshou3", "驯龙高手3", "驯龙3"],
    #         "hasActor": "fd"
    #     }
    # ]
    # p.putin(entity_arr)






