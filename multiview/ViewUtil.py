import json
from py2neo.data import Node, Relationship
import numpy as np
from functools import cmp_to_key

from neo4jdb.Neo4jUtil import NeoUtil


with open('./conf/neo4jConf.json', 'r') as f:
    neo4j_conf = json.load(f)
    source_neo4j_conf = neo4j_conf['source']
    target_neo4j_conf = neo4j_conf['target']
source_neo4j = NeoUtil(source_neo4j_conf['url'], source_neo4j_conf['username'], source_neo4j_conf['password'])
target_neo4j = NeoUtil(target_neo4j_conf['url'], target_neo4j_conf['username'], target_neo4j_conf['password'])


class ViewConfig:

    def __init__(self, view_conf):
        self.labels = view_conf['labels']
        self.rels = view_conf['rels']
        self.vrels = view_conf['vrels']
        self._sort_vrels(self.vrels)

    def _sort_vrels(self, vrels):
        """
        根据依赖重新排序虚拟关系vrels
        :param vrels:
        :return:
        """
        all_rel_types = set()
        for vrel in vrels:
            all_rel_types.add(vrel['type'])
            all_rel_types.add(vrel['rels'][0]['type'])
            all_rel_types.add(vrel['rels'][1]['type'])
        # 0: HAS_ACTOR, 1: HAS_DIRECTOR, 2: COOP
        all_rel_types = list(all_rel_types)
        # HAS_ACTOR: 0, HAS_DIRECTOR: 1, COOP: 2
        all_rel_type2id_map = {}
        for i in range(len(all_rel_types)):
            all_rel_type2id_map[all_rel_types[i]] = i
        # if COOP -> HAS_ACTOR and id(COOP) = 2 and id(HAS_ACTOR) = 0
        # then dependency_matrix[2][0] = 1
        dependency_matrix = np.zeros((len(all_rel_types), len(all_rel_types)))
        for vrel in vrels:
            vrel_type_id = all_rel_type2id_map[vrel['type']]
            subrel1_type_id = all_rel_type2id_map[vrel['rels'][0]['type']]
            subrel2_type_id = all_rel_type2id_map[vrel['rels'][1]['type']]
            dependency_matrix[vrel_type_id][subrel1_type_id] = 1
            dependency_matrix[vrel_type_id][subrel2_type_id] = 1
        degree_list = np.zeros(len(all_rel_types))
        for prev_id in range(len(dependency_matrix)):
            degree_list += dependency_matrix[prev_id]
        zero_indegree_queue = []
        for id in range(len(degree_list)):
            indegree = degree_list[id]
            if indegree == 0:
                zero_indegree_queue.append(id)
        # {HAS_ACTOR: 0, HAS_DIRECTOR: 1, COOP: 2}
        priority_type_index_map = {}
        priority_count = 0
        while len(zero_indegree_queue) > 0:
            rel_id = zero_indegree_queue.pop(0)
            rel_dependency_list = dependency_matrix[rel_id]
            for dependency_id in range(len(rel_dependency_list)):
                # rel_id -> dependency_id
                if rel_dependency_list[dependency_id] == 1:
                    degree_list[dependency_id] -= 1
                    if degree_list[dependency_id] == 0:
                        zero_indegree_queue.append(dependency_id)
            priority_type_index_map[all_rel_types[rel_id]] = priority_count
            priority_count += 1
        def comp_func(x, y):
            vrel1_index = priority_type_index_map[x['type']]
            vrel2_index = priority_type_index_map[y['type']]
            return vrel2_index - vrel1_index
        vrels.sort(key=cmp_to_key(comp_func))


class ViewUtil:

    REAL_ID = 'realId'  # 视图中节点保存原节点id的属性key
    START_ID = 'start_id'
    END_ID = 'end_id'

    def extract_nodes_by_label(self, neo_util, label):
        """
        抽取指定label 的所有节点
        :param label: String
        :return: set<Node>
        """
        node_set = set()
        step = 2000  # cursor 遍历的步长
        cursor = neo_util.matcher.match(label)
        total = len(cursor)
        idx = 0
        while idx < total:
            r = cursor.limit(step).skip(idx)
            node_set.update(r)
            idx += step
        return node_set

    def extract_rels(self, neo_util, label, start_types, end_types):
        """
        抽取出指定label 的所有关系
        :param label: String
        :return: set<Record<start_id, end_id>>
        """
        rel_set = set()
        step = 2000
        cypher_match_statement = '(n1)-[:%s]->(n2)' % (label)

        def cypher_where_statement(start_types, end_types):
            start_stmt_list = []
            end_stmt_list = []
            for start_type in start_types:
                start_stmt_list.append('n1:`%s`' % start_type)
            start_stmt = '(%s)' % (' or '.join(start_stmt_list))
            for end_type in end_types:
                end_stmt_list.append('n2:`%s`' % end_type)
            end_stmt = '(%s)' % (' or '.join(end_stmt_list))
            return '%s and %s' % (start_stmt, end_stmt)

        cypher_where_stmt = cypher_where_statement(start_types, end_types)
        cypher_query_count = 'match %s where %s return count(*) as count' % (cypher_match_statement, cypher_where_stmt)

        def cypher_query(skip, limit):
            return 'match %s where %s return id(n1) as %s, id(n2) as %s skip %s limit %s' \
                   % (cypher_match_statement, cypher_where_stmt, self.START_ID, self.END_ID, skip, limit)
        total = int(neo_util.graph.run(cypher_query_count).next()['count'])
        idx = 0
        while idx < total:
            r = neo_util.graph.run(cypher_query(skip=idx, limit=step))
            rel_set.update(r)
            idx += step
        return rel_set

    def extract_vrels(self, neo_util, vrel_type, subrel1_conf, subrel2_conf):
        """
        抽取出定义的三角模式的虚拟关系vrel
        :return: set<Record<start_id, end_id, rel>>
        """
        vrel_set = set()
        cypher_match_statements = []
        if subrel1_conf['inverse']:
            # cypher_match_statements.append(
            #     '(n0:`%s`)<-[:%s]-(n1:`%s`)' % (subrel1_conf['startLabel'], subrel1_conf['type'], subrel1_conf['endLabel']))
            cypher_match_statements.append(
                '(n1:`%s`)-[:%s]->(n0:`%s`)' % (subrel1_conf['startLabel'], subrel1_conf['type'], subrel1_conf['endLabel']))
        else:
            cypher_match_statements.append(
                '(n0:`%s`)-[:%s]->(n1:`%s`)' % (subrel1_conf['startLabel'], subrel1_conf['type'], subrel1_conf['endLabel']))
        if subrel2_conf['inverse']:
            # cypher_match_statements.append(
            #     '(n1:`%s`)<-[:%s]-(n2:`%s`)' % (subrel2_conf['startLabel'], subrel2_conf['type'], subrel2_conf['endLabel']))
            cypher_match_statements.append(
                '(n2:`%s`)-[:%s]->(n1:`%s`)' % (subrel2_conf['startLabel'], subrel2_conf['type'], subrel2_conf['endLabel']))
        else:
            cypher_match_statements.append(
                '(n1:`%s`)-[:%s]->(n2:`%s`)' % (subrel2_conf['startLabel'], subrel2_conf['type'], subrel2_conf['endLabel']))
        cypher_match_statement = ', '.join(cypher_match_statements)
        cypher_query_count = 'match %s return count(*) as count' % cypher_match_statement

        def cypher_query(skip, limit):
            return 'match %s return n0.%s as %s, n2.%s as %s, "%s" as rel skip %s limit %s' \
                   % (cypher_match_statement, self.REAL_ID, self.START_ID, self.REAL_ID, self.END_ID, vrel_type, skip, limit)
        step = 2000
        total = int(neo_util.graph.run(cypher_query_count).next()['count'])
        idx = 0
        while idx < total:
            r = neo_util.graph.run(cypher_query(skip=idx, limit=step))
            vrel_set.update(r)
            idx += step
        return vrel_set

    def gen_pattern_exist(self, neo_util, gen_pattern):
        """
        关系合成模式在图中是否存在
        :param neo_util:
        :param gen_pattern:
        :return:
        """
        cypher_match_statements = []
        if gen_pattern.startInverse:
            cypher_match_statements.append(
                '(n1:`%s`)-[:%s]->(n0:`%s`)' % (gen_pattern.startRel.startLabel, gen_pattern.startRel.rel, gen_pattern.startRel.endLabel))
        else:
            cypher_match_statements.append(
                '(n0:`%s`)-[:%s]->(n1:`%s`)' % (gen_pattern.startRel.startLabel, gen_pattern.startRel.rel, gen_pattern.startRel.endLabel))
        if gen_pattern.endInverse:
            cypher_match_statements.append(
                '(n2:`%s`)-[:%s]->(n1:`%s`)' % (gen_pattern.endRel.startLabel, gen_pattern.endRel.rel, gen_pattern.endRel.endLabel))
        else:
            cypher_match_statements.append(
                '(n1:`%s`)-[:%s]->(n2:`%s`)' % (gen_pattern.endRel.startLabel, gen_pattern.endRel.rel, gen_pattern.endRel.endLabel))
        cypher_match_statement = ', '.join(cypher_match_statements)
        cypher_query_count = 'match %s return id(n0) limit 1' % cypher_match_statement
        res = list(neo_util.graph.run(cypher_query_count))
        print('gen_pattern_exist: ', len(res) > 0, cypher_query_count)
        return len(res) > 0

    def gen_relationship_frequency(self, neo_util, gen_relationship):
        """
        该关系在对应startLabel 和endLabel 下的频次
        :param neo_util:
        :param gen_relationship:
        :return:
        """
        cypher_query_specific_count = 'match (n0:`%s`)-[:%s]->(n1:`%s`) return count(*) as count' \
                                      % (gen_relationship.startLabel, gen_relationship.rel, gen_relationship.endLabel)
        cypher_query_unspecific_count = 'match (n0:`%s`)-[]->(n1:`%s`) return count(*) as count' \
                                        % (gen_relationship.startLabel, gen_relationship.endLabel)
        specific_total = int(neo_util.graph.run(cypher_query_specific_count).next()['count'])
        unspecific_total = int(neo_util.graph.run(cypher_query_unspecific_count).next()['count'])
        freq = 0
        if unspecific_total is 0:
            freq = 0
        else:
            freq = specific_total / unspecific_total
        print('gen_relationship_frequency: ', freq, cypher_query_specific_count)
        return freq


    def get_relationship_domain_range(self, neo_util :NeoUtil, rel_type: str) -> dir:
        """
        指定关系的domain 和range 标签列表
        :param neo_util:
        :param rel_type:
        :return: {domain: list, range: list}
        """
        cypher_query = 'match (n)-[:%s]->(m) return collect(labels(n)) as domain, collect(labels(m)) as range' \
                        % (rel_type)
        res = neo_util.graph.run(cypher_query).next()
        domain_list = res['domain']
        range_list = res['range']
        domain_set = set()
        range_set = set()
        for domain in domain_list:
            domain_set.update(domain)
        for range in range_list:
            range_set.update(range)
        return {
            'domain': list(domain_set),
            'range': list(range_set)
        }


    def get_relationship_types(self, neo_util: NeoUtil, limit: int, skip: int = 0) -> list:
        """
        获取库中的关系类型
        :param neo_util:
        :param limit:
        :return:
        """
        cypher_query = 'match ()-[r]-() return r.type as type skip %s limit %s' % (skip, limit)
        iter = neo_util.graph.run(cypher_query)
        types = set()
        for rec in iter:
            types.add(rec['type'])
        return list(types)


class ViewGenerator:

    REAL_ID = 'realId'  # 视图中节点保存原节点id的属性key
    START_ID = 'start_id'
    END_ID = 'end_id'

    def __init__(self, view_config, source_neo4j, target_neo4j):
        self.view_config = view_config
        self.source_neo4j = source_neo4j
        self.target_neo4j = target_neo4j
        self.view_util = ViewUtil()
        self.node_map = {}

    def gen(self):
        self.gen_nodes()
        self.gen_rels()
        self.gen_vrels()

    def expand_gen(self):
        self._get_nodes()
        self.gen_vrels()

    def _get_nodes(self):
        node_set = set()
        for label in self.view_config.labels:
            nodes = self.view_util.extract_nodes_by_label(self.target_neo4j, label)
            node_set.update(nodes)
        for node in node_set:
            self.node_map[node[self.REAL_ID]] = node

    def gen_nodes(self):
        """
        load node from source_neo4j to target_neo4j
        :return:
        """
        node_set = set()
        for label in self.view_config.labels:
            nodes = self.view_util.extract_nodes_by_label(self.source_neo4j, label)
            node_set.update(nodes)
        # tx = self.target_neo4j.graph.begin()
        subgraph = None
        step = 3000
        local_idx = 0
        for node in node_set:
            new_node = Node()
            new_node[self.REAL_ID] = node.identity
            for label in node.labels:
                new_node.add_label(label)
            props = dict(node)
            for key in props:
                new_node[key] = props[key]
            if subgraph is None:
                subgraph = new_node | new_node
            else:
                subgraph = subgraph | new_node
            # tx.merge(new_node, primary_label, self.REAL_ID)
            self.node_map[node.identity] = new_node
            local_idx += 1
            if local_idx >= step:
                tx = self.target_neo4j.graph.begin()
                tx.create(subgraph)
                subgraph = None
                local_idx = 0
                tx.commit()
                # tx = self.target_neo4j.graph.begin()
        tx = self.target_neo4j.graph.begin()
        tx.create(subgraph)
        tx.commit()

    def _gen_rel(self, rel_conf):
        """
        根据关系的配置从source_neo4j 中找到对应的关系，添加到target_neo4j 中
        :param rel_conf: {
            "symmetrical": false,
            "type": "HAS_ACTOR",
            "startLabels": [
                "Movie"
            ],
            "transitive": false,
            "endLabels": [
                "Actor"
            ]
        }
        :return: set<Relationship> 新创建的关系（已被添加到target_neo4j 中）
        """
        tx = self.target_neo4j.graph.begin()
        step = 3000
        idx = 0
        rel_type = rel_conf['type']
        start_labels = rel_conf['startLabels']
        end_labels = rel_conf['endLabels']
        rel_records = self.view_util.extract_rels(self.source_neo4j, rel_type, start_labels, end_labels)
        # 记录下新创建的关系，可能用于对称和传递的关系扩展
        new_rel_set = set()
        for record in rel_records:
            new_start_node = self.node_map[record[self.START_ID]]
            new_end_node = self.node_map[record[self.END_ID]]
            new_rel = Relationship(new_start_node, rel_type, new_end_node)
            tx.create(new_rel)
            new_rel_set.add(new_rel)
            idx += 1
            if idx >= step:
                idx = 0
                tx.commit()
                tx = self.target_neo4j.graph.begin()
        tx.commit()
        return new_rel_set

    def gen_rels(self):
        """
        load rels from source_neo4j to target_neo4j
        :return:
        """
        for rel_conf in self.view_config.rels:
            new_rel_set = self._gen_rel(rel_conf)
            self._expand_rels(new_rel_set, rel_conf)

    def _gen_vrel(self, vrel_conf):
        """
        :param vrel_conf:
        :return:
        """
        vrels_id_set = self.view_util.extract_vrels(
            self.target_neo4j, vrel_conf['type'], vrel_conf['rels'][0], vrel_conf['rels'][1])
        vrel_type = vrel_conf['type']

        tx = self.target_neo4j.graph.begin()
        step = 3000
        local_idx = 0
        new_rel_set = set()
        for record in vrels_id_set:
            new_start_node = self.node_map[record[self.START_ID]]
            new_end_node = self.node_map[record[self.END_ID]]
            new_rel = Relationship(new_start_node, vrel_type, new_end_node)
            tx.create(new_rel)
            new_rel_set.add(new_rel)
            local_idx += 1
            if local_idx >= step:
                local_idx = 0
                tx.commit()
                tx = self.target_neo4j.graph.begin()
        tx.commit()
        return new_rel_set

    def gen_vrels(self):
        """
        gen vrel from it's sub_rels
        :return:
        """
        for vrel_conf in self.view_config.vrels:
            new_rel_set = self._gen_vrel(vrel_conf)
            self._expand_rels(new_rel_set, vrel_conf)

    def _expand_rels(self, rel_set, rel_conf):
        if len(rel_set) == 0:
            return
        symmetrical = rel_conf['symmetrical']
        transitive = rel_conf['transitive']
        rel_type = rel_conf['type']
        if (not symmetrical) and (not transitive):
            return
        # 收集所有相关的node
        node_set = set()
        for rel in rel_set:
            node_set.add(rel.start_node)
            node_set.add(rel.end_node)
        # 给node分配id，用于矩阵计算， 0: node0, 1: node1
        node_list = list(node_set)
        node_size = len(node_list)
        # 给node分配id，用于矩阵计算，node0: 0, node1: 1
        node2id_map = {}
        for idx in range(node_size):
            node2id_map[node_list[idx]] = idx
        # 扩展之后的矩阵 - 原邻接矩阵 得到的边就是需要扩展的边
        original_adjacency_marix = np.zeros((node_size, node_size))
        expanded_adjacency_matrix = np.zeros((node_size, node_size))
        for rel in rel_set:
            start_id = node2id_map[rel.start_node]
            end_id = node2id_map[rel.end_node]
            original_adjacency_marix[start_id][end_id] = 1
            expanded_adjacency_matrix[start_id][end_id] = 1
        if symmetrical:
            self._symmetrical_expand(expanded_adjacency_matrix)
        if transitive:
            self._transitive_expand(expanded_adjacency_matrix)
        new_adjacency_matrix = expanded_adjacency_matrix - original_adjacency_marix
        # 需要创建的关系
        new_rel_set = set()
        for i in range(node_size):
            for j in range(node_size):
                if new_adjacency_matrix[i][j] == 1:
                    new_start_node = node_list[i]
                    new_end_node = node_list[j]
                    new_rel = Relationship(new_start_node, rel_type, new_end_node)
                    new_rel_set.add(new_rel)
        # 创建关系，每step 个提交一次
        tx = self.target_neo4j.graph.begin()
        step = 3000
        local_idx = 0
        for new_rel in new_rel_set:
            tx.create(new_rel)
            local_idx += 1
            if local_idx >= step:
                local_idx = 0
                tx.commit()
                tx = self.target_neo4j.graph.begin()
        tx.commit()


    def _symmetrical_expand(self, adjacency_matrix):
        """
        计算对称闭包，修改传入的矩阵
        :param adjacency_matrix:
        :return:
        """
        size = len(adjacency_matrix)
        for x in range(size):
            for y in range(size):
                if adjacency_matrix[x][y] == 1:
                    adjacency_matrix[y][x] = 1

    def _transitive_expand(self, adjacency_matrix):
        """
        计算传递闭包，修改传入的矩阵
        :param adjacency_matrix:
        :return:
        """
        size = len(adjacency_matrix)
        for i in range(size):
            for j in range(size):
                if adjacency_matrix[j][i] == 1:
                    for k in range(size):
                        adjacency_matrix[j][k] = adjacency_matrix[j][k] | adjacency_matrix[i][k]





def gen_graph(curr_conf):
    curr_conf = json.loads(curr_conf)
    # with open('./conf/neo4jConf.json', 'r') as f:
    #     neo4j_conf = json.load(f)
    #     source_neo4j_conf = neo4j_conf['source']
    #     target_neo4j_conf = neo4j_conf['target']
    # source_neo4j = NeoUtil(source_neo4j_conf['url'], source_neo4j_conf['username'], source_neo4j_conf['password'])
    # target_neo4j = NeoUtil(target_neo4j_conf['url'], target_neo4j_conf['username'], target_neo4j_conf['password'])

    view_config = ViewConfig(curr_conf)
    view_generator = ViewGenerator(view_config, source_neo4j, target_neo4j)
    view_generator.gen()


def expand_gen_graph(last_conf, curr_conf):
    last_conf = json.loads(last_conf)
    curr_conf = json.loads(curr_conf)
    # with open('./conf/neo4jConf.json', 'r') as f:
    #     neo4j_conf = json.load(f)
    #     source_neo4j_conf = neo4j_conf['source']
    #     target_neo4j_conf = neo4j_conf['target']
    # source_neo4j = NeoUtil(source_neo4j_conf['url'], source_neo4j_conf['username'], source_neo4j_conf['password'])
    # target_neo4j = NeoUtil(target_neo4j_conf['url'], target_neo4j_conf['username'], target_neo4j_conf['password'])

    view_config0 = ViewConfig(last_conf)
    view_config1 = ViewConfig(curr_conf)

    for vrel0 in view_config0.vrels:
        for vrel1 in view_config1.vrels:
            if vrel0 == vrel1:
                view_config1.vrels.remove(vrel0)
                break

    # view_generator = ViewGenerator(view_config0, source_neo4j, target_neo4j)
    # view_generator.gen()
    view_generator = ViewGenerator(view_config1, source_neo4j, target_neo4j)
    view_generator.expand_gen()


def gen_pattern_exist(gen_pattern):
    # with open('./conf/neo4jConf.json', 'r') as f:
    #     neo4j_conf = json.load(f)
    #     if gen_pattern.isFirstInterval:
    #         neo4j_conf = neo4j_conf['source']
    #     else:
    #         neo4j_conf = neo4j_conf['target']
    # neo_util = NeoUtil(neo4j_conf['url'], neo4j_conf['username'], neo4j_conf['password'])
    if gen_pattern.isFirstInterval:
        neo_util = source_neo4j
    else:
        neo_util = target_neo4j
    view_util = ViewUtil()
    return view_util.gen_pattern_exist(neo_util, gen_pattern)


def gen_relationship_frequency(gen_relationship):
    # with open('./conf/neo4jConf.json', 'r') as f:
    #     neo4j_conf = json.load(f)
    #     target_neo4j_conf = neo4j_conf['target']
    # target_neo4j = NeoUtil(target_neo4j_conf['url'], target_neo4j_conf['username'], target_neo4j_conf['password'])
    view_util = ViewUtil()
    return view_util.gen_relationship_frequency(target_neo4j, gen_relationship)



def test1():
    with open('./conf/viewConf1.json', 'r') as f:
        view_conf = json.load(f)
    with open('./conf/neo4jConf.json', 'r') as f:
        neo4j_conf = json.load(f)
        source_neo4j_conf = neo4j_conf['source']
        target_neo4j_conf = neo4j_conf['target']
    print(view_conf)
    print(neo4j_conf)
    view_config = ViewConfig(view_conf)
    source_neo4j = NeoUtil(source_neo4j_conf['url'], source_neo4j_conf['username'], source_neo4j_conf['password'])
    target_neo4j = NeoUtil(target_neo4j_conf['url'], target_neo4j_conf['username'], target_neo4j_conf['password'])

    view_generator = ViewGenerator(view_config, source_neo4j, target_neo4j)
    view_generator.gen()


def test2():
    with open('./conf/viewConf0.json', 'r') as f:
        view_conf0 = json.load(f)
    with open('./conf/viewConf1.json', 'r') as f:
        view_conf1 = json.load(f)
    with open('./conf/neo4jConf.json', 'r') as f:
        neo4j_conf = json.load(f)
        source_neo4j_conf = neo4j_conf['source']
        target_neo4j_conf = neo4j_conf['target']
    source_neo4j = NeoUtil(source_neo4j_conf['url'], source_neo4j_conf['username'], source_neo4j_conf['password'])
    target_neo4j = NeoUtil(target_neo4j_conf['url'], target_neo4j_conf['username'], target_neo4j_conf['password'])

    view_config0 = ViewConfig(view_conf0)
    view_config1 = ViewConfig(view_conf1)

    for vrel0 in view_config0.vrels:
        for vrel1 in view_config1.vrels:
            if vrel0 == vrel1:
                view_config1.vrels.remove(vrel0)
                break

    view_generator = ViewGenerator(view_config0, source_neo4j, target_neo4j)
    view_generator.gen()

    view_generator = ViewGenerator(view_config1, source_neo4j, target_neo4j)
    view_generator.expand_gen()

    print(view_config0, view_config1)


def test3():
    import multiview.multiview_pb2 as mv

    start_rel = mv.GenRelationship()
    start_rel.startLabel = 'Company'
    start_rel.endLabel = 'Person'
    start_rel.rel = 'founder'
    gen_pattern = mv.GenPattern(
        startRel=start_rel,
        endRel=start_rel,
        startInverse=False,
        endInverse=True
    )
    print(gen_pattern_exist(gen_pattern))
    print(gen_relationship_frequency(start_rel))
    

def test4():
    with open('./conf/viewConf3.json', 'r', encoding='utf-8') as f:
        view_conf = json.load(f)
    with open('./conf/neo4jConf1.json', 'r') as f:
        neo4j_conf = json.load(f)
        source_neo4j_conf = neo4j_conf['source']
        target_neo4j_conf = neo4j_conf['target']
    print(view_conf)
    print(neo4j_conf)
    view_config = ViewConfig(view_conf)
    source_neo4j = NeoUtil(source_neo4j_conf['url'], source_neo4j_conf['username'], source_neo4j_conf['password'])
    target_neo4j = NeoUtil(target_neo4j_conf['url'], target_neo4j_conf['username'], target_neo4j_conf['password'])

    view_generator = ViewGenerator(view_config, source_neo4j, target_neo4j)
    view_generator.gen()

def test5():
    neo_util = NeoUtil("bolt://localhost:11001", "neo4j", "123123")
    view_util = ViewUtil()
    # res = view_util.get_relationship_domain_range(neo_util, "中文名")
    # print(res)
    types = view_util.get_relationship_types(neo_util, 10000 * 3)
    entities = set()
    config = {
        "vrels": [],
        "labels": [],
        "rels": []
    }
    for type in types:
        res = view_util.get_relationship_domain_range(neo_util, type)
        domain = res['domain']
        range = res['range']
        entities.update(domain)
        entities.update(range)
        config['rels'].append({
            "symmetrical": False,
            "type": type,
            "startLabels": list(domain),
            "transitive": False,
            "endLabels": list(range)
        })
    config['labels'].extend(entities)
    with open('./conf/viewConf_kgAttr1k.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def test6():
    NODE_LABEL = "Resource"
    REL_TYPES = ["isLeaderOf", "isLocatedIn", "isAffiliatedTo", "owns",
                 "hasGender", "wasBornIn", "isCitizenOf", "created", "diedIn",
                 "happenedIn", "hasCapital", "graduatedFrom", "isPoliticianOf",
                 "worksAt", "participatedIn", "hasOfficialLanguage", "imports",
                 "hasNeighbor", "hasCurrency", "exports", "influences", "playsFor",
                 "hasChild", "isMarriedTo", "actedIn", "directed", "hasWonPrize",
                 "isConnectedTo", "isInterestedIn", "hasMusicalRole", "dealsWith",
                 "wroteMusicFor", "edited"]
    neo_util = NeoUtil("bolt://localhost:7687", "neo4j", "123123")
    view_util = ViewUtil()
    cypher_query = 'match (n0)-[r1]-(n1)-[r2]-(n2) return type(r1), type(r2) limit 10'

if __name__ == '__main__':
    test6()