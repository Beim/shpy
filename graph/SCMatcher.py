from neo4jdb.ExSubgraph import ExSubgraph
from py2neo.data import Node, Relationship
import numpy as np
import Levenshtein
from functools import cmp_to_key
import copy
from neo4jdb.Neo4jUtil import NeoUtil


class SCMatcher:
    """
    匹配查询子图

    Attributes:
        gq: ExSubgraph query graph
        gd_vnum: number
        gq_vnum: number
        gd_nodes: list
        gq_nodes: list
        rs: float range [0,1] # structur sim rate
        ra: float ra=1-rs # attr sim rate
        K: int # top k sim matrix
        st: float # similarity threshold
        schema_local: dict # local cache
    """

    def __init__(self, gd_nodes, gq, rs, K=1, st=0.9, neo_util: NeoUtil = None):
        self.neo_util = neo_util
        self.gq = gq
        self.gd_vnum = len(gd_nodes)
        self.gd_nodes = list(gd_nodes)
        if self.gq.subgraph is not None:
            self.gq_vnum = len(self.gq.subgraph.nodes)
            self.gq_nodes = list(self.gq.subgraph.nodes)
        self.rs = rs
        if rs > 1 or rs < 0:
            raise Exception('rs %s out of bound [0, 1]' % rs)
        self.ra = 1 - rs
        self.K = min(K, self.gd_vnum)
        self.st = st
        self.schema_local = dict()

    def run(self):
        matcher = {}
        if self.gd_vnum == 0:
            for node in self.gq_nodes:
                matcher[node] = None
        else:
            sim_matrix = self._similarity_matrix()
            topk_matrix = self._topk_matrix(sim_matrix, self.K, self.st)
            all_match_graph = self._all_match_graph(topk_matrix)
            matcher = self._best_match_graph(all_match_graph, sim_matrix)
        return matcher

    # 从topk 相似矩阵生成所有的匹配方案
    def _all_match_graph(self, topk_matrix):
        match = {}
        for nq in range(self.gq_vnum):
            match[nq] = None
        all_match = [match]
        for nq in range(self.gq_vnum):
            if len(topk_matrix[nq]) > 0:
                copy_match = [all_match]
                for i in range(len(topk_matrix[nq]) - 1):
                    copy_match.append(copy.deepcopy(all_match))
                for i in range(len(topk_matrix[nq])):
                    for match in copy_match[i]:
                        match[nq] = topk_matrix[nq][i]
                all_match = []
                for matches in copy_match:
                    all_match.extend(matches)
        return all_match

    # 所有匹配中计算属性和边的相似度最高的匹配
    def _best_match_graph(self, all_match_graph, sim_matrix):
        best_match = all_match_graph[0]
        biggest_sim = 0
        for match in all_match_graph:
            sim = self._match_attr_sim(match, sim_matrix)
            sim += self._match_struc_sim(match)
            if sim > biggest_sim:
                best_match = match
                biggest_sim = sim
        # 将dict 中的index number 改为node
        nmatch = {}
        for k in best_match:
            if best_match[k] is None:
                nmatch[self.gq_nodes[k]] = None
            else:
                nmatch[self.gq_nodes[k]] = self.gd_nodes[best_match[k]]
        return nmatch

    # 计算一种Gq与Gd 的匹配的属性相似度之和
    def _match_attr_sim(self, match, sim_matrix):
        sim = 0
        for nq in match:
            nd = match[nq]
            if nd != None:
                sim += sim_matrix[nq][nd]
        return sim

    def _match_struc_sim(self, match):
        neo4j = self.neo_util
        nmatch = {}
        for k in match:
            if match[k] is None:
                nmatch[self.gq_nodes[k]] = None
            else:
                nmatch[self.gq_nodes[k]] = self.gd_nodes[match[k]]
        sim = 0
        for r in self.gq.subgraph.relationships:
            start_nd = nmatch[r.start_node]
            end_nd = nmatch[r.end_node]
            if start_nd is not None and end_nd is not None:
                m = neo4j.graph.match([start_nd, end_nd], type(r))
                if m.first() is not None:
                    sim += 1
        return sim

    def _topk_matrix(self, sim_matrix, K, threshold):
        topk = []
        for i in range(self.gq_vnum):
            topk.append([])
            temp_list = [i for i in range(self.gd_vnum)]
            temp_list.sort(key=cmp_to_key(lambda x, y: sim_matrix[i][y] - sim_matrix[i][x]))
            for j in range(K):
                if sim_matrix[i][temp_list[j]] < threshold:
                    break
                topk[i].append(temp_list[j])
        return topk

    def _similarity_matrix(self):
        sim_matrix = np.zeros([self.gq_vnum, self.gd_vnum])
        for gq_idx in range(self.gq_vnum):
            for gd_idx in range(self.gd_vnum):
                node_gq = self.gq_nodes[gq_idx]
                node_gd = self.gd_nodes[gd_idx]
                common_labels = node_gd.labels & node_gq.labels
                if len(common_labels) > 0:
                    sim_matrix[gq_idx][gd_idx] = self._node_similarity(node_gd, node_gq)
        return sim_matrix

    def _node_similarity(self, node_gd, node_gq):
        return 1 - ( self.rs * self._struc_dist(node_gd, node_gq) + self.ra * self._attr_dist(node_gd, node_gq) )

    def _struc_dist(self, node_gd, node_gq):
        return 1

    def _attr_dist(self, node_gd, node_gq):
        """
        计算属性相似度
        :param node_gd: node from database
        :param node_gq: node from query
        :return:
        """
        numeric_type = [int, float]
        str_type = [str]
        list_type = [list]
        categorical_type = [bool]

        props_gd = dict(node_gd)
        props_gq = dict(node_gq)
        common_props = props_gd.keys() & props_gq.keys()

        dist = 0

        if len(common_props) == 0:
            return 1

        def categorical_dist(pgd, pgq):
            if pgd == pgq:
                return 0
            else:
                return 1

        def numeric_dist(pgd, pgq):
            return 1 - 1 / (1 + np.abs(pgd - pgq))


        def text_dist(pgd, pgq):
            if pgd == '' and pgq == '':
                return 0
            if len(pgd) > 20:
                pgd = pgd[:20]
            if len(pgq) > 20:
                pgq = pgq[:20]
            return Levenshtein.distance(pgd, pgq) / max(len(pgd), len(pgq))

        def text_alias_dist(pgd, pgq):
            min_dist = 1
            for p1 in pgd:
                for p2 in pgq:
                    min_dist = min(min_dist, Levenshtein.distance(p1, p2) / max(len(p1), len(p2)))
            return min_dist

        def list_dist(pgd_list, pgq_list):
            """
            计算两个list 属性的距离，取每一对的最小值
            假设前提：两个list 中的项都是单个值且类型相同
            :param pgd_list:
            :param pgq_list:
            :return:
            """
            min_dist = 1
            if len(pgd_list) is 0 or len(pgq_list) is 0:
                return 1
            if type(pgd_list[0]) in numeric_type:
                dist_func = numeric_dist
            elif type(pgd_list[0]) in str_type:
                dist_func = text_dist
            elif type(pgd_list[0]) in categorical_type:
                dist_func = categorical_dist

            for p1 in pgd_list:
                for p2 in pgd_list:
                    min_dist = min(min_dist, dist_func(p1, p2))
            return min_dist

        # 以_ 开头的属性不参与相似度计算
        skip_prop_num = 0
        for prop in common_props:
            if prop[0] == '_':
                skip_prop_num += 1
                continue
            pgd = props_gd[prop]
            pgq = props_gq[prop]
            # if type(pgd) != type(pgq):
            #     dist += 1
            #     continue
            pgd_type = type(pgd)
            pgq_type = type(pgq)
            # pgd 与pgq 都是数字
            if (pgd_type in numeric_type) and (pgq_type in numeric_type):
                dist += numeric_dist(pgd, pgq)
            # pgd 与pgq 都是字符
            elif (pgd_type in str_type) and (pgq_type in str_type):
                dist += text_dist(pgd, pgq)
            elif (pgd_type in categorical_type) and (pgq_type in categorical_type):
                dist += categorical_dist(pgd, pgq)
            # pgd 是数组，pgq 是数组
            elif (pgd_type in list_type) and (pgq_type in list_type):
                dist += list_dist(pgd, pgq)
            # pgd 是数组 ，pgq 不是数组（假设是单个值）
            elif pgd_type in list_type:
                dist += list_dist(pgd, [pgq])
            # pgd 不是数组（假设是单个值），pgq 是数组
            elif pgq_type in list_type:
                dist += list_dist([pgd], pgq)
            else:
                raise Exception('attr_type[%s][%s] not defined' % (pgd_type, pgq_type))
        return dist / (len(common_props) - skip_prop_num)


# def test1():
#     a = Node('Person', name='p1')
#     b = Node('Person', name='p2')
#     c = Node('Person', name='p3')
#     ab = Relationship(a, 'knows', b)
#     ac = Relationship(a, 'knows', c)
#
#     d = Node('Person', name='p4')
#     e = Node('Person', name='xy')
#     de = Relationship(d, 'knows', e)
#
#     gd = ExSubgraph()
#     gq = ExSubgraph()
#
#     gd.add_relationship(ab)
#     gd.add_relationship(ac)
#
#     gq.add_relationship(de)
#
#     sc_matcher = SCMatcher(gd.get_nodes(), gq, 0, 3, 0.9)
#     # match = sc_matcher.run()
#
#     dist = sc_matcher._attr_dist(a, b)
#     print('dist = %s' % dist)


if __name__ == '__main__':
    pass
    # test1()

    
