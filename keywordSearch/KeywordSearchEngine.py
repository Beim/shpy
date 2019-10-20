from keywordSearch.KsDbService import KsDbService
from neo4jdb.ExSubgraph import ExSubgraph
import jieba
from functools import cmp_to_key
import re

class KeywordSearchEngine:
    """
    参考论文: Elbassuoni S, Blanco R. Keyword search over RDF graphs[C]//Proceedings of the 20th ACM international conference on Information and knowledge management. ACM, 2011: 237-242.

    Attributes:
        ks_db_service: KsDbService
        ks_service_cache: KsServiceCache
        alpha:
        beta:
        gama:
    """

    def __init__(self, alpha=0.4, beta=0.5, gama=0.4):
        self.ks_db_service = KsDbService()
        self.ks_service_cache = KsServiceCache(self.ks_db_service)
        self.alpha = alpha
        self.beta = beta
        self.gama = gama

    def keyword_search(self, query):
        """
        关键字搜索的主要方法
        :param query: 查询字符串
        :return: 子图
        """
        keywords = query.split(' ')
        print(keywords)
        match_lists = []
        query_graph = ExSubgraph()
        query_graph.add_init_node()
        for k in keywords:
            rels = self.ks_db_service.get_related_relationships(k)
            match_lists.append(rels)
            for rel in rels:
                query_graph.add_relationship(rel)
        invert_match = self._invert_match_lists(match_lists)
        subgraphs = self._retrieve_subgraphs(invert_match, query_graph)
        subgraphs, subgraph_prob = self._rank_subgraphs(subgraphs, keywords)
        return subgraphs, subgraph_prob

    def _retrieve_subgraphs(self, invert_match, query_graph):
        """
        Algorithm1
        取出subgraphs
        :param invert_match:  rel 到keywords 的idx 的集合的映射 {rel1: {1, 2, 3, ...}, rel2: {...}, ...}
        :param query_graph: ExSubgraph
        :return: subgraphs [subgraph1, subgraph2, ...]
        """
        subgraphs = []
        for rel in query_graph.subgraph.relationships:
            G = ExSubgraph()
            G.add_relationship(rel)
            X = self._adjacent_relationship_subgraph(rel, invert_match, query_graph)
            self._extend_subgraph(G, X, subgraphs, invert_match, query_graph)
        return subgraphs

    @staticmethod
    def _adjacent_relationship_subgraph(rel, invert_match, query_graph):
        """
        取出与rel 邻接的rels，并构成子图
        X ← {t ∈ A(t)}
        :param rel: 对应公式中的t
        :param invert_match: query word idx 与rel 的匹配, {rel1: {1, 2, 3, ...}, rel2: {...}, ...}
        :param query_graph: ExSubgraph
        :return: 对应公式中的X
        """
        neighbour_rels = query_graph.neighbour_rels(rel)
        X = ExSubgraph()
        X.add_init_node()
        for neighbour_rel in neighbour_rels:
            """In order to retrieve only unique subgraphs, 
            we associate with each edge t_i an id 
            and we only add a neighbor to the adjacency list of t_i 
            if its id is greater than that of t_i.
            """
            if neighbour_rel.identity <= rel.identity:
                continue
            """to ensure that we do not consider joining triples that match the same set of keywords
            """
            if invert_match[neighbour_rel] <= invert_match[rel] or invert_match[neighbour_rel] > invert_match[rel]:
                continue
            X.add_relationship(neighbour_rel)
        return X

    def _invert_match_lists(self, match_lists):
        """
        将keywords 的idx 到相关的relationship集合 L{rel1, rel2, ...} 的映射
        invert 成rel 到idx 的集合的映射
        :param match_lists: [[rel1, rel2, ...], [...], ...]
        :return: invert_match {rel1: {1, 2, 3, ...}, rel2: {...}, ...}
        """
        invert_match = {}
        for idx in range(len(match_lists)):
            match = match_lists[idx]
            for rel in match:
                if rel not in invert_match:
                    invert_match[rel] = {idx}
                else:
                    invert_match[rel].add(idx)
        return invert_match

    def _extend_subgraph(self, G, X, subgraphs, invert_match, query_graph):
        """
        Algorithm2
        扩展G，扩展完成后添加到subgraphs 中
        :param G: ExSubgraph 需要扩展的子图
        :param X: ExSubgraph 用于扩展的relationships 的子图
        :param subgraphs: 所有retrieve 的子图list
        :param invert_match: rel 到keywords 的idx 的集合的映射 {rel1: {1, 2, 3, ...}, rel2: {...}, ...}
        :return: None
        """
        def related_lists(graph, invert_match):
            """
            The function L(G) returns the set of lists the edges of a subgraph G belong to.
            :param graph: ExSubgraph
            :param invert_match:
            :return: set
            """
            lists = set()
            for rel in graph.subgraph.relationships:
                lists.update(invert_match[rel])
            return lists

        def neighbours(rel_t, G, query_graph):
            """
            The function NEIGHBORS(t,G) retrieves all neighbors of an edge t that are not neighbors to edges in G.
            :param rel_t:
            :param G:
            :param query_graph:
            :return:
            """
            neighbours_of_t = query_graph.neighbour_rels(rel_t)
            neighbours_of_G = set()
            for rel in G.subgraph.relationships:
                neighbours_of_G.update(query_graph.neighbour_rels(rel))
            return neighbours_of_t - neighbours_of_G

        def append_subgraph(G, subgraphs):
            """
            将G 添加到子图集合中，保证G的Maximal
            the function MAXIMAL(G) ensures that the retrieved subgraph is unique and maximal.
            :param G:
            :param subgraphs:
            :return: None
            """
            for idx in range(len(subgraphs)):
                graph = subgraphs[idx]
                if G.subgraph.relationships <= graph.subgraph.relationships:
                    return
                if G.subgraph.relationships > graph.subgraph.relationships:
                    subgraphs[idx] = G
                    return
            subgraphs.append(G)

        while len(X.subgraph.relationships) > 0:
            rel_t = X.pop_relationship()
            L_t = invert_match[rel_t]
            L_G = related_lists(G, invert_match)
            """ensures that only edges that belong to at least one different list 
            other than the lists the edges of the current subgraph G belong to are considered.
            This ensures that we construct only subgraphs whose edges match different sets of keywords.
            """
            if L_t <= L_G or L_G <= L_t:
                continue
            neighbours_temp = neighbours(rel_t, G, query_graph)
            X1 = X.copy()
            ngraph = ExSubgraph()
            for rel in neighbours_temp:
                ngraph.add_relationship(rel)
            X1.combine(ngraph)
            G1 = G.copy()
            G1.add_relationship(rel_t)
            self._extend_subgraph(G1, X1, subgraphs, invert_match, query_graph)
        append_subgraph(G, subgraphs)

    def _rank_subgraphs(self, subgraphs, query_words):
        """
        利用概率模型排序子图
        :param subgraphs: list [subgraph1, subgraph2, ...]
        :param query_words: list 查询关键字
        :return: subgraphs
        :return: subgraph_prob
        """
        alpha = self.alpha
        beta = self.beta
        gama = self.gama
        subgraph_prob = dict()
        for subgraph in subgraphs:
            subgraph_prob[subgraph] = self._prob_Q_G(query_words, subgraph, alpha, beta, gama)
        subgraphs.sort(key=cmp_to_key(lambda x,y: subgraph_prob[y] - subgraph_prob[x]))
        return subgraphs, subgraph_prob

    def _prob_Q_G(self, Q, G, alpha, beta, gama):
        """
        P(Q|G) = \sum_{i=1}^m P(q_i|G)
        :param Q: list query_words
        :param G: ExSubgraph subgraph
        :param alpha:
        :param beta:
        :param gama:
        :return: number P(Q|G)
        """
        D = G.get_relationships()

        def prob_q_G(i):
            """
            P(q_i|G) = \sum_{j=1}^n (1/n) P(q_i|D_j, r_j)
            :param i:
            :return:
            """
            n = len(D)
            sumj = 0
            for j in range(n):
                sumj += prob_q_D_r(i, j)
            return sumj / n

        def prob_q_D_r(i, j):
            """
            P(q_i|D_j, r_j) = \beta P(q_i|D_j) P(R_j|q_i) + (1-\beta) P(q_i|D_j)
            :param i:
            :param j:
            :return:
            """
            D_j = D[j]
            type_name = re.search(".*py2neo\.data\.(.*)'>.*", str(type(D_j))).group(1)
            k = self.ks_service_cache.get_relationship_types().index(type_name)
            return beta * prob_q_D(i, j) * prob_R_q(i, k) + (1 - beta) * prob_q_D(i, j)

        def prob_q_D(i, j):
            """
            P(q_i|D_j) = \alpha \cfrac{c(q_i, D_j)}{|D_j|} + (1 - \alpha) \cfrac{c(q_i, Col)}{|Col|}
            这里假设D_j 中元素不重复，c(q_i, D_j) = 1/0
            :param i:
            :param j:
            :return:
            """
            return alpha * frac_q_D_D(i, j) + (1 - alpha) * frac_q_Col_Col(i)

        def frac_q_D_D(i, j):
            """
            c(q_i, D_j) / |D_j|
            :param i:
            :param j:
            :return:
            """
            q_i = Q[i]
            D_j = D[j]
            document = D_j['document']
            return document.count(q_i) / len(document)

        def frac_q_Col_Col(i):
            """
            c(q_i, Col) / |Col|
            :param i:
            :return:
            """
            q_i = Q[i]
            return self.ks_service_cache.get_related_relationships_count(keyword=q_i, r_type=None) \
                / self.ks_service_cache.get_related_relationships_count(keyword=None, r_type=None)

        def prob_R_q(i, j):
            """
            P(R_j|q_i) = (P(q_i|R_j) P(R_j)) / (\sum_k P(q_i|R_k) P(R_k))
            :return:
            """
            sumk = 1
            n = self.ks_service_cache.get_relationship_type_count()
            for k in range(n):
                sumk += prob_q_R(i, k) * prob_R(k)
            return (prob_q_R(i, j) * prob_R(j)) / sumk

        def prob_R(k):
            """
            P(R_k) is the prior probability of the document Rj being relevant to any term,
            which we set uniformly
            :param k:
            :return:
            """
            # R_k = type(D[k])
            R_k = self.ks_service_cache.get_relationship_types()[k]
            return self.ks_service_cache.get_related_relationships_count(keyword=None, r_type=R_k) \
                / self.ks_service_cache.get_related_relationships_count(keyword=None, r_type=None)

        def prob_q_R(i, j):
            """
            P(q_i|R_j) = \gama (c(q_i|R_j) / |R_j|) + (1-\gama) (c(q_i, ColR) / |ColR|)
            :param i:
            :param j:
            :return:
            """
            return gama * frac_q_R_R(i, j) * (1 - gama) * frac_q_ColR_ColR(i)

        def frac_q_R_R(i, j):
            """
            c(q_i|R_j) / |R_j|
            :param i:
            :param j:
            :return:
            """
            q_i = Q[i]
            # D_j = D[j]
            # R_j = type(D_j)
            R_j = self.ks_service_cache.get_relationship_types()[j]
            return self.ks_service_cache.get_related_relationships_count(keyword=q_i, r_type=R_j) \
                   / self.ks_service_cache.get_related_relationships_count(keyword=None, r_type=R_j)

        def frac_q_ColR_ColR(i):
            """
            c(q_i, ColR) / |ColR)
            :param i:
            :return:
            """
            q_i = Q[i]
            return self.ks_service_cache.get_related_relationships_count(keyword=q_i, r_type=None) \
                   / self.ks_service_cache.get_related_relationships_count(keyword=None, r_type=None)

        p = 1
        for i in range(len(Q)):
            p *= prob_q_G(i)
        return p

    @staticmethod
    def _cut_words(s):
        seg_list = jieba.cut(s)
        words = []
        for seg in seg_list:
            if seg.strip() != '':
                words.append(seg.strip())
        return words


class KsServiceCache:
    """
    缓存

    Attributes:
        ks_db_service: KsDbService
        rlc_keyword: dict 缓存ks_db_service.get_related_relationships_count(keyword=q_i, r_type=None)
        rlc_rtype: dict 缓存ks_db_service.get_related_relationships_count(keyword=None, r_type=R_j)
        rlc_keyword_rtype: dict(dict) 缓存ks_db_service.get_related_relationships_count(keyword=q_i, r_type=R_j)
        rlc_none_none: number 缓存self.ks_db_service.get_related_relationships_count(keyword=None, r_type=None)
        R_type_count: number 缓存self.ks_db_service.get_relationship_type_count()
    """

    def __init__(self, ks_db_service):
        self.ks_db_service = ks_db_service
        self.rlc_keyword = dict()
        self.rlc_rtype = dict()
        self.rlc_keyword_rtype = dict()
        self.rlc_none_none = None
        self.R_type_count = None
        self.relationship_types = None

    def get_relationship_types(self):
        """
        调用ks_db_service.get_relationship_types
        缓存结果
        :return:
        """
        if self.relationship_types is None:
            self.relationship_types = self.ks_db_service.get_relationship_types()
            self.R_type_count = len(self.relationship_types)
        return self.relationship_types

    def get_relationship_type_count(self):
        """
        调用ks_db_service.get_relationship_types
        缓存结果
        :return: number
        """
        if self.R_type_count is None:
            self.get_relationship_types()
            # self.R_type_count = self.ks_db_service.get_relationship_type_count()
        return self.R_type_count

    def get_related_relationships_count(self, keyword=None, r_type=None):
        """
        调用ks_db_service.get_related_relationships_count
        缓存结果
        :param keyword:
        :param r_type:
        :return: number
        """
        if keyword is None and r_type is None:
            if self.rlc_none_none is None:
                self.rlc_none_none = self.ks_db_service.get_related_relationships_count(keyword=keyword, r_type=r_type)
            return self.rlc_none_none
        elif keyword is None and r_type is not None:
            if r_type not in self.rlc_rtype:
                self.rlc_rtype[r_type] = self.ks_db_service.get_related_relationships_count(keyword=keyword, r_type=r_type)
            return self.rlc_rtype[r_type]
        elif keyword is not None and r_type is None:
            if keyword not in self.rlc_keyword:
                self.rlc_keyword[keyword] = self.ks_db_service.get_related_relationships_count(keyword=keyword, r_type=r_type)
            return self.rlc_keyword[keyword]
        else:
            if keyword not in self.rlc_keyword_rtype:
                self.rlc_keyword_rtype[keyword] = dict()
                self.rlc_keyword_rtype[keyword][r_type] = self.ks_db_service.get_related_relationships_count(keyword=keyword, r_type=r_type)
            else:
                if r_type not in self.rlc_keyword_rtype[keyword]:
                    self.rlc_keyword_rtype[keyword][r_type] = self.ks_db_service.get_related_relationships_count(keyword=keyword, r_type=r_type)
            return self.rlc_keyword_rtype[keyword][r_type]


if __name__ == '__main__':
    e = KeywordSearchEngine()
    # subgraphs, prob = e.keyword_search('comedy academy award')
    # subgraphs, prob = e.keyword_search('手游 腾讯云')
    # subgraphs, prob = e.keyword_search('手游 古风 腾讯云')
    subgraphs, prob = e.keyword_search('云服务器 标准型')
    # subgraphs, prob = e.keyword_search('证券 云服务')
    # subgraphs, prob = e.keyword_search('comedy innerspace')
    idx = 0
    for g in subgraphs:
        idx += 1
        rels = list(g.subgraph.relationships)
        for rel in rels:
            print(rel.start_node['名'], type(rel), rel.end_node['名'])
        print(prob[g])
        print()
        if idx > 5:
            break