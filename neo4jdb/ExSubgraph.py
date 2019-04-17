from py2neo.data import Node, Relationship


class ExSubgraph:
    """
    Attributes:
        subgraph: ExSubgraph
        start_rels: dict {a: {rel1, rel2, ...}}，以该点(key)为起点的边的集合(val)
        end_rels: dict
        unimportant_nodes: set
        # max_degree:
        # max_degree_in:
        # max_degree_out:
    """

    def __init__(self):
        self.subgraph = None
        self.start_rels = dict()
        self.end_rels = dict()
        self.unimportant_nodes = set()

    def pop_relationship(self):
        """
        任意删除一条边并返回
        :return:
        """
        the_rel = None
        for rel in self.subgraph.relationships:
            the_rel = rel
            break
        self.remove_relationship(the_rel)
        return the_rel

    def add_init_node(self):
        """
        添加一个初始化节点到subgraph 中，以免删除所有关系的同时删除了所有点时报错
        :return: None
        """
        init_node = Node('Init')
        self.add_node(init_node)

    def remove_relationship(self, rel):
        """
        从子图中删除一个relationship
        若该rel 两端的节点还有其他的边则保留，否则删除
        若该rel 在子图中不存在，则直接返回
        :param rel: Relationship
        :return: None
        """
        if rel not in self.subgraph.relationships:
            return
        self.subgraph = self.subgraph - rel
        start_node = rel.start_node
        end_node = rel.end_node
        if self.node_degree(start_node, 'bi') == 1:
            self.start_rels.pop(start_node)
        else:
            if len(self.start_rels[start_node]) == 1:
                self.start_rels.pop(start_node)
            else:
                self.start_rels[start_node].remove(rel)
        if self.node_degree(end_node, 'bi') == 1:
            self.end_rels.pop(end_node)
        else:
            if len(self.end_rels[end_node]) == 1:
                self.end_rels.pop(end_node)
            else:
                self.end_rels[end_node].remove(rel)
        self.unimportant_nodes.discard(start_node)
        self.unimportant_nodes.discard(end_node)

    def copy(self):
        """
        复制
        :return: ExSubgraph 返回一个新的ExSubgraph 对象
        """
        new_subgraph = ExSubgraph()
        if self.subgraph is None:
            return new_subgraph
        for node in self.subgraph.nodes:
            new_subgraph.add_node(node)
        for rel in self.subgraph.relationships:
            new_subgraph.add_relationship(rel)
        return new_subgraph

    def combine(self, ex_subgraph):
        """
        self 与ex_subgraph 合并
        :param ex_subgraph: ExSubgraph
        :return: None
        """
        if ex_subgraph.subgraph is None:
            return
        for node in ex_subgraph.subgraph.nodes:
            self.add_node(node)
        for rel in ex_subgraph.subgraph.relationships:
            self.add_relationship(rel)

    def neighbour_rels(self, rel):
        """
        给定relationship 的邻居relationship，不考虑方向
        若a1-[r1]->b1 是a2-[r2]->b2 的邻居，则a1 == a2 或 a1 == b2 或 b1 == a2 或 b1 == b2
        :param rel:
        :return: 邻居relationships set
        """
        snode = rel.start_node
        enode = rel.end_node
        neighbours = set()
        if snode in self.start_rels:
            neighbours.update(self.start_rels[snode])
        if snode in self.end_rels:
            neighbours.update(self.end_rels[snode])
        if enode in self.start_rels:
            neighbours.update(self.start_rels[enode])
        if enode in self.end_rels:
            neighbours.update(self.end_rels[enode])
        neighbours.discard(rel)
        return neighbours

    def can_reach_in_k(self, n1, n2, k, has_direction=False):
        """
        节点n1 和节点n2 能否在k 步内到达
        :param n1: 节点1
        :param n2: 节点2
        :param k: 步数
        :param has_direction: 是否将subgraph 视为有向图
        :return: boolean
        """
        if n1 == n2:
            return True
        if n1 is None or n2 is None:
            return False
        if has_direction:
            raise NotImplementedError()
        else:
            if (n1 not in self.start_rels and n1 not in self.end_rels) or (n2 not in self.start_rels and n2 not in self.end_rels):
                return False
            visited = set()
            to_visit = [[n1, 0]]
            while len(to_visit) > 0:
                visit_node, deep = to_visit.pop(0)
                visited.add(visit_node)
                deep += 1
                candidate_nodes = set()
                if visit_node in self.start_rels:
                    temp_nodes = list(map(lambda x: x.end_node, self.start_rels[visit_node]))
                    candidate_nodes.update(temp_nodes)
                if visit_node in self.end_rels:
                    temp_nodes = list(map(lambda x: x.start_node, self.end_rels[visit_node]))
                    candidate_nodes.update(temp_nodes)
                candidate_nodes = candidate_nodes - visited
                if n2 in candidate_nodes:
                    return True
                if deep == k:
                    continue
                for n in candidate_nodes:
                    to_visit.append([n, deep])
            return False

    def node_degree(self, node, direction='bi'):
        """
        节点的度（入度/出度/入+出度）
        :param node: 节点
        :param direction: 方向{in: 入，out: 出, bi: 双向,入+出}
        :return: number
        """
        # in and out
        degree = 0
        if direction == 'bi':
            if node in self.start_rels:
                degree = degree + len(self.start_rels[node])
            if node in self.end_rels:
                degree = degree + len(self.end_rels[node])
        # in
        elif direction == 'in':
            if node in self.end_rels:
                degree = len(self.end_rels[node])
        # out
        elif direction == 'out':
            if node in self.start_rels:
                degree = len(self.start_rels[node])
            pass
        else:
            raise Exception('direction not defined')
        return degree

    def add_node(self, node):
        if self.subgraph == None:
            self.subgraph = node | node
        else:
            self.subgraph = self.subgraph | node

    def add_relationship(self, relationship):
        """
        将relationship 合并到subgraph 中
        若relationship 已存在则直接返回
        :param relationship:
        :return: None
        """
        if (self.subgraph is not None) and (relationship in self.subgraph.relationships):
            return
        start_node = relationship.start_node
        end_node = relationship.end_node
        # 添加start relation
        if start_node in self.start_rels:
            self.start_rels[start_node].add(relationship)
        else:
            self.start_rels[start_node] = {relationship}
        # 计算最大出度
        # self.max_degree_out = max(self.max_degree_out, len(self.start_rels[start_node]))

        # 添加end relation
        if end_node in self.end_rels:
            self.end_rels[end_node].add(relationship)
        else:
            self.end_rels[end_node] = {relationship}
        # 计算最大入度
        # self.max_degree_in = max(self.max_degree_in, len(self.end_rels[end_node]))

        # 计算最大度
        # if start_node in self.end_rels:
        #     start_node_degree = len(self.start_rels[start_node]) + len(self.end_rels[start_node])
        # else:
        #     start_node_degree = len(self.start_rels[start_node])
        # self.max_degree = max(self.max_degree, start_node_degree)
        #
        # if end_node in self.start_rels:
        #     end_node_degree = len(self.start_rels[end_node]) + len(self.end_rels[end_node])
        # else:
        #     end_node_degree = len(self.end_rels[end_node])
        # self.max_degree = max(self.max_degree, end_node_degree)

        # 将relation 合并到子图
        if self.subgraph == None:
            self.subgraph = relationship | relationship  # py2neo.data.Subgraph
        else:
            self.subgraph = self.subgraph | relationship

    def get_nodes(self):
        if self.subgraph == None:
            return []
        else:
            return list(self.subgraph.nodes)

    def get_relationships(self):
        if self.subgraph == None:
            return []
        else:
            return list(self.subgraph.relationships)

    def get_node_if_exist(self, new_node, check_label = True, return_new_node = True):
        """
        若当前子图中存在与new_node 相同label 与属性的node，则返回old_node
        否则直接返回new_node
        :param new_node:
        :param check_label:
        :param return_new_node:
        :return: Node
        """
        if self.subgraph is not None:
            for old_node in self.subgraph.nodes:
                has_old_node = dict(old_node) == dict(new_node)
                if check_label:
                    has_old_node = has_old_node and (new_node.labels == old_node.labels)
                if has_old_node:
                    return old_node
        if return_new_node:
            return new_node
        else:
            return None

if __name__ == '__main__':
    pass

    def print_graph(g):
        print(list(g.subgraph.nodes))
        print(list(g.subgraph.relationships))
        print(g.start_rels)
        print(g.end_rels)
        print()

    a = Node('Person', name='p1')
    b = Node('Person', name='p2')
    c = Node('Person', name='p3')
    d = Node('Person', name='p4')
    ab = Relationship(a, 'knows', b)
    ac = Relationship(a, 'knows', c)

    es = ExSubgraph()
    # es.add_init_node()
    es.add_relationship(ab)
    es.add_relationship(ac)
    es.add_node(d)

    print_graph(es)
    es1 = es.copy()
    es1.remove_relationship(ab)
    # es.pop_relationship()
    # es.remove_relationship(ab)
    print_graph(es)
    print_graph(es1)


    # print(es.subgraph.relationships <= es1.subgraph.relationships)

    # print(list(es.subgraph.nodes))
    # print(list(es.subgraph.relationships))
    #
    # es.subgraph = es.subgraph - ab
    # print(list(es.subgraph.nodes))
    # print(list(es.subgraph.relationships))
    #
    # es.subgraph = es.subgraph - ac
    # print(list(es.subgraph.nodes))
    # print(list(es.subgraph.relationships))
    # print(es.subgraph.relationships)








