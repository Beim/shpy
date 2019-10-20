from neo4jdb.ExSubgraph import ExSubgraph
from py2neo.data import Node, Relationship
import numpy as np
import random

class xNetMF():

    gd = None  # data graph
    gq = None  # query graph
    gd_nodes_num = 0
    gq_nodes_num = 0
    total_nodes_num = 0
    p = 0
    K = 0
    rs = 0.0
    ra = 0.0
    # {node1: [set(), {n2, n3}, {n4}, ...], node2: [...]}
    R = {}
    # {node1: [[], [1, 0, 3, ...]1*(b+1), ...], node2: [...]},
    # after extract: {node1: [0.5, 0.25, ...]1*(b+1), node2: [...], ...}
    d = {}
    delta = 0.5
    C = None  # similarity matrix  n*p
    all_nodes = []

    def __init__(self, gd, gq, p, K, rs, ra):
        self.gd = gd
        self.gq = gq
        if self.gd.subgraph is not None:
            self.gd_nodes_num = len(self.gd.subgraph.nodes)
        self.gq_nodes_num = len(self.gq.subgraph.nodes)
        self.total_nodes_num = self.gd_nodes_num + self.gq_nodes_num
        self.p = p
        self.K = K
        self.rs = rs
        self.ra = ra
        self.init_R()
        self.init_d()
        self.init_all_nodes()
        self.C = np.zeros([self.total_nodes_num, self.p])

    def init_R(self):
        def extract_nodes(relationship, position):
            if position == 'start':
                return set(map(lambda r: r.start_node, relationship))
            elif position == 'end':
                return set(map(lambda r: r.end_node, relationship))

        self.R = {}
        if self.gd.subgraph is not None:
            for node in self.gd.subgraph.nodes:
                self.R[node] = [set() for n in range(0, self.K + 1)]
            self.R[node][0].add(node)
        for node in self.gq.subgraph.nodes:
            self.R[node] = [set() for n in range(0, self.K + 1)]
            self.R[node][0].add(node)
        if self.gd.subgraph is not None:
            for node_u in self.gd.subgraph.nodes:
                existed_nodes = {node_u}
                for k in range(1, self.K+1):
                    if len(self.R[node_u][k - 1]) == 0:
                        break
                    for node_prev in self.R[node_u][k - 1]:
                        if node_prev in self.gd.start_rels:
                            nodes_without_duplication = extract_nodes(self.gd.start_rels[node_prev], 'end') - existed_nodes
                            self.R[node_u][k].update(nodes_without_duplication)
                        if node_prev in self.gd.end_rels:
                            nodes_without_duplication = extract_nodes(self.gd.end_rels[node_prev], 'start') - existed_nodes
                            self.R[node_u][k].update(nodes_without_duplication)
                        existed_nodes.update(self.R[node_u][k])
        for node_u in self.gq.subgraph.nodes:
            existed_nodes = {node_u}
            for k in range(1, self.K + 1):
                if len(self.R[node_u][k - 1]) == 0:
                    break
                for node_prev in self.R[node_u][k - 1]:
                    if node_prev in self.gq.start_rels:
                        nodes_without_duplication = extract_nodes(self.gq.start_rels[node_prev],
                                                                  'end') - existed_nodes
                        self.R[node_u][k].update(nodes_without_duplication)
                    if node_prev in self.gq.end_rels:
                        nodes_without_duplication = extract_nodes(self.gq.end_rels[node_prev],
                                                                  'start') - existed_nodes
                        self.R[node_u][k].update(nodes_without_duplication)
                    existed_nodes.update(self.R[node_u][k])

    def init_d(self):
        self.d = {}
        D = max(self.gq.max_degree, self.gd.max_degree, 1)
        D = max(D, 1)
        b = int(np.ceil(np.log2(D)))
        if self.gd.subgraph is not None:
            for node in self.gd.subgraph.nodes:
                self.d[node] = np.zeros([self.K + 1, b + 1])
        for node in self.gq.subgraph.nodes:
            self.d[node] = np.zeros([self.K + 1, b + 1])

    def init_all_nodes(self):
        self.all_nodes = []
        if self.gd.subgraph is not None:
            for node in self.gd.subgraph.nodes:
                self.all_nodes.append(node)
        for node in self.gq.subgraph.nodes:
            self.all_nodes.append(node)

    def run(self):
        # step1
        self.node_identity_extraction()

        # step2a
        L_idx = self.choose_landmarks_idx()
        for node_u_idx in range(0, self.total_nodes_num):
            for node_v_idx in range(0, self.p):
                self.C[node_u_idx][node_v_idx] = self.calc_similarity(self.all_nodes[node_u_idx], self.all_nodes[L_idx[node_v_idx]])

        # step2b
        W = np.zeros([self.p, self.p])
        for row in range(self.p):
            for col in range(self.p):
                W[row][col] = self.C[L_idx[row]][col]
        W1 = np.linalg.pinv(W)
        u, s, vh = np.linalg.svd(W1, full_matrices=True)
        U = u
        S = np.diag(s)
        Y = self.C.dot(U).dot(np.power(S, 0.5))
        # TODO normalize Y
        Yd = Y[0:self.gd_nodes_num]
        Yq = Y[self.gd_nodes_num:]
        return Yd, Yq

    def node_identity_extraction(self):
        if self.gd.subgraph is not None:
            for node_u in self.gd.subgraph.nodes:
                for k in range(1, self.K + 1):
                    for neighbor_node in self.R[node_u][k]:
                        degree = self.gd.node_degree(neighbor_node)
                        if degree == 0:
                            i = 0
                        else:
                            i = int(np.floor(np.log2(degree)))
                        self.d[node_u][k][i] += 1
                for k in range(1, self.K + 1):
                    self.d[node_u][0] += np.power(self.delta, k) * self.d[node_u][k]
        for node_u in self.gq.subgraph.nodes:
            for k in range(1, self.K + 1):
                for neighbor_node in self.R[node_u][k]:
                    degree = self.gq.node_degree(neighbor_node)
                    if degree == 0:
                        i = 0
                    else:
                        i = int(np.floor(np.log2(degree)))
                    self.d[node_u][k][i] += 1
            for k in range(1, self.K + 1):
                self.d[node_u][0] += np.power(self.delta, k) * self.d[node_u][k]

        for node in self.d:
            self.d[node] = self.d[node][0]
        return self.d

    '''
    :return L  # [node1, node2, ...] 1*p
    '''
    def choose_landmarks_idx(self):
        n = self.total_nodes_num
        p = self.p
        L = random.sample(range(0, n), p)
        return L

    '''
    :return similarity :double
    '''
    def calc_similarity(self, node_u, node_v):
        return 1.0

    def struc_dist(self, node_u, node_v):
        dist = np.linalg.norm(self.d[node_u] - self.d[node_v])
        return 1 - (1 / (1 + dist))

    def attr_dist(self, node_u, node_v):
        pass

    '''
    :param R_uk {n1, n2, ...}
    :param d_uk [0, 0, 0, ...] (1*D)
    '''
    @staticmethod
    def count_degree_distribution(graph, R_uk, d_uk):
        for node in R_uk:
            degree = graph.node_degree(node)
            i = int(np.floor(np.log2(degree))) + 1
            d_uk[i] += 1


if __name__ == '__main__':
    a = Node('Person', name='p1')
    b = Node('Person', name='p2')
    c = Node('Person', name='p3')
    ab = Relationship(a, 'knows', b)
    ac = Relationship(a, 'knows', c)

    d = Node('Person', name='p4')
    e = Node('Person', name='p5')
    de = Relationship(d, 'knows', e)

    gd = ExSubgraph()
    gq = ExSubgraph()
    gq.add_relationship(de)
    f = Node('Person', name='p6')
    gd.add_node(f)
    gd.add_relationship(ab)
    gd.add_relationship(ac)


    p = 3
    K = 2
    rs = 0
    ra = 1
    x = xNetMF(gd, gq, p, K, rs, ra)
    Yd, Yq = x.run()
    print(Yd)
    print(Yq)

