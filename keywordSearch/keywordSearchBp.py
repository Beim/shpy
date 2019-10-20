from flask import Blueprint, request
from keywordSearch.KeywordSearchEngine import KeywordSearchEngine
import json
from py2neo.data import Node
from functools import cmp_to_key
import re

keyword_search_bp = Blueprint('keywordSearch', __name__)
ks_engine = KeywordSearchEngine()

"""
POST
/keywordsearch/
{
    "query": ""
}
"""
@keyword_search_bp.route('/', methods=('POST',))
def search():
    print('get request')
    try:
        body = request.json
        query = body['query']
        subgraphs, prob = ks_engine.keyword_search(query)
        return json.dumps(parse_subgraphs(subgraphs, 10), indent=2, ensure_ascii=False)
    except Exception as e:
        print(e)
        raise(e)
        return "failed"

def parse_subgraphs(subgraphs, k):
    """
    将子图序列中每一个节点转为json，按子图排序及出现次数排序
    [
        {'名': ...},
        {...},
        ...
    ]
    :param subgraphs: list<ExSubgraph>
    :param k: int 选取前k 个子图
    :return: list<list<dict>>
    """
    subgraphs = subgraphs[:k]
    res = []
    nodes = []
    node_freq = dict()
    for graph in subgraphs:
        for rel in graph.subgraph.relationships:
            start_node = rel.start_node
            end_node = rel.end_node
            if start_node in node_freq:
                node_freq[start_node] += 1
            else:
                nodes.append(start_node)
                node_freq[start_node] = 1
            if end_node in node_freq:
                node_freq[end_node] += 1
            else:
                nodes.append(end_node)
                node_freq[end_node] = 1
    nodes.sort(key=cmp_to_key(lambda x,y: node_freq[y] - node_freq[x]))
    for node in nodes:
        node.labels
        res.append(dict(node))
    return res

    # for graph in subgraphs:
    #     rel_data = []
    #     for rel in graph.subgraph.relationships:
    #         rel.start_node.labels
    #         rel.end_node.labels
    #         start_node = dict(rel.start_node)
    #         end_node = dict(rel.end_node)
    #         rel_name = re.search(".*py2neo\.data\.(.*)'>.*", str(type(rel))).group(1)
    #         rel_data.append({
    #             'start': start_node,
    #             'end': end_node,
    #             'rel': rel_name
    #         })
    #     res.append(rel_data)
    # return res

