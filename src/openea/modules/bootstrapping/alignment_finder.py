import itertools
import time

import igraph as ig
import numpy as np


def find_potential_alignment_greedily(sim_mat, sim_th):
    return find_alignment(sim_mat, sim_th, 1)


def find_potential_alignment_mwgm(sim_mat, sim_th, k,
                                  kg1_entity_ids=None, kg2_entity_ids=None,
                                  kg1_dict=None, kg2_dict=None,
                                  heuristic=True,
                                  force_right=False,
                                  correct=True,
                                  ):
    t = time.time()
    potential_aligned_pairs = find_alignment(sim_mat, sim_th, k)
    if potential_aligned_pairs is None:
        return None
    t1 = time.time()
    if heuristic:
        selected_aligned_pairs = mwgm(
            potential_aligned_pairs, sim_mat, mwgm_graph_tool)
    else:
        selected_aligned_pairs = mwgm(
            potential_aligned_pairs, sim_mat, mwgm_igraph)
    check_new_alignment(selected_aligned_pairs, context="after mwgm")
    if force_right:
        # 输出错误对齐
        print_wrong_alignment(selected_aligned_pairs, kg1_entity_ids,
                              kg2_entity_ids, kg1_dict, kg2_dict, sim_mat)
        # TODO: only for test
        force_right_alignment(selected_aligned_pairs, correct)
        check_new_alignment(
            selected_aligned_pairs, context=f"after force right {'(delete)' if not correct else ''}")
    print("mwgm costs time: {:.3f} s".format(time.time() - t1))
    print("selecting potential alignment costs time: {:.3f} s".format(
        time.time() - t))
    return selected_aligned_pairs

def print_wrong_alignment(aligned_pairs, kg1_entity_ids, kg2_entity_ids, kg1_dict, kg2_dict, sim_mat):
    print('========== worng alignment')
    for x, y in [(x, y) for x, y in aligned_pairs if x != y][0:10]:
        if x != y: # 输出错误
            print('({}, {}, sim: {:.3f}) -> ({}, {}, sim: {:.3f})'.format(
                kg1_dict[kg1_entity_ids[x]],
                kg2_dict[kg2_entity_ids[y]],
                sim_mat[x][y],
                kg1_dict[kg1_entity_ids[x]],
                kg2_dict[kg2_entity_ids[x]],
                sim_mat[x][x],
            ))
    print('==========')


def force_right_alignment(aligned_pairs, correct=True, right_count=0, sim_mat=None, range=None, only_top=False):
    count = 0
    remove_count = 0
    top_5_count = 0
    total = 0

    check_new_alignment(aligned_pairs, context=f"验证前",
                            sim_mat=sim_mat, range=range)

    for x, y in aligned_pairs.copy():
        should_check = False
        if range is not None:
            # 错误对齐 且 在范围内
            if x != y and (range[0] <= sim_mat[x][y] <= range[1]):
                should_check = True
        else:
            # 错误对齐
            if x != y: # 强制正确
                should_check = True

        if should_check: #需要检查
            total += 1
            # sort sim_mat[x] desc and check if y is in the top 5
            is_top = False
            rank = np.argpartition(-sim_mat[x], 5)
            if x in rank[0:5]:
                is_top = True
                top_5_count += 1

            aligned_pairs.remove((x, y))

            # 需要更正 & 只更正 top
            if correct and count < right_count:
                if is_top and only_top:
                    aligned_pairs.add((x, y))
                    count += 1
                elif not only_top:
                    aligned_pairs.add((x, y))
                    count += 1
                else:
                    remove_count += 1
            else:
                break

    if range is not None:
        print(f'修正[{range[0]},{range[1]})个数: {count}, 删除个数: {remove_count}, top 5: {top_5_count} ({top_5_count / (total or 1) * 100:.2f}%)')
    else:
        print(f'修正个数: {count}, 删除个数: {remove_count}, top 5: {top_5_count} ({top_5_count / (total or 1) * 100:.2f}%)')
    return aligned_pairs


def find_alignment(sim_mat, sim_th, k):
    """
    Find potential pairs of aligned entities from the similarity matrix.
    The potential pair (x, y) should satisfy: 1) ____sim(x, y) > sim_th____; 2) y is among the nearest-k neighbors of x.

    Parameters
    ----------
    :param sim_mat:
    :param sim_th:
    :param k:
    :return:
    """
    potential_aligned_pairs = filter_sim_mat(sim_mat, sim_th)
    if len(potential_aligned_pairs) == 0:
        return None
    check_new_alignment(potential_aligned_pairs, context="after filtering by sim threshold")
    if k <= 0:
        return potential_aligned_pairs
    nearest_k_neighbors = search_nearest_k(sim_mat, k)
    potential_aligned_pairs &= nearest_k_neighbors # 从 k 个最近邻居找对齐
    if len(potential_aligned_pairs) == 0:
        return None
    check_new_alignment(potential_aligned_pairs, context="after filtering by sim and nearest k")
    return potential_aligned_pairs


def filter_sim_mat(mat, threshold, greater=True, equal=False): # 从相似矩阵相似度找大于阈值的
    if greater and equal:
        x, y = np.where(mat >= threshold)
    elif greater and not equal:
        x, y = np.where(mat > threshold)
    elif not greater and equal:
        x, y = np.where(mat <= threshold)
    else:
        x, y = np.where(mat < threshold)
    return set(zip(x, y))


def search_nearest_k(sim_mat, k):
    assert k > 0
    neighbors = set()
    num = sim_mat.shape[0]
    for i in range(num):
        rank = np.argpartition(-sim_mat[i, :], k)
        pairs = [j for j in itertools.product([i], rank[0:k])]
        neighbors |= set(pairs)
        # del rank
    assert len(neighbors) == num * k
    return neighbors


def mwgm(pairs, sim_mat, func):
    return func(pairs, sim_mat)


def mwgm_graph_tool(pairs, sim_mat):
    from graph_tool.all import Graph, max_cardinality_matching  # necessary
    if not isinstance(pairs, list):
        pairs = list(pairs)
    g = Graph()
    weight_map = g.new_edge_property("float")
    nodes_dict1 = dict()
    nodes_dict2 = dict()
    edges = list()
    for x, y in pairs:
        if x not in nodes_dict1.keys():
            n1 = g.add_vertex()
            nodes_dict1[x] = n1
        if y not in nodes_dict2.keys():
            n2 = g.add_vertex()
            nodes_dict2[y] = n2
        n1 = nodes_dict1.get(x)
        n2 = nodes_dict2.get(y)
        e = g.add_edge(n1, n2)
        edges.append(e)
        weight_map[g.edge(n1, n2)] = sim_mat[x, y]
    print("graph via graph_tool", g)
    res = max_cardinality_matching(g, heuristic=True, weight=weight_map, minimize=False)
    edge_index = np.where(res.get_array() == 1)[0].tolist()
    matched_pairs = set()
    for index in edge_index:
        matched_pairs.add(pairs[index])
    return matched_pairs


def mwgm_igraph(pairs, sim_mat):
    if not isinstance(pairs, list):
        pairs = list(pairs)
    index_id_dic1, index_id_dic2 = dict(), dict()
    index1 = set([pair[0] for pair in pairs])
    index2 = set([pair[1] for pair in pairs])
    for index in index1:
        index_id_dic1[index] = len(index_id_dic1)
    off = len(index_id_dic1)
    for index in index2:
        index_id_dic2[index] = len(index_id_dic2) + off
    assert len(index1) == len(index_id_dic1)
    assert len(index2) == len(index_id_dic2)
    edge_list = [(index_id_dic1[x], index_id_dic2[y]) for (x, y) in pairs]
    weight_list = [sim_mat[x, y] for (x, y) in pairs]
    leda_graph = ig.Graph(edge_list)
    leda_graph.vs["type"] = [0] * len(index1) + [1] * len(index2)
    leda_graph.es['weight'] = weight_list
    res = leda_graph.maximum_bipartite_matching(weights=leda_graph.es['weight'])
    print(res)
    selected_index = [e.index for e in res.edges()]
    matched_pairs = set()
    for index in selected_index:
        matched_pairs.add(pairs[index])
    return matched_pairs


def check_new_alignment(aligned_pairs, context="check alignment", sim_mat=None, range=None):
    if aligned_pairs is None or len(aligned_pairs) == 0:
        print("{}, empty aligned pairs".format(context))
        return

    if range is not None:
        num = 0
        filter_pairs_len = 0
        for x, y in aligned_pairs:
            if range[0] <= sim_mat[x][y] < range[1]:
                filter_pairs_len += 1
                if x == y: # IMPORTANT: 逐行读取数据保证了对齐实体在相似度矩阵对角线上
                    num += 1
        ratio = num / (filter_pairs_len or 1)
        print(f"{context}, range: [{range[0]},{range[1]}), right alignment: {num}/{filter_pairs_len}={ratio:.3f}")
        return

    num = 0
    for x, y in aligned_pairs:
        if x == y: # IMPORTANT: 逐行读取数据保证了对齐实体在相似度矩阵对角线上
            num += 1
    print("{}, right alignment: {}/{}={:.3f}".format(context, num, len(aligned_pairs), num / len(aligned_pairs)))
