# from __future__ import annotations

# import os
# import sys


# # Get the directory containing the script
# script_dir = os.path.dirname(__file__)

# # Add the parent directory to the Python path
# parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
# sys.path.append(parent_dir)

# import numpy as np
# from testlib import count_edges_using_dependency_mapper

# import pytato as pt


# def test_empty_dag_edge_count():
#     from pytato.analysis import get_edge_multiplicities, get_num_edges

#     empty_dag = pt.make_dict_of_named_arrays({})

#     # Verify that get_num_edges returns 0 for an empty DAG
#     assert get_num_edges(empty_dag, count_duplicates=False) == 0

#     counts = get_edge_multiplicities(empty_dag)
#     assert len(counts) == 0


# def test_single_node_dag_edge_count():
#     from pytato.analysis import get_edge_multiplicities, get_num_edges

#     data = np.random.rand(4, 4)
#     single_node_dag = pt.make_dict_of_named_arrays(
#         {"result": pt.make_data_wrapper(data)})

#     edge_counts = get_edge_multiplicities(single_node_dag)

#     # Assert that there are no edges in a single-node DAG
#     assert len(edge_counts) == 0

#     # Get total number of edges
#     total_edges = get_num_edges(single_node_dag, count_duplicates=False)

#     assert total_edges == 0


# def test_small_dag_edge_count():
#     from pytato.analysis import get_edge_multiplicities, get_num_edges

#     # Make a DAG using two nodes and one operation
#     a = pt.make_placeholder(name="a", shape=(2, 2), dtype=np.float64)
#     b = a + 1
#     dag = pt.make_dict_of_named_arrays({"result": b})   # b = a + 1

#     # Verify that get_num_edges returns 1 for a DAG with one edge
#     assert get_num_edges(dag, count_duplicates=False) == 1

#     counts = get_edge_multiplicities(dag)
#     assert len(counts) == 1
#     assert counts[(a, b)] == 1  # One edge between a and b


# def test_large_dag_edge_count():
#     from testlib import make_large_dag

#     from pytato.analysis import get_edge_multiplicities, get_num_edges

#     iterations = 100
#     dag = make_large_dag(iterations, seed=42)

#     # Verify that the number of edges is equal to the number of iterations
#     assert get_num_edges(dag, count_duplicates=False) == iterations

#     counts = get_edge_multiplicities(dag)
#     assert len(counts) == iterations


# def test_random_dag_edge_count():
#     from testlib import get_random_pt_dag

#     from pytato.analysis import get_num_edges
#     for i in range(100):
#         dag = get_random_pt_dag(seed=i, axis_len=5)

#         edge_count = get_num_edges(dag, count_duplicates=False)

#         sum_edges = count_edges_using_dependency_mapper(dag)

#         assert edge_count == sum_edges


# # def compare_edge_counts(dag):
# #     from pytato.analysis import EdgeCountMapper, get_num_edges, get_num_nodes
# #     # Use DependencyMapper to find all nodes in the graph
# #     dep_mapper = pt.transform.DependencyMapper()
# #     all_nodes = dep_mapper(dag)

# #     # Custom EdgeCountMapper
# #     custom_edge_counter = EdgeCountMapper()
# #     custom_edge_count = get_num_edges(dag, True)

# #     # DirectPredecessorsGetter edge count
# #     edge_count = 0
# #     pred_getter = pt.analysis.DirectPredecessorsGetter()

# #     processed_nodes = []
# #     for node in all_nodes:
# #         processed_nodes.append(node)
# #         direct_predecessors = list(pred_getter(node))
# #         custom_dependencies = custom_edge_counter.get_dependencies(node)

# #         print("pred getter:", len(direct_predecessors))
# #         print("custom:", len(custom_dependencies))

# #         if len(direct_predecessors) != len(custom_dependencies):
# #             print(f"Node: {node}")
# #             print(f"DirectPredecessorsGetter: {direct_predecessors}")
# #             print(f"Custom EdgeCountMapper: {custom_dependencies}")
# #             print(f"DirectPredecessorsGetter count: {len(direct_predecessors)}, Custom EdgeCountMapper count: {len(custom_dependencies)}")
# #             missing_in_predecessors = [dep for dep in custom_dependencies if dep not in direct_predecessors]
# #             extra_in_predecessors = [dep for dep in direct_predecessors if dep not in custom_dependencies]
# #             print(f"Missing in DirectPredecessorsGetter: {missing_in_predecessors}")
# #             print(f"Extra in DirectPredecessorsGetter: {extra_in_predecessors}")
# #             print("-" * 50)

# #         edge_count += len(direct_predecessors)

# #     print(f"Custom Edge Count: {custom_edge_count}")
# #     print(f"DirectPredecessorsGetter Edge Count: {edge_count}")

# #     # Print out all nodes processed
# #     print(f"Total nodes processed: {len(processed_nodes)}")
# #     print("get num nodes with no dupes:", get_num_nodes(dag, count_duplicates=False))

# #     # print(f"Processed Nodes: {processed_nodes}")

# #     return custom_edge_count, edge_count


# # def test_comparison():
# #     from testlib import get_random_pt_dag
# #     dag = get_random_pt_dag(seed=43, axis_len=5)
# #     c, e = compare_edge_counts(dag)

# #     assert c == e

#     # assert False


# def test_small_dag_with_duplicates_edge_count():
#     from testlib import make_small_dag_with_duplicates

#     from pytato.analysis import (
#         get_num_edges,
#     )

#     dag = make_small_dag_with_duplicates()

#     # Get the number of edges, including duplicates
#     edge_count = get_num_edges(dag, count_duplicates=True)
#     expected_edge_count = 3
#     assert edge_count == expected_edge_count


# def test_large_dag_with_duplicates_edge_count():
#     from testlib import make_large_dag_with_duplicates

#     from pytato.analysis import (
#         get_edge_multiplicities,
#         get_num_edges,
#     )

#     iterations = 100
#     dag = make_large_dag_with_duplicates(iterations, seed=42)

#     # Get the number of edges, including duplicates
#     edge_count = get_num_edges(dag, count_duplicates=True)

#     # Get the number of occurrences of each unique edge
#     edge_multiplicity = get_edge_multiplicities(dag)
#     assert any(count > 1 for count in edge_multiplicity.values())

#     expected_edge_count = sum(edge_multiplicity.values())
#     assert edge_count == expected_edge_count

#     # Check that duplicates are correctly calculated
#     num_duplicates = sum(count - 1 for count in edge_multiplicity.values())

#     # Ensure edge count is accurate
#     assert edge_count - num_duplicates == get_num_edges(
#         dag, count_duplicates=False)
