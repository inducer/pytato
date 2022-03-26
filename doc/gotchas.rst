Frequently Encountered Lazy Evaluation Gotchas
==============================================

Handling :class:`RecursionError`
--------------------------------

Errors such as "RecursionError: maximum recursion depth exceeded" are seen when
the computation graph for which code is being generated for has a path of the
order of 1000 intermediate arrays. Following remedies may be helpful:

- Assessing if the same result can be achieved with fewer number of array
  operations.
- Increasing the recursion limit via :func:`sys.setrecursionlimit`.
- Checking for any broken memoization implementation in the sub-classed
  :class:`pytato.transform.Mapper`.


Traversal order in a :class:`pytato.transform.Mapper`
-----------------------------------------------------

Although the direction of our DAG is similar to a data-flow graph, the
traversal order in mapper is the opposite way around i.e. the mapper method of
a node's user would be *entered* before the node's mapper method. However, the
mapper method of a node would be *returned* before it's user's mapper method is
returned. Similar traversal order is routinely seen in `visitors
<https://en.wikipedia.org/wiki/Visitor_pattern>`__ of other packages like
:mod:`pymbolic`, `pycparser <https://github.com/eliben/pycparser>`__, etc.
