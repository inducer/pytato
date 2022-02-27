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
