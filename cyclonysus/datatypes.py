import dionysus as ds
from typing import List, Set, Tuple, Dict, Union

# TYPES
Number = Union[int, float]
Node = Union[int, str]
Edge = Tuple[Node, Node]
Vertex = int
EdgeCycle = Union[List[Edge], List[ds.Simplex]]
NodeCycle = List[Node]
