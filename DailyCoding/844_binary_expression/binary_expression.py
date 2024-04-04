# A Boolean formula can be said to be satisfiable if there
# is a way to assign truth values to each variable such that
# the entire formula evaluates to true.

# For example, suppose we have the following formula,
# where the symbol ¬ is used to denote negation:

# (¬c OR b) AND (b OR c) AND (¬b OR c) AND (¬c OR ¬a)

# One way to satisfy this formula would be to let
# a = False, b = True, and c = True.

# This type of formula, with AND statements joining tuples
# containing exactly one OR, is known as 2-CNF.

# Given a 2-CNF formula, find a way to assign truth values
# to satisfy it, or return False if this is impossible.

from typing import List, Tuple, Dict, Set


def make_nodes(pair: Tuple[str,str]):
    if '~' == pair[0][0]:
        start = ""
    else:
        start = "~"
    first_node = start + pair[0][-1]
    second_node = pair[1]
    return first_node, second_node

def add_nodes(graph, first,second):
    if first not in graph:
        graph[first] = set(str)
    if second not in graph:
        graph[second] = set(str)
    graph[first].add(second)

class SSC:
    def __init__(self,graph):
        self.graph = graph
        self.n_nodes = len(self.graph)
        self.UNVISITED = -1
        self.ids = [self.UNVISITED]*self.n_nodes
        self.low = [0]*self.n_nodes
        self.on_stack =[False]*self.n_nodes
        self.stack = []
        self.visited = {}
        
    def build_SSCs(self):
        for node in self.graph.keys():
            if self.ids[node]

def find_expression(formula: List[Tuple[str,str]]) -> Tuple[bool, Dict[str,int]]:
    graph: Dict[str,Set[str]] = {}
    for pair in formula:
        # create 1st implication for the pair
        first_node, second_node = make_nodes(pair)
        
        # add nodes to graph
        add_nodes(graph, first_node,second_node)
        
        # make second implication 
        first_node, second_node = make_nodes((pair[1],pair[0]))
        
        # add nodes to graph
        add_nodes(graph, first_node, second_node)
        
    # build list of SSCs
    SSCs = SSC(graph).build_SSCs()
        
    # check satisfiable
    
    
    # assuming it's satisfiable, then we can do topological sort
    # to assign values
    
        
        
         
    return


formula = [('~c', 'b'), ('b', 'c'), ('~b', 'c'), ('~c', '~a')]

