

def topological_order(nodes, parents, return_levels=False):
    r"""Return the given nodes in topological order.

    Nodes will be sorted in topological order and, secondly, if possible, 
    in their own order defined by \_\_lt\_\_ or \_\_gt\_\_.

    Args:
        nodes (list): list of nodes to sort. Each node must be hashable.
        parents (dict): dict { node: parents } with each node's parents.
        return_levels (bool): whether to also return the node level.
        
    Returns: 
        list with all nodes in topological order, if return_levels is False.
        If return_levels is True, the resulting list contains tuples
        (level, nodes) with level an int indicating the topological level
        and nodes a list with all nodes in that level.        
    """
    
    n = len(nodes)
    nodes = set(nodes)
    assert len(nodes) == n # no duplicates
    
    # Don't mutate the outside variables -> copy parents
    parents = { node: set(parents) for node, parents in parents.items() } 

    # Any parent that is not in nodes will be ignored
    for node, ps in parents.items():
        for p in list(ps): # we'll modify ps, so copy the iterator
            if p not in nodes:
                ps.remove(p) # these nodes will not be considered

    def is_sortable(obj):
        # https://stackoverflow.com/questions/19614260/
        # check-if-an-object-is-order-able-in-python
        cls = obj.__class__

        return cls.__lt__ != object.__lt__ or \
               cls.__gt__ != object.__gt__

    sort = []
    level = 0
    while parents:
        # Which nodes do not have parents at this point?
        consider = { 
            node
            for node, parents in parents.items()
            if not parents
        }

        # There should be at least one; otherwise, there's a loop
        if not consider:
            raise ValueError("Topological sorting of a cyclic graph.")

        # Now, secondary ordering, if possible:
        if all(is_sortable(c) for c in consider):
            consider = sorted(consider)
        else:
            consider = list(consider)

        # Add consider to the sorted nodes list
        sort.append((level, consider))
        level += 1

        # Finally, update parents to exclude the previously added nodes
        for node in consider:
            for node2, parents2 in parents.items():
                if node in parents2:
                    parents2.remove(node)

            parents.pop(node)
            
    if return_levels:
        return sort
    else:
        return sum(map(lambda x: x[1], sort), [])