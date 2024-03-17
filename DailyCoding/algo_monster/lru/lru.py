from typing import List, Tuple, Dict

class Node:
    def __init__(self, val) -> None:
        self.val = val
        self.prev = None
        self.next = None

class LRU:
    def __init__(self, capacity:int) -> None:
        self.oldest: Node = None
        self.latest: Node = None
        self.capacity: int = capacity
        self.current_size: int = 0
        self.key_to_node_mapper: Dict[int,Node] = {}

    def get(self,k):
        if k in self.key_to_node_mapper:
            return self.key_to_node_mapper[k].val
        else:
            return -1
        
    def set(self,k,v):
        # case where the cache is empty
        if self.current_size == 0:
            new_node = Node(v)
            new_node.next = new_node
            new_node.prev = new_node
            self.key_to_node_mapper[k] = new_node
            self.current_size += 1
            self.oldest = new_node
            self.latest = new_node
        # case where cache is size 1
        if self.current_size == 1:
            # key already being used
            if k in self.key_to_node_mapper or self.capacity == 1:
                self.key_to_node_mapper[k].val = v
            else:
                first_key = self.key_to_node_mapper.keys()[0]
                
                # create new node and add it to cache
                new_node = Node(v)
                self.key_to_node_mapper[k]=new_node
                
                # update first node prev/next
                self.key_to_node_mapper[first_key].next = new_node
                self.key_to_node_mapper[first_key].prev = new_node

                # update new node prev/next
                self.key_to_node_mapper[k].next = self.key_to_node_mapper[first_key]
                self.key_to_node_mapper[k].prev = self.key_to_node_mapper[first_key]

        # current size is > 1
        else:
            # size < capacity and k isn't in cache
            if self.current_size > 1 and self.current_size < self.capacity and k not in self.key_to_node_mapper:
                # add new node
                new_node = Node(v)
                self.key_to_node_mapper[k] = new_node

                # make this new node the latest
                self.latest.next = new_node
                
                # make oldest prev be this node
                self.oldest.prev = new_node

                # set this new nodes next/prev
                new_node.next = self.oldest
                new_node.prev = self.latest

                # now make latest point here
                self.latest = new_node

                # update size
                self.current_size += 1

            # This case covers where capacity > 1 or key is already in the cache
            else:
                # add this node if it's not already there by overwriting the oldest node

                # update this node
                node_to_update = self.key_to_node_mapper[k]
                node_to_update.val = v

                # have prev node point to node past this one
                node_to_update.prev.next = node_to_update.next
                
                # this node's next needs to oldest
                node_to_update.next = self.oldest

                # this node's prev needs to put to latest
                node_to_update.prev = self.latest

                # now update lastest to be this node
                self.latest = node_to_update




def lru_cache(cap: int, commands: List[str]) -> None:
    lru = LRU(cap)
    for command in commands:
        items = command.split(" ")
        op = items[0]
        if op == 'get':
            lru.get(items[1])
        elif op == 'set':
            lru.set(items[1],items[2])

    
