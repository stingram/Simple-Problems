from typing import List, Tuple, Dict

class Node:
    def __init__(self, key, val) -> None:
        self.val = val
        self.key = key
        self.prev = None
        self.next = None
        
    def __str__(self) -> str:
        return f"k: {self.key},v: {self.val}"

class LRU:
    def __init__(self, capacity:int) -> None:
        self.oldest: Node = None
        self.latest: Node = None
        self.capacity: int = capacity
        self.current_size: int = 0
        self.key_to_node_mapper: Dict[int,Node] = {}

    def __str__(self) -> str:
        s = f"\n"
        s += f"oldest: {self.oldest.key}, oldest next: {self.oldest.next.key}\n"
        s += f"latest: {self.latest.key}, latest next: {self.latest.next.key}\n"
        for k,v in self.key_to_node_mapper.items():
           s += f"{v}\n"
        return s 

    def get(self,k):
        if k in self.key_to_node_mapper:
            # print(f"keys: {list(self.key_to_node_mapper.keys())}")
            node = self.key_to_node_mapper[k]
            
            if self.current_size == 1:
                return node.val
            elif self.oldest == node:
                self.oldest = self.oldest.next
                self.latest = node
            elif self.latest == node:
                pass
            else:
                # have prev node point to node past this one
                node.prev.next = node.next

                # this node's prev needs to put to latest
                node.prev = self.latest

                # now update lastest to be this node
                self.latest = node            
                
            return node.val
        else:
            return -1
        
    def put(self,k,v):
        # case where the cache is empty
        if self.current_size == 0:
            new_node = Node(k,v)
            new_node.next = new_node
            new_node.prev = new_node
            self.key_to_node_mapper[k] = new_node

            self.oldest = new_node
            self.latest = new_node
            
            self.current_size += 1
        
        # case where cache is size 1
        if self.current_size == 1:
            # key already being used
            if k in self.key_to_node_mapper or self.capacity == 1:
                self.key_to_node_mapper[k].val = v
            else:
                first_key = list(self.key_to_node_mapper.keys())[0]
                
                # create new node and add it to cache
                new_node = Node(k,v)
                self.key_to_node_mapper[k]=new_node
                
                # update first node prev/next
                self.key_to_node_mapper[first_key].next = new_node
                self.key_to_node_mapper[first_key].prev = new_node

                # update new node prev/next
                self.key_to_node_mapper[k].next = self.key_to_node_mapper[first_key]
                self.key_to_node_mapper[k].prev = self.key_to_node_mapper[first_key]
                
                # update size
                self.current_size += 1
                
                # update latest
                self.latest = new_node
                

        # current size is > 1
        else:
            # size < capacity and k isn't in cache
            if self.current_size > 1 and self.current_size < self.capacity and k not in self.key_to_node_mapper:
                # add new node
                new_node = Node(k,v)
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
                
                # print(f"H")

            # This case covers where capacity > 1 or key is already in the cache
            else:
                # add this node if it's not already there by overwriting the oldest node
                if k not in self.key_to_node_mapper:
                    node_to_update = self.oldest
                    old_key = self.oldest.key
                    
                    # update key to node mapper
                    self.key_to_node_mapper[k] = node_to_update
                    node_to_update.key = k
                    node_to_update.val = v
                    
                    
                    self.oldest = self.oldest.next
                    self.latest = node_to_update
                    # print(f"deleting key: {old_key}")
                    del self.key_to_node_mapper[old_key]                   
                    
                else:    
                    # update this node
                    node_to_update = self.key_to_node_mapper[k]

                    node_to_update.key = k
                    node_to_update.val = v
                    
                    # if this is the oldest one, update oldest
                    if node_to_update == self.oldest:
                        # print(f"updating oldest: {self.oldest.key}, next: {self.oldest.next.key}")
                        self.oldest = self.oldest.next

                    # have prev node point to node past this one
                    node_to_update.prev.next = node_to_update.next
                    
                    # this node's next needs to point to oldest
                    node_to_update.next = self.oldest

                    # this node's prev needs to put to latest
                    node_to_update.prev = self.latest

                    # now update lastest to be this node
                    self.latest = node_to_update

def simulate_lru_cache(commands: List[List[str]]) -> None:
    for i,command in enumerate(commands):
        # print(f"command:{command}")
        if i == 0:
            lru = LRU(int(command[1]))
            continue
        op = command[0]
        if op == 'get':
            print(f"{lru.get(int(command[1]))}")
        elif op == 'put':
            lru.put(int(command[1]),int(command[2]))
            
        # print(f"STATE: {lru}")

    
operations = [
                ["LRUCache", "2"],
                ["put", "1", "1"],
                ["put", "2", "2"],
                ["get", "1"],
                ["put", "3", "3"],
                ["get", "2"],
                ["put", "4", "4"],
                ["get", "1"],
                ["get", "3"],
                ["get", "4"]
            ]
simulate_lru_cache(operations)
