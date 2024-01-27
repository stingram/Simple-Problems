from typing import List
import heapq
class Node():
    def __init__(self,val: int=None, next: 'Node'=None):
        self.val = val
        self.next = next

def merge_k_lists(lists: List[Node]) -> 'Node':
# add first item of all lists to the heap
    N = len(lists)
    heap = [0]*N
    for i in range(N):
    # we push/sort on the value but also include a reference to the node
        heap[i] = (lists[i].val, lists[i])
        
    
    # call heapify
    heapq.heapify(heap)

    # loop over all lists
    Head = None
    Curr = Head
    Prev = Head
    while len(heap) > 0:
    # pop node, update current
        _, node = heapq.heappop(heap)
        Curr = node

        if Head is None:
            Head = Curr

        # add new item to heap
        if node.next is not None:
            heapq.heappush(heap, (node.next.val, node.next))
        # Change the previous node.next to point to current
        if Prev is not None:
            # link previous to current
            Prev.next = Curr
        Prev = Curr
    return Head

#
l1 = Node(1,Node(2,Node(3)))
l2 = Node(4,Node(5,Node(6)))
lists=  [l1,l2]
final_list = merge_k_lists(lists)
while final_list:
    print(f"{final_list.val}->")
    final_list = final_list.next
