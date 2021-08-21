import heapq



def add(num, min_heap, max_heap):
    # special case where one heap has at most one value
    if len(min_heap) + len(max_heap) <= 1:
        heapq.heappush(max_heap, -num)
        return
    
    # now if we have more than 1 element, we need to get median
    median = get_median(min_heap, max_heap)
    
    # we need to compare median to this new number (num)
    if num > median:
        # add to min_heap
        heapq.heappush(min_heap, num)
    else:
        # add to max_heap
        heapq.heappush(max_heap, -num)

def rebalance(min_heap, max_heap):
    # check lengths of heaps, if difference of lengths is > 1, we need to rebalance
    # we need to move an element from the larger heap to the smaller one
    if len(min_heap) > len(max_heap) + 1:
        # min heap is too big so we move head to max_heap
        root = heapq.heappop(min_heap)
        heapq.heappush(max_heap, -root)
    elif len(max_heap) > len(min_heap) + 1:
        # max heap is too big so we move head to min_heap
        root = -heapq.heappop(max_heap) # make negative because we are keeping negative values in the max heap
        heapq.heappush(min_heap, root)
    return

def print_median(min_heap, max_heap):
    print(get_median(min_heap,max_heap))
    return 

def get_median(min_heap, max_heap):
    # we want to return the top of the heap that has the most elements if the heaps are of different size
    if(len(min_heap)>len(max_heap)):
        return min_heap[0]
    elif(len(max_heap)>len(min_heap)):
        return -max_heap[0]   # python doesn't have max heap so we are inserting negative values
    # we will return average if both are the same size
    else:
        return (min_heap[0]-max_heap[0])/2.0

def running_median(stream):
    # make two heaps
    min_heap = []
    max_heap = []

    answer = []
    
    for num in stream:
        # for each num, either add value to either min or max heap
        add(num, min_heap, max_heap)

        # rebalance if necessary
        rebalance(min_heap, max_heap)
        
        # print median
        answer.append(get_median(min_heap, max_heap))

    return answer

print(running_median([2, 1, 4, 7, 2, 0, 5]))
# [2, 1.5, 2, 3, 2, 2, 2]