from typing import List
import random
import heapq


def min_meeting_rooms(intervals: List[List[int]]) -> int:
    """
    Return the minimum number of rooms required
    to schedule all meetings.
    """
    num_rooms = 0
    intervals.sort()
    heap = []
    for (start,end) in intervals:
        # pop from heap the meeting that doesn't overlap
        # with this current interval by checking if what's
        # on top of the heap has ended before this current
        # interval
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        # push interval
        heapq.heappush(heap,end)
    
    return len(heap)


def _ref_min_meeting_rooms(intervals):
    if not intervals:
        return 0

    intervals = sorted(intervals)
    heap = []

    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)

    return len(heap)

def run_tests():
    cases = [
        ([[0,30],[5,10],[15,20]], 2),
        ([[7,10],[2,4]], 1),
        ([[1,5],[2,6],[3,7],[4,8]], 4),
        ([[1,2],[2,3],[3,4]], 1),
        ([[1,4],[2,3],[3,5]], 2),
    ]

    for intervals, want in cases:
        got = min_meeting_rooms(intervals)
        assert got == want, f"Failed: {intervals} → {got}, want {want}"

    # randomized tests
    for _ in range(200):
        n = random.randint(1, 50)
        intervals = []
        for _ in range(n):
            s = random.randint(0, 50)
            e = s + random.randint(1, 10)
            intervals.append([s,e])

        got = min_meeting_rooms(intervals)
        want = _ref_min_meeting_rooms(intervals)

        assert got == want, f"Random failed: {intervals}"

    print("✅ All tests passed!")

if __name__ == "__main__":
    run_tests()