from typing import Dict, List

from collections import defaultdict

class Trie:
    def __init__(self,level=0):
        self.level = level
        self.children: Dict[str,Trie] = {}

    def _helper(self,curr):
        if len(curr)==3:
            print(f"{curr}")
            return
        for char,child in self.children.items():
            curr.append(char)
            child._helper(curr)
            del curr[-1]
        return

    def print_branches(self):
        curr = []
        self._helper(curr)

    def add(self,char,counter,parents=None):
        if self.level == 0:
            parents = []
        if self.level > 2:
            return
        node = self
        for c,child in node.children.items():
            parents.append(c)
            child.add(char,counter,parents)
            del parents[-1]
        
        if char not in node.children:
            self.children[char] = Trie(self.level+1)

            if self.level == 2:
                parents.append(char)
                counter["_".join(parents)] += 1
                del parents[-1]
                


class Solution:
    def mostVisitedPattern(self, username: List[str], timestamp: List[int], website: List[str]) -> List[str]:
        biggest = 0
        # map names to websites
        d = {}
        trie = Trie()

        # collect information for each user
        for user, time, site in zip(username,timestamp,website):
            if user not in d:
                d[user] = []
            d[user].append((time,site))

        # use trie to build counter for each activity
        counter = defaultdict(int)
        for user, activities in d.items():
            trie = Trie()
            activities.sort()

            for time,site in activities:
                trie.add(site,counter)
        del d
        del trie
        max_seqs = []
        max_count = 0
        for k,v in counter.items():
            if v > max_count:
                max_count = v
                max_seqs=[k]
            elif v == max_count:
                max_seqs.append(k)
        if len(max_seqs) > 1:
            max_seqs.sort()
        return max_seqs[0].split("_")


vals = ["a", "b", "c", "a", "a"]
# vals = ["a", "a", "a", "b"]
trie = Trie()
# for val in vals:
#     trie.add(val)

trie.print_branches()
# abc
# aba
# aca
# aaa
# bca
# baa
# caa