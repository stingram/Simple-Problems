from typing import List, Dict

class Trie:
    def __init__(self) -> None:
        self.val='$'
        self.children: Dict[str,Trie] = {}

    def __repr__(self):
        s = f""
        for k,v in self.children.items():
            s += f"{k,v}"
        return s
    def find(self, s: str):
        if s == "":
            return True
        if s[0] in self.children:
            return self.children[s[0]].find(s[1:])
        return False
    def add_word(self, word: str):
        if len(word) == 0:
            return
        if word[0] not in self.children:
            self.children[word[0]] = Trie()
        trie = self.children[word[0]]
        trie.add_word(word[1:])

def _compute_substrings(s) -> List[str]:
    res = [""]
    for i in range(1,len(s)): # for all sizes
        num_sweeps = len(s)-i+1
        for start in range(num_sweeps):
            res.append(s[start:start+i])
    return res
def filter_substrings(s: str , bad_words: List[str]) -> int:
    res = 0
    trie = Trie()
    for bad_word in bad_words:
        trie.add_word(bad_word)
    substrings = _compute_substrings(s)
    # print(f"substrings: {substrings}")
    print(f"Trie:{trie}")
    for substring in substrings:
        for i in range(len(substring)):
            if not trie.find(substring[i:]):
                substring_len = len(substring[i:])
                if substring_len > res:
                    print(f"ss:{substring[i:]}")
                    res = substring_len
    return res

s = "cbaaaabc"
bad_words = ["aaa","cb"]
print(f"{filter_substrings(s, bad_words)}")