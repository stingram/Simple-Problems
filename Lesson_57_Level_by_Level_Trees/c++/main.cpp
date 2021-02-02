#include <deque>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>


class Node
{
    public:
    int val;
    std::vector<Node*> children;
    Node(char val, std::vector<Node*> children)
    {
        this->val = val;
        this->children = children;
    }
};

class Solution
{
    public:
    std::string level_be_level(Node* root)
    {
        std::deque<Node*> q;
        q.push_back(root);
        std::string result = std::string(1,root->val) + "\n";
        while(q.empty() != true)
        {
            int num = q.size();
            while(num > 0)
            {
                Node* s = q.front();
                q.pop_front();
                for(auto n : s->children)
                {
                    q.push_back(n);
                    result += std::string(1,n->val);
                }
                num -= 1;
            }
            result += "\n";
        }
        return result;
    }
};

int main()
{
    Node* tree = new Node('a', {});
    tree->children = {new Node('b', {}), new Node('c', {})};
    tree->children[0]->children = {new Node('g', {})};
    tree->children[1]->children = {new Node('d', {}), new Node('e', {}), new Node('f', {})};


    std::cout << Solution().level_be_level(tree) << "\n";
    return 0;
}

// from collections import deque

// class Node(object):
//   def __init__(self, val, children):
//     self.val = val
//     self.children = children
    
// class Solution:
//     def level_by_level(self, root: Node) -> str:
//         q = deque()
//         q.append(root)
//         result = str(root.val) +"\n"
//         while q:
//             # We use this num to know how many items in the queue we
//             # must process before making a new line
//             num = len(q)
//             while num > 0:
//                 s = q.popleft()
//                 # print(s.val)
//                 for n in s.children:
//                     q.append(n)
//                     result += str(n.val)
//                 num -= 1
//             result += "\n"
            
//         return result
    
    
// tree = Node('a', [])
// tree.children = [Node('b', []), Node('c', [])]
// tree.children[0].children = [Node('g', [])]
// tree.children[1].children = [Node('d', []), Node('e', []), Node('f', [])]

// print(Solution().level_by_level(tree))