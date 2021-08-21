#include <string>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <unordered_map>


class Node
{
    public:
        int count;
        std::unordered_map<char, Node*> children;
    Node(void)
    {
        this->count = 0;
        this->children = {};
    }
};

class Trie
{
    public:
        Node* root;
        Trie(void)
        {
            this->root=new Node();
        }
        void insert(std::string word)
        {
            Node* node = root;
            for(const char & c: word)
            {
                if(node->children.find(c) == node->children.end())
                {
                    node->children[c] = new Node();
                }
                node = node->children[c];
                node->count += 1; 
            }
        }

        std::string unique_prefix(std::string word)
        {
            Node* node = root;
            std::string prefix = ""; 
            for(const char & c: word)
            {
                if(node->count == 1)
                {
                    return prefix;
                }
                else
                {
                    prefix += c;
                    node = node->children[c];
                }
            }
            return prefix;
        }
};


std::vector<std::string> find_unique_prefix(std::vector<std::string> words)
{
    Trie trie = Trie();

    for(std::string word : words)
    {
        trie.insert(word);
    }
    std::vector<std::string> res;
    for(std::string word : words)
    {
        res.push_back(trie.unique_prefix(word));
    }
    return res;

}

int main()
{
    std::vector<std::string> words = {"jon", "john", "jack", "techlead"};
    std::vector<std::string> res = find_unique_prefix(words);
    for (std::string word: res)
    {
        std::cout << word << ", ";
    }
    std::cout << "\n";
    return 0;
}
