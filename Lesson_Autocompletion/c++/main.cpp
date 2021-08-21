#include <stack>
#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>

class Node
{
    public:
    bool isWord;
    std::unordered_map<char,Node*> children;
    Node(bool is_word, const std::unordered_map<char,Node*>& children)
    {
        this->isWord=is_word;
        this->children=children;
    }
};


class Solution
{
    public:
        Node* Trie;

    void build(const std::vector<std::string>& words)
    {
        Node* trie = new Node(false, std::unordered_map<char,Node*>());
        Node* current = trie;

        // add all words to trie
        for(std::string word : words)
        {
            // add all chars to trie
            for(char& character : word)
            {
                if(current->children.find(character) == current->children.end())
                {
                    // add char to current's children list
                    current->children[character] = new Node(false, std::unordered_map<char,Node*>());
                }
                // update current
                current = current->children[character];
            }
            // set current isWord to true since we finished adding a word
            // reset current back to root
            current->isWord = true;
            current = trie;
        }
        // set Trie
        this->Trie = trie;
    }

    std::vector<std::string> autocomplete(std::string word)
    {
        std::vector<std::string> words = {};
        Node* current = this->Trie;
        for(char& character: word)
        {
            if(current->children.find(character) == current->children.end())
            {
                return words;
            }
            current = current->children[character];
        }
        // now that we're here, we can do dfs to return all words that start with prefix
        this->dfs_iterative(current, word, words);

        return words;
    }
    void dfs_iterative(Node*& node, std::string& prefix, std::vector<std::string>& words)
    {
        // create first element and push to stack
        std::stack<std::pair<Node*,std::string>> stack; 
        stack.push({node, prefix});
        
        // process stack
        while(stack.size() > 0)
        {
            std::pair<Node*,std::string> data = stack.top();
            stack.pop();
            Node* current = data.first;
            std::string prefix = data.second;
            if(current->isWord == true)
            {
                words.push_back(prefix);
            }
            for(auto& it : current->children)
            {
                std::pair<Node*,std::string> data = {it.second,prefix+it.first};
                stack.push(data);
            }

        }
    }
};



int main()
{
    Solution s = Solution();
    s.build({"dog", "dark", "cat", "door", "dodge"});
    
    std::vector<std::string> words = s.autocomplete("do");
    for(auto word: words)
    {
        std::cout << word << ", ";
    }
    std::cout << "\n";
    return 0;
}