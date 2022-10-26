// Implement an autocomplete system. That is, given a query string s and 
// a set of all possible query strings, return all strings in the set that have s as a prefix.

// For example, given the query string de and 
// the set of strings [dog, deer, deal], return [deer, deal].

// Hint: Try preprocessing the dictionary into a more
// efficient data structure to speed up queries.

#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <stack>

class Node
{
    public:
    bool is_word;
    std::unordered_map<char,Node*> children;
    Node(const bool is_word, const std::unordered_map<char,Node*>& children)
    {
        this->is_word = is_word;
        this->children = children;
    }
};

class Trie
{
    public:
        Node* root;
        Trie()
        {
            std::unordered_map<char,Node*> children;
            this->root = new Node(false, children);
        }
        void build_tree(const std::vector<std::string>& words)
        {
            Node* curr_node;
            // loop over words
            for(std::string word: words)
            {
                curr_node = this->root;
                for(const char c : word)
                {
                    // if c not in children, add it
                    if(curr_node->children.find(c) == curr_node->children.end())
                    {
                        std::unordered_map<char,Node*> t_children;
                        curr_node->children[c] = new Node(false, t_children);
                    }
                    // update curr node
                    curr_node = curr_node->children[c];
                }
                curr_node->is_word = true;
            }
        }

        void dfs_traversal(Node*& node, const std::string prefix, std::vector<std::string>& words)
        {
            // We need to keep track of prefix and current node
            std::stack<std::pair<std::string,Node*>> s;
            s.push({prefix,node});

            while(s.size() > 0)
            {
                // pop node and examine
                std::pair<std::string,Node*> data = s.top();
                s.pop();
                if(data.second->is_word == true)
                {
                    words.push_back(data.first);
                }
                for(std::pair<char,Node*> child : data.second->children)
                {
                    s.push({data.first+child.first,child.second});
                }
            }
        }

        std::vector<std::string> find_words(const std::string prefix)
        {
            std::vector<std::string> words = {};
            Node* curr_node = this->root;
            
            // Traverse down trie
            for(const char c: prefix)
            {
                if(curr_node->children.find(c) == curr_node->children.end())
                {
                    return words;
                }
                curr_node = curr_node->children[c];
            }
            dfs_traversal(curr_node, prefix, words);
            std::cout << "HERE\n";
            return words; 
        }
};

int main()
{
    std::vector<std::string> words = {"dog", "deer", "deal"};
    Trie* trie = new Trie();
    trie->build_tree(words);
    std::vector<std::string> res = trie->find_words("de");
    for(std::string word: res)
    {
        std::cout << word << ", ";
    }
    std::cout << "\n";
    return 0;
}