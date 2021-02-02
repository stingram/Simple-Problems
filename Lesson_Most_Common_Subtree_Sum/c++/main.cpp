#include <iostream>
#include <string>
#include <unordered_map>


class Node
{
    public:
        int val;
        Node* left;
        Node* right;
        Node(int val, Node* left=nullptr, Node* right=nullptr)
        {
            this->val=val;
            this->left = left;
            this->right = right;
        }
};


int subtree_sum(Node*& node, std::unordered_map<int,int>& counter)
{
    if(node == nullptr)
    {
        return 0;
    }
    int total = node->val + subtree_sum(node->left, counter) + subtree_sum(node->right, counter);
    if(counter.find(total) != counter.end())
    {
        counter[total] += 1;
    }
    else
    {
        counter[total] = 1;
    }
    return total;
}


int most_common_subtree_sum(Node*& node)
{
    std::unordered_map<int, int> counter;

    subtree_sum(node, counter);
    int most_freq = 0;
    int count = 0;

    for(auto it=counter.begin(); it!=counter.end();it++)
    {
        if(it->second>count){
            most_freq = it->first;
            count = it->second;
        }
    }

    return most_freq;
}




int main()
{
    Node* root = new Node(3, new Node(1), new Node(-3));
    std::cout << most_common_subtree_sum(root) << "\n";
    return 0;
}