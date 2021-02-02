#include <iostream>
#include <vector>


class Node
{
    public:
        int val;
        Node* left;
        Node* right;

        Node(int value)
        {
            val = value;
        }

};


std::vector<int> range(int l, int r)
{
    std::vector<int> nums;
    for(int i=l;i<r;i++)
    {
        nums.push_back(i);
    }
    return nums;
}

std::vector<Node*> gen_helper(std::vector<int>& nums)
{
    if(nums.size()==0)
    {
        return std::vector<Node*>{nullptr};
    }
    if(nums.size()==1)
    {
        return std::vector<Node*>{new Node(nums[0])};
    }

    std::vector<Node*> bsts;
    for(auto n : nums)
    {
        std::vector<int> lrange = range(nums[0],n);
        std::vector<int> rrange = range(n+1,nums.back()+1);
        std::vector<Node*> lefts = gen_helper(lrange);
        std::vector<Node*> rights = gen_helper(rrange);

        for(auto left: lefts)
        {
            for(auto right: rights)
            {
                Node* node = new Node(n);
                node->left = left;
                node->right = right;
                bsts.push_back(node);
            }
        }
    }

    return bsts;
}


std::vector<Node*> gen_trees(int N)
{
    std::vector<int> nums(N,0);
    for(int i=1;i<N+1;i++)
    {
        nums[i-1] = i;
    }
    return gen_helper(nums);
}


int main()
{
    int N = 3;
    std::vector<Node*> trees = gen_trees(N);
    std::cout << "Number of trees: " << trees.size() << "\n";
    return 0;
}