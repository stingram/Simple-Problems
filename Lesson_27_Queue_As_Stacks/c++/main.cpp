#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stack>


class Queue
{
    private:
        std::stack<int> s1;
        std::stack<int> s2;
    public:
    void enqueue(int val)
    {
        s1.push(val);
    }
    int dequeue(void)
    {
        int val;
        if(!s2.empty())
        {
            val =  s2.top();
            s2.pop();
            return val;
        }
        if(!s1.empty())
        {
            while(!s1.empty())
            {
                s2.push(s1.top());
                s1.pop();
            }
            val = s2.top();
            s2.pop();
            return val;
        }
        return -1; 
    }

};


int main()
{
    Queue q = Queue();
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);
    q.enqueue(4);
    std::cout << q.dequeue() << q.dequeue()
              << q.dequeue() << q.dequeue() << "\n";
    return 0;
}