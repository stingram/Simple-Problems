#include <cstdlib>
#include <iostream>
#include <list>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>


bool Solve_AStar()
{
    // Reset Navigation Graph - default all node states
    for (int x = 0; x < nMapWidth; x++)
        for (int y = 0; y < nMapHeight; y++)
        {
            nodes[y*nMapWidth + x].bVisited = false;
            nodes[y*nMapWidth + x].fGlobalGoal = INFINITY;
            nodes[y*nMapWidth + x].fLocalGoal = INFINITY;
            nodes[y*nMapWidth + x].parent = nullptr;	// No parents
        }

    auto distance = [](sNode* a, sNode* b) // For convenience
    {
        return sqrtf((a->x - b->x)*(a->x - b->x) + (a->y - b->y)*(a->y - b->y));
    };

    auto heuristic = [distance](sNode* a, sNode* b) // So we can experiment with heuristic
    {
        return distance(a, b);
    };

    // Setup starting conditions
    sNode *nodeCurrent = nodeStart;
    nodeStart->fLocalGoal = 0.0f;
    nodeStart->fGlobalGoal = heuristic(nodeStart, nodeEnd);

    // Add start node to not tested list - this will ensure it gets tested.
    // As the algorithm progresses, newly discovered nodes get added to this
    // list, and will themselves be tested later
    list<sNode*> listNotTestedNodes;
    listNotTestedNodes.push_back(nodeStart);

    // if the not tested list contains nodes, there may be better paths
    // which have not yet been explored. However, we will also stop 
    // searching when we reach the target - there may well be better
    // paths but this one will do - it wont be the longest.
    while (!listNotTestedNodes.empty() && nodeCurrent != nodeEnd)// Find absolutely shortest path // && nodeCurrent != nodeEnd)
    {
        // Sort Untested nodes by global goal, so lowest is first
        listNotTestedNodes.sort([](const sNode* lhs, const sNode* rhs){ return lhs->fGlobalGoal < rhs->fGlobalGoal; } );
        
        // Front of listNotTestedNodes is potentially the lowest distance node. Our
        // list may also contain nodes that have been visited, so ditch these...
        while(!listNotTestedNodes.empty() && listNotTestedNodes.front()->bVisited)
            listNotTestedNodes.pop_front();

        // ...or abort because there are no valid nodes left to test
        if (listNotTestedNodes.empty())
            break;

        nodeCurrent = listNotTestedNodes.front();
        nodeCurrent->bVisited = true; // We only explore a node once
        
                
        // Check each of this node's neighbours...
        for (auto nodeNeighbour : nodeCurrent->vecNeighbours)
        {
            // ... and only if the neighbour is not visited and is 
            // not an obstacle, add it to NotTested List
            if (!nodeNeighbour->bVisited && nodeNeighbour->bObstacle == 0)
                listNotTestedNodes.push_back(nodeNeighbour);

            // Calculate the neighbours potential lowest parent distance
            float fPossiblyLowerGoal = nodeCurrent->fLocalGoal + distance(nodeCurrent, nodeNeighbour);

            // If choosing to path through this node is a lower distance than what 
            // the neighbour currently has set, update the neighbour to use this node
            // as the path source, and set its distance scores as necessary
            if (fPossiblyLowerGoal < nodeNeighbour->fLocalGoal)
            {
                nodeNeighbour->parent = nodeCurrent;
                nodeNeighbour->fLocalGoal = fPossiblyLowerGoal;

                // The best path length to the neighbour being tested has changed, so
                // update the neighbour's score. The heuristic is used to globally bias
                // the path algorithm, so it knows if its getting better or worse. At some
                // point the algo will realise this path is worse and abandon it, and then go
                // and search along the next best path.
                nodeNeighbour->fGlobalGoal = nodeNeighbour->fLocalGoal + heuristic(nodeNeighbour, nodeEnd);
            }
        }	
    }

    return true;
}