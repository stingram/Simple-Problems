#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <unordered_map>

class KNN {
private:
    std::vector<std::pair<double,double>> x;
    std::vector<int> y;
    int k;

public:
    KNN (int k) : k(k) {};
    void fit(const std::vector<std::pair<double,double>>& X, const std::vector<int>& y) {
        x = X;
        this->y=y;
    }
    double distance(const std::pair<double,double>& x1, const std::pair<double,double>& x2) const
    {
        return std::sqrt(std::pow(x1.first-x2.first,2) + std::pow(x1.second-x2.second,2));
    }
    int predict(const std::pair<double,double>& x) const {
        std::vector<double> distances(this->x.size(),0);
        for(int i=0;i<this->x.size();i++)
        {
            distances[i] = distance(x, this->x[i]);
        }
        
        // allocate vector for indices
        std::vector<int> V(this->x.size());
        std::iota(V.begin(), V.end(),0);

        // sort V based on values of distances
        std::sort(V.begin(),V.end(),[&](double i, double j) {return distances[i]< distances[j];});
        
        //first k elements of V have labels we care about
        std::unordered_map<int,int> counts;

        // count up all labels
        int curr_max = -1;
        int label_max = -1;
        for(int i=0;i<k;i++)
        {
            if(counts.find(y[V[i]])!=counts.end())
            {
                counts[y[V[i]]]++;
            }
            else
            {
                counts[y[V[i]]] = 1;
            }
            if(counts[y[V[i]]] > curr_max)
            {
                curr_max = counts[y[V[i]]];
                label_max = y[V[i]];
            }
        }
        return label_max;
    }
};

int main() {
    // Load your dataset.
    std::vector<std::pair<double,double>> X = {{1.0, 1.0}, {2.0, 2.0}, {3.0,3.0},  {4.0, 4.0}, {5.0, 5.0}};
    std::vector<int> y = {0, 0, 1, 2, 2};

    // Initialize and train the Linear Regression model.
    KNN model(3);
    model.fit(X, y);

    // Make predictions and evaluate the model.
    double prediction = model.predict({6.0,6.0});

    std::cout << "Prediction for x={6.0,6.0}: " << prediction << std::endl;

    return 0;
}