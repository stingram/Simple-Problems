#include <iostream>
#include <vector>

class SVM {
private:
    double weight1;
    double weight2;
    double bias;
    double learning_rate;
    double lambda; 
    int n_iters;

public:
    SVM(double alpha, double lambda) : learning_rate(alpha), lambda(lambda), weight1(0.0), weight2(0.0), bias(0.0), n_iters(1000) {}

    void fit(const std::vector<std::pair<double, double>>& X, const std::vector<int>& y) {
        // Implement an optimization technique (e.g., gradient descent) to learn the SVM parameters.
        // Store the learned parameters in 'weight1', 'weight2', and 'bias'.

        // from scratch
        // https://www.youtube.com/watch?v=UX0f9BNBcsY

        // Model
        // we want some dividing hyperplane such that w*x -b = 0
        // for yi= 1, w*xi-b >= 1
        // for yi = -1, w*xi -b <= -1
        
        // Given the above, we seek a model that results in yi*(w*xi-b) >= 1 for all cases
        // yi is truth, and w*xi-b is our estimate. We want the signs to be the same so we
        // always want their product >= 1

        // Then our loss function is hinge loss, which for a single example is
        // loss_incorrect = max(0, 1 - yi*(w*xi-b))
        // l = {0, if y*f(x) >= 1
        // l = {1-y*f(x), otherwise

        // Note that the margin is 2/||w||, which is the distance between the two boundaries

        // We also add a regularlization term to keeps weights small
        // loss_r = lambda*(||w||)^2

        // So then our total cost function for all data points is
        // J = loss_r + 1/n*sum(loss_incorrect)

        // if yi*f(x) >= 1, then
        // Ji = lambda*(||w||)^2
        // else
        // Ji = lambda*(||w||)^2 + 1 - yi*(w*xi-b)


        // Now we can use the gradient for different casues to do SGD
        // https://math.stackexchange.com/questions/883016/gradient-of-2-norm-squared

        // So if yi*f(x) >= 1, then the classification is correct and we have is Ji = lambda*(||w||)^2
        // dJ_i/dw_k = 2*lambda*w_k and dJi/db = 0
        // else the function we need to find partial derivatives of is Ji = lambda*(||w||)^2 + 1 - yi*(w*xi-b)
        // dJi/dw_k = 2*lambda*w_k - y_i*x_i and dJi/db = y_i

        // for each traning sample
        // w_k_new = w_k_old - alpha*dJi/dw_k
        // b_bew = b_old - alpha*dJi/db

        // number of training points
        int m = X.size();

        double d_w1;
        double d_w2;
        double d_b;

        // do for n iterations
        // loop over all data points
        for(int n = 0; n<n_iters; n++)
        {
            d_w1 = 0;
            d_w2 = 0;
            d_b = 0;
            for(int i=0;i<m;i++)
            {
                std::pair<double,double> curr_pair = X[i];
                if(y[i]*(weight1*curr_pair.first+weight2*curr_pair.second - bias) >= 1)
                {
                    d_w1 = 2*lambda * weight1;
                    d_w2 = 2*lambda * weight2;
                    d_b = 0;
                }
                else
                {
                    d_w1 = (2*lambda * weight1 - y[i]*curr_pair.first);
                    d_w2 = (2*lambda * weight2 - y[i]*curr_pair.second);
                    d_b = y[i];
                }
                // Update parameters
                weight1 -= learning_rate*d_w1;
                weight2 -= learning_rate*d_w2;
                bias -= learning_rate*d_b;
                }
            std::cout << "Weights and Bias: " << weight1 << ", " << weight2 << ", " << bias << ".\n";
        }
        std::cout << "Weights and Bias: " << weight1 << ", " << weight2 << ", " << bias << ".\n";
    }

    int predict(double x1, double x2) const {
        // Use the learned parameters to predict the class label (+1 or -1) for a given data point.
        double val = weight1*x1+weight2*x2 - bias;
        if(val >= 1)
            return 1;
        return -1;
    }
};

double calculateAccuracy(const SVM& model, const std::vector<std::pair<double, double>>& X, const std::vector<int>& y) {
    // Implement a function to calculate the accuracy of the model on a test dataset.
    // Accuracy = (number of correctly classified data points) / (total number of data points)
    double num_data_points = X.size();
    double y_pred;
    double num_correct = 0;
    for(int i=0;i<num_data_points;i++)
    {
        y_pred = model.predict(X[i].first, X[i].second);
        if(static_cast<int>(y_pred) == y[i])
            num_correct++;
    }
    return num_correct/num_data_points; 
}

int main() {
    // Load your training dataset.
    std::vector<std::pair<double, double>> X_train = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}};
    std::vector<int> y_train = {1, 1, -1, -1};

    // Initialize and train the SVM classifier.
    SVM model(0.01, 0.01);
    model.fit(X_train, y_train);

    // Load a test dataset.
    std::vector<std::pair<double, double>> X_test = {{1.5, 2.5}, {3.5, 4.5}};

    // Evaluate the model's accuracy on the test dataset.
    double accuracy = calculateAccuracy(model, X_test, {1, -1});

    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}