#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

class LogisticRegression {
private:
    std::vector<double> weights;
    double intercept;

    double _dot()
    {
        return 0.0;
    }

public:
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        // Assume cost function is 
        // J(theta) = 1/n*sum((y*log(y_pred) + (1-y)*log(1-y_pred))

        // Derivative are
        // dcost/dtheta = 1/n*sum(2*xj*(y_pred-y))
        //dcost/db = 1/n*sum(2*(y_pred-y))
        int n_features = X[0].size();
        int n_samples = X.size();

        weights.resize(n_samples);

        int n_iters = 10000;
        double lr = 0.001;

        std::vector<double> derivatives(n_features+1,0);
        std::vector<double> y_pred(n_samples);
        std::vector<double> y_diff(n_samples);


        for(int i =0;i<n_iters;i++)
        {
            y_pred = predict(X);
            std::transform(y_pred.begin(),y_pred.end(),y.begin(),y_diff.begin(),std::minus<double>());
            // calculate derivate for each feature
            for(int j =0; j<n_features; j++)
            {
                derivatives[j] = 1.0/double(n_samples)*2.0*_dot(_extract_column(X,j),y_diff);
            }
            derivatives[n_features] = 1.0/double(n_samples)*2.0*std::accumulate(y_diff.begin(),y_diff.end(),0.0);

            // Update parameters with the gradients
            for(int j=0;j<n_features;j++)
            {
                weights[j] -= lr*derivatives[j];
            }
            intercept -= lr*derivatives[n_features];
        }

    }

    double _dot(const std::vector<double>& left, const std::vector<double>& right)
    {
        return std::inner_product(left.begin(),left.end(),right.begin(),0);
    }

    std::vector<double> _extract_column(const std::vector<std::vector<double>>&X, int j)
    {
        std::vector<double> column(X.size());

        for(int i =0;i<X.size(); i++)
        {
            column.push_back(X[i][j]);
        }
        return column;
    }

    std::vector<double> predict(const std::vector<std::vector<double>> & X) const
    {
        std::vector<double> y_pred(X.size(),0);
        int y_index = 0;
        for(auto const & row: X)
        {
            double temp = 0.0;
            int feature_index = 0;
            for(const double& x: row)
            {
                temp += x*weights[feature_index];
                feature_index++;
            }
            temp += intercept;
            temp = 1.0/(1.0+std::exp(-temp));
            if(temp>= 0.5)
                y_pred[y_index] = 1.0;
            else
                y_pred[y_index] = 0.0;
        }
        return y_pred;
    }
};

double calculateMSE(const LogisticRegression& model, const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    // Implement a function to calculate the Mean Squared Error (MSE) of the model's predictions.
    // MSE = sum((y_true - y_pred)^2) / n
    size_t N = X.size();
    std::vector<double> MSE(N,0);
    std::vector<double> y_pred;
    y_pred = model.predict(X);
    for(size_t i = 0;i<N;i++)
    {
        
        MSE[i] = std::pow(y[i]-y_pred[i],2);
    }
    return static_cast<double>(std::accumulate(MSE.begin(),MSE.end(),0))/N;

}

int main() {
    // Load your dataset.
    std::vector<std::vector<double>> X = {{1.0, 2.0}, {30.0, 40.0}};
    std::vector<double> y = {1.0, 0.0};

    // Initialize and train the Linear Regression model.
    LogisticRegression model;
    model.fit(X, y);

    

    // Make predictions and evaluate the model.
    std::vector<double> prediction = model.predict({{5.0,6.0}});
    double mse = calculateMSE(model, X, y);

    std::cout << "Prediction for x={5.0,6.0}: " << prediction[0] << std::endl;
    std::cout << "Mean Squared Error: " << mse << std::endl;

    return 0;
}