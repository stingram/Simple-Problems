#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

class LinearRegression {
private:
    double slope;
    double intercept;

public:
    void fit(const std::vector<double>& X, const std::vector<double>& y) {
        // Implement the least squares method to learn the parameters (slope and intercept).
        // Store the learned parameters in the 'slope' and 'intercept' member variables.

        size_t N = y.size();
        double X_mean = static_cast<double>(std::accumulate(X.begin(), X.end(),0))/N;
        double y_mean = static_cast<double>(std::accumulate(y.begin(), y.end(),0))/N;

        std::vector<double> xy_prod(N,0);
        std::transform(X.begin(),X.end(),y.begin(),xy_prod.begin(),std::multiplies<double>());
        double xy_sum = static_cast<double>(std::accumulate(xy_prod.begin(),xy_prod.end(),0))/N;

        std::vector<double> xx_prod(N,0);
        std::transform(X.begin(),X.end(),X.begin(),xx_prod.begin(),std::multiplies<double>());
        double xx_sum = static_cast<double>(std::accumulate(xx_prod.begin(),xx_prod.end(),0))/N;

        // slope = (sum(X*y)-N*y_mean*x_mean)/(sum(X^2)-N(X_mean)^2)
        slope = (xy_sum - N*X_mean*y_mean)/(xx_sum-N*X_mean*X_mean);

        // intercept = y_mean - B1*X_mean
        intercept = y_mean - slope*X_mean;
    }

    double predict(double x) const {
        // Use the learned parameters to predict the output for a given input 'x'.
        return intercept + slope*x;
    }
};

double calculateMSE(const LinearRegression& model, const std::vector<double>& X, const std::vector<double>& y) {
    // Implement a function to calculate the Mean Squared Error (MSE) of the model's predictions.
    // MSE = sum((y_true - y_pred)^2) / n
    size_t N = X.size();
    std::vector<double> MSE(N,0);
    double y_pred;
    for(size_t i = 0;i<N;i++)
    {
        y_pred = model.predict(X[i]);
        MSE[i] = std::pow(y[i]-y_pred,2);
    }
    return static_cast<double>(std::accumulate(MSE.begin(),MSE.end(),0))/N;

}

int main() {
    // Load your dataset.
    std::vector<double> X = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {2.0, 3.0, 3.5, 4.5};

    // Initialize and train the Linear Regression model.
    LinearRegression model;
    model.fit(X, y);

    // Make predictions and evaluate the model.
    double prediction = model.predict(5.0);
    double mse = calculateMSE(model, X, y);

    std::cout << "Prediction for x=5.0: " << prediction << std::endl;
    std::cout << "Mean Squared Error: " << mse << std::endl;

    return 0;
}