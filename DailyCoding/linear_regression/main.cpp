#include <iostream>
#include <vector>
#include <numeric>

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


        // slope = (sum(X*y)-N*y_mean*x_mean)/(sum(X^2)-N(X_mean)^2)

        // intercept = y_mean - B1*X_mean
    }

    double predict(double x) {
        // Use the learned parameters to predict the output for a given input 'x'.
    }
};

double calculateMSE(const LinearRegression& model, const std::vector<double>& X, const std::vector<double>& y) {
    // Implement a function to calculate the Mean Squared Error (MSE) of the model's predictions.
    // MSE = sum((y_true - y_pred)^2) / n
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