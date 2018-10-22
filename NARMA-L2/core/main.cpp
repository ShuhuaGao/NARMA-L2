#include <Eigen/Dense>
#include <tuple>
#include <iostream>
#include <fstream>
#include <string>
#include "NARMAL2Network.h"
#include "util.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using narmal2::NARMAL2Network;
using narmal2::writeCSV;

/**
 * Generate data for the example system
 * @param n number of samples to be produced
 * @return X and y
 */
std::tuple<MatrixXd, VectorXd> generateData(const VectorXd& u)
{
	int n = u.size();
	double y0 = 0.3, y_1 = 0, u_1 = 0; // intial values
	MatrixXd X{ n, 3 };
	VectorXd y{ n + 1 };
	// compute y(1) manually: [y(k), y(k-1), u(k-1)] -> y(k+1)
	y(0) = y0;
	X.row(0) << y0, y_1, u_1;
	y(1) = 1.5 * y0 * y_1 / (1 + y0 * y_1 + y_1*y_1)
		+ std::sin(y0 + y_1) + 0.8 * u_1 + u(0) * std::cos(y0 - u_1);

	for (int k = 1; k < n; k++)
	{
		X.row(k) << y(k), y(k - 1), u(k - 1);
		y(k + 1) = 1.5 * y(k) * y(k - 1) / (1 + y(k) * y(k) + y(k - 1)*y(k - 1))
			+ std::sin(y(k) + y(k - 1)) + 0.8 * u(k - 1) + std::cos(y(k) - u(k-1)) * u(k);
	}
	return { X, y.bottomRows(n) };
}

/**
 * Generate the training or test dataset.
 * @param which "train" or "test"
 * @return X, u, and y
 */
std::tuple<MatrixXd, VectorXd, VectorXd> generateDataSet(const std::string& which)
{
	assert(which == "train" || which == "test");
	if (which == "train")
	{
		// generate data for training set
		ArrayXd k;
		int n = 5000;
		MatrixXd X;
		VectorXd y;
		k.setLinSpaced(n, 1, n);
		Eigen::VectorXd u = (1 * 3.14 / 3 * k).sin() + (2 * 3.14 / 24 * k).sin()
			+ ArrayXd::Random(n) * 0.2;
		std::tie(X, y) = generateData(u);
		return { X, u, y };
	}
	else
	{
		// generate data for test set
		int ntest = 200;
		ArrayXd k;
		k.setLinSpaced(ntest, 1, ntest); // [1, 2, ..., ntest]
		VectorXd utest = (1 * 3.14 / 5 * k).cos() + (2 * 3.14 / 25 * k).sin() 
			+ ArrayXd::Random(ntest) * 0.1;
		MatrixXd Xtest;
		VectorXd ytest;
		std::tie(Xtest, ytest) = generateData(utest);
		return { Xtest, utest, ytest };
	}
}


int main()
{
	// get training set
	MatrixXd X;
	VectorXd u, y;
	std::tie(X, u, y) = generateDataSet("train");

	// train a network: 3 inputs, 
	// the upper subnet has 30 hidden neurons and the lower 20 hidden neurons.
	NARMAL2Network net{ 3, {40}, {30}, "ReLU" };
	int batchSize = 500;
	int nIterations = 3000;
	auto stats = net.trainBatch(X, u, y, 1e-5, batchSize, nIterations, "momentum", 
		{ {"learningRate", 1e-2}, {"gamma", 0.9}, {"decayRate", 0.98} });
	auto yp = net.predict(X, y);

	// get the test set and make predictions
	MatrixXd Xtest;
	VectorXd utest, ytest;
	std::tie(Xtest, utest, ytest) = generateDataSet("test");
	auto ytestp = net.predict(Xtest, utest);
	
	// write results CSV files for further analysis using for example MATLAB
	writeCSV("trainingLoss.csv", stats.trainLossHistory);
	Eigen::MatrixXd testResults{ytest.rows(), 2 };
	testResults.col(0) = ytest;
	testResults.col(1) = ytestp;
	writeCSV("testResults.csv", testResults);
	Eigen::MatrixXd trainingResults{ yp.rows(), 2 };
	trainingResults.col(0) = y;
	trainingResults.col(1) = yp;
	writeCSV("trainingResults.csv", trainingResults);

	std::getchar();
}