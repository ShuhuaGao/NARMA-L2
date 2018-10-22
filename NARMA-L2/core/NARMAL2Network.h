/**
 * @file NARMAL2Network.h
 * @author Gao Shuhua
 * @date 10/22/2018
 * @brief File defining the L2 network.
 */

#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <unordered_map>
#include <functional>
#include "statistics.h"
#include "layer.h"

namespace narmal2
{
	class NARMAL2Network
	{
	private:
		std::vector<std::vector<Layer>> _subnets;
		Eigen::VectorXd _output;
		Eigen::RowVectorXd _mean;
		Eigen::RowVectorXd _std;
		double _meanu;
		double _stdu;
		bool _standardize = false;
		double _learningRate = 0.01;
	public:
		/**
		 * Create a new NARMA-L2 network.
		 * @param inputSize length of each input sample [y(k), ..., y(k-na), u(k-1), ..., u(k-nb)]
		 * @param nHiddenNeurons1 number of neurons in the 1st, 2nd, ..., hidden layers of the 1st subnet
		 * @param nHiddenNeurons2 number of neurons in the 1st, 2nd, ..., hidden layers of the 2nd subnet
		 * @param hiddenActivation the activation function to be used in each hidden layer.
		 *		"ReLU" (default) and "Tanh" are supported.
		 */
		NARMAL2Network(int inputSize, const std::vector<int>& nHiddenNeurons1, const std::vector<int>& nHiddenNeurons2,
			const std::string& hiddenActivation = "ReLU");
		~NARMAL2Network();

		/**
		 * Train a NARMA-L2 network in batch or mini-batch mode. 
		 * X, u and y are used for training, while XVal, uVal and yVal are used for validation if provided.
		 * @param X each row is a sample in form [y(k), ..., y(k-na), u(k-1), ..., u(k-nb)]
		 * @param u each element is the control input u[k]
		 * @param y each element is the system output y[k+d]
		 * @param reg regularization strength
		 * @param batchSize size of each mini-batch
		 * @param numIterations number of iterations for training. Each pass of a mini-batch takes an iteration
		 * @param method method for gradient descent
		 * @param methodParams specify the parameter values needed in the gradient descent method
		 * @param XVal each row is a sample in form [y(k), ..., y(k-na), u(k-1), ..., u(k-nb)]
		 * @param uVal each element is the control input u[k]
		 * @param yVal each element is the system output y[k]
		 * @note 
		 * - if method is "sgd", then two parameters are allowed: "learningRate" and "decayRate", 
		 *	where the decay rate is using to decay the learning rate after every epoch, that is, 
		 *	learningRate = learningRate * decayRate after each epoch.
		 * - if method is "momentum", then another parameter "gamma" is allowed, which is the forgetting
		 *  factor. 
		 *
		 */
		Statistics trainBatch(const Eigen::MatrixXd & X, const Eigen::VectorXd & u, const Eigen::VectorXd & y,
			double reg, int batchSize, int numIterations,
			const std::string & method, const std::unordered_map<std::string, double>& methodParams,
			const Eigen::MatrixXd& XVal = Eigen::MatrixXd{}, const Eigen::VectorXd& uVal = Eigen::VectorXd{},
			const Eigen::VectorXd& yVal = Eigen::VectorXd{});

		/**
		 * Predict the outputs given inputs.
		 * @param X each row is a sample in form [y(k), ..., y(k-na), u(k-1), ..., u(k-nb)]
		 * @param u each element is the control input u[k]
		 */
		Eigen::VectorXd predict(const Eigen::MatrixXd& X, const Eigen::VectorXd& u);

	private:
		/**
		 * Forward pass
		 * @param X each row is a sample in form [y(k), ..., y(k-na), u(k-1), ..., u(k-nb)]
		 * @param u each element is the control input u[k]
		 * @return predicted output of the network
		 */
		Eigen::VectorXd forwardPass(const Eigen::MatrixXd& X, const Eigen::VectorXd & u);

		/**
		 * Forward pass
		 * @param X each row is a sample in form [y(k), ..., y(k-na), u(k-1), ..., u(k-nb)]
		 * @param u each element is the control input u[k]
		 * @param y each element is the system output y[k]
		 * @param reg regularization strength
		 * @return predicted output of the network  in the forward pass
		 */
		Eigen::VectorXd backwardPass(const Eigen::MatrixXd& X, const Eigen::VectorXd& u, const Eigen::VectorXd& y, double reg);

		// initialize the weights of the network randomly in range [-1, 1]
		void initializeWeightsRandomly();

		/**
		 * Perform simple stochastics gradient descent to update weights
		 * @param eta learning rate
		 */
		void naiveSGD(double eta);

		/**
		 * Perform stochastics gradient descent with momentum to update weights
		 * @param eta learning rate
		 * @param gamma 
		 */
		void momentumSGD(double eta, double gamma);

		/**
		 * Compute the MSE loss with regularization
		 * @param y each element is the system output y[k]
		 * @param yp each element is the predicted system output
		 * @param reg regularization strength
		 * @return the loss, including both data loss and regularization loss
		 */
		double computeLoss(const Eigen::VectorXd& y, const Eigen::VectorXd& yp, double reg);


		/**
		 * Train a NARMA-L2 network. X, u and y are used for training, while XVal, uVal and yVal are used
		 * for
		 * @param X each row is a sample in form [y(k), ..., y(k-na), u(k-1), ..., u(k-nb)]
		 * @param u each element is the control input u[k]
		 * @param y each element is the system output y[k+d]
		 * @param reg regularization strength
		 * @param batchSize size of each mini-batch
		 * @param numIterations number of iterations for training. Each pass of a mini-batch takes an iteration
		 * @param method method for gradient descent
		 * @param methodParams specify the parameter values needed in the gradient descent method
		 * @param standardize whether rescale the data have a mean of zero and a standard deviation of one
		 * @param resetWeights whether reinitialize the weights randomly. If false, then incremental learning is supported.
		 * @param XVal each row is a sample in form [y(k), ..., y(k-na), u(k-1), ..., u(k-nb)]
		 * @param uVal each element is the control input u[k]
		 * @param yVal each element is the system output y[k]
		 */
		Statistics train(const Eigen::MatrixXd & X, const Eigen::VectorXd & u, const Eigen::VectorXd & y,
			double reg, int batchSize, int numIterations,
			const std::string & method, const std::unordered_map<std::string, double>& methodParams,
			bool standardize = false, bool resetWeights = true,
			const Eigen::MatrixXd& XVal = Eigen::MatrixXd{}, const Eigen::VectorXd& uVal = Eigen::VectorXd{},
			const Eigen::VectorXd& yVal = Eigen::VectorXd{});

		/**
		 * A utility function to verify our backward gradient computation using numerical approximation
		 * @return the relative error for W and b respectively
		 */
		std::pair<double, double> checkNumericalGradient(int whichSubnet, int whichLayer, 
			const Eigen::MatrixXd & X, const Eigen::VectorXd & u,
			const Eigen::VectorXd& y, double reg);
	};
}

