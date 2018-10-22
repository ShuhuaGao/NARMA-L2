/**
 * @file NARMAL2Network.h
 * @author Gao Shuhua
 * @date 10/22/2018
 * @brief File defining the L2 network.
 */

#include <algorithm>
#include <cctype>
#include <random>
#include <numeric>
#include <iostream>
#include "NARMAL2Network.h"
#include "activation.h"
#include "util.h"

// uncomment the following macro definition if you want to check the gradient numerically
//#define CHECK_GRADIENT_NUMERICALLY

namespace narmal2
{
	NARMAL2Network::NARMAL2Network(int inputSize, const std::vector<int>& nHiddenNeurons1, 
		const std::vector<int>& nHiddenNeurons2, const std::string & hiddenActivation)
		: _subnets(2)
	{
		// check the activation function for hidden layers
		auto activation = capitalize(hiddenActivation);
		assert(activation == "TANH" || activation == "RELU");
		std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> activate, dactivate;
		if (activation == "TANH")
		{
			activate = tanh;
			dactivate = dtanh;
		}
		else
		{
			activate = relu;
			dactivate = drelu;
		}
		// create input layers
		_subnets[0].emplace_back(inputSize);
		_subnets[1].emplace_back(inputSize);
		// create hidden layers in each subnet
		for (auto n : nHiddenNeurons1)
		{
			_subnets[0].emplace_back(activate, dactivate, n);
		}
		for (auto n : nHiddenNeurons2)
		{
			_subnets[1].emplace_back(activate, dactivate, n);
		}
		// create the output layer in each subnet
		_subnets[0].emplace_back(identity, didentity, 1);
		_subnets[1].emplace_back(identity, didentity, 1);
	}

	NARMAL2Network::~NARMAL2Network()
	{
	}

	Statistics NARMAL2Network::trainBatch(const Eigen::MatrixXd & X, const Eigen::VectorXd & u, const Eigen::VectorXd & y, double reg, int batchSize, int numIterations, const std::string & method, const std::unordered_map<std::string, double>& methodParams, const Eigen::MatrixXd & XVal, const Eigen::VectorXd & uVal, const Eigen::VectorXd & yVal)
	{
		return train(X, u, y, reg, batchSize, numIterations, method, methodParams, true, true,
			XVal, uVal, yVal);
	}

	Statistics NARMAL2Network::train(const Eigen::MatrixXd & X, const Eigen::VectorXd & u, const Eigen::VectorXd & y, 
		double reg, int batchSize, int numIterations,
		const std::string & method, const std::unordered_map<std::string, double>& methodParams,
		bool standardize, bool resetWeights,
		const Eigen::MatrixXd & XVal, const Eigen::VectorXd & uVal, const Eigen::VectorXd & yVal)
	{
		assert(batchSize >= 1);
		assert(numIterations >= 1);

		auto it = methodParams.find("learningRate");
		if (it != methodParams.end())
			_learningRate = it->second;
		double lr = _learningRate;

		double decayRate = 1;
		it = methodParams.find("decayRate");
		if (it != methodParams.end())
			decayRate = methodParams.at("decayRate");


		auto sgdMethod = capitalize(method);
		int n = X.rows(); // number of samples
		int m = X.cols();	// length of each sample
		batchSize = std::min(batchSize, n);
		int nIterationsPerEpoch = n / batchSize;

		std::vector<int> idx(n);
		std::iota(idx.begin(), idx.end(), 0);  // [0, 1, 2, ..., m-1]
		// respresent one mini-batch
		Eigen::MatrixXd Xb(batchSize, m);
		Eigen::VectorXd ub(batchSize);
		Eigen::VectorXd yb(batchSize);

		auto Xs = X; // scaled X
		auto us = u;
		_standardize = standardize;
		if (standardize)
		{
			_mean = X.colwise().mean();
			_std = Eigen::sqrt((X.rowwise() - _mean).array().square().colwise().sum() / n);
			Xs = (X.rowwise() - _mean).array().rowwise() / _std.array();
			_meanu = u.mean();
			_stdu = std::sqrt((u.array() - _meanu).square().mean());
			us = (u.array() - _meanu) / _stdu;
		}

		if (resetWeights)
		{
			initializeWeightsRandomly();
		}
		
		Statistics stat;
		for (int iter = 0; iter < numIterations; ++iter)
		{
			// get the minbatch
			if (iter % nIterationsPerEpoch == 0)  // a new epoch. Shuffle the input samples.
			{
				std::random_device rd;
				std::mt19937 g(rd());
				std::shuffle(idx.begin(), idx.end(), g);
			}
			for (int i = 0; i < batchSize; ++i)
			{
				auto index = (iter % nIterationsPerEpoch) * batchSize + i;
				Xb.row(i) = Xs.row(idx[index]);
				ub.row(i) = us.row(idx[index]);
				yb.row(i) = y.row(idx[index]);
			}
			// train once
			auto ybp = backwardPass(Xb, ub, yb, reg);
#ifdef CHECK_GRADIENT_NUMERICALLY
			std::cout << "[Check gradient (relative error)] " << iter << std::endl;
			for (int i = 0; i < 2; i++)
				for (int j = 1; j < _subnets[i].size(); j++)
				{
					auto res = checkNumericalGradient(i, j, Xb, ub, yb, reg);
					std::cout << "\tLayer(" << i << ", " << j << ") -> "
						<< res.first << ", " << res.second << std::endl;
				}
#endif // CHECK_GRADIENT_NUMERICALLY
			// decay the learning rate per iteration
			if (iter > 0 && iter % nIterationsPerEpoch == 0)
				lr *= decayRate;
			// update the weights
			if (sgdMethod == "SGD")
			{
				naiveSGD(lr);
				
			}
			else if (sgdMethod == "MOMENTUM")
			{
				momentumSGD(lr, methodParams.at("gamma"));
			}
			// compute statistics
			auto trainLoss = computeLoss(yb, ybp, reg);
			stat.trainLossHistory.push_back(trainLoss);
			if (XVal.rows() > 0)
			{
				auto yValp = predict(XVal, uVal);
				auto valLoss = computeLoss(yVal, yValp, reg);
				stat.valLossHistory.push_back(valLoss);
			}
			
			std::cout << "[loss] " << iter << " : " << trainLoss << std::endl;
		}
		return stat;
	}

	std::pair<double, double> NARMAL2Network::checkNumericalGradient(int whichSubnet, int whichLayer,
		const Eigen::MatrixXd & X, const Eigen::VectorXd & u, const Eigen::VectorXd& y, double reg)
	{
		assert(whichSubnet == 0 || whichSubnet == 1);
		assert(whichLayer >= 1);
		const double h = 1e-6;
		auto& layer = _subnets[whichSubnet][whichLayer];
		// W
		auto dW = layer.dW;  // numerical derivative
		for (int i = 0; i < dW.rows(); i++)
			for (int j = 0; j < dW.cols(); j++)
			{
				double old = layer.W(i, j);
				// f(x+h)
				layer.W(i, j) = old + h;
				auto yp = forwardPass(X, u);
				auto lossph= computeLoss(y, yp, reg);
				// f(x - h)
				layer.W(i, j) = old - h;
				yp = forwardPass(X, u);
				auto lossmh = computeLoss(y, yp, reg);
				// numerical gradient
				dW(i, j) = (lossph - lossmh) / (2 * h);
				// reset
				layer.W(i, j) = old;
			}
		// relative error of p and q: |p-q| / (|p|+|q|). Avoid |p| and |q| being zero.
		auto t1 = (dW - layer.dW).array().abs().eval();
		auto reW = (t1 / (dW.array().abs() + layer.dW.array().abs()).max(1e-8)).eval();
		// b
		auto db = layer.db;
		for (int i = 0; i < db.size(); i++)
		{
			auto old = layer.b(i);
			layer.b(i) = old + h;
			auto yp = forwardPass(X, u);
			auto lossph = computeLoss(y, yp, reg);
			layer.b(i) = old - h;
			yp = forwardPass(X, u);
			auto lossmh = computeLoss(y, yp, reg);
			db(i) = (lossph - lossmh) / (2 * h);
			layer.b(i) = old;
		}
		auto reb = (db - layer.db).array().abs() / (db.array().abs() + layer.db.array().abs()).max(1e-8);
		return { reW.maxCoeff(), reb.maxCoeff() };
	}

	Eigen::VectorXd NARMAL2Network::predict(const Eigen::MatrixXd & X, const Eigen::VectorXd & u)
	{
		auto Xs = X;
		auto us = u;
		if (_standardize)
		{
			Xs = (X.rowwise() - _mean).array().rowwise() / _std.array();
			us = (u.array() - _meanu) / _stdu;
		}
		return forwardPass(Xs, us);
	}


	// forward pass. Compute the input and output matrices for each layer in the two subnets and cache them.
	// Also compute and return the final output of the whole network.
	Eigen::VectorXd NARMAL2Network::forwardPass(const Eigen::MatrixXd & X, const Eigen::VectorXd & u)
	{
		// the "output" of the input layer is just the training data 
		_subnets[0][0].Q = X;
		_subnets[1][0].Q = X;
		// hidden layers and output layers in each subnet
		for (auto & subnet : _subnets)
		{
			for (int i = 1; i < subnet.size(); i++)
			{
				subnet[i].P = (subnet[i - 1].Q * subnet[i].W).rowwise() + subnet[i].b;
				subnet[i].Q = subnet[i].activate(subnet[i].P);
			}
		}
		// final output
		const auto& n1 = _subnets[0].back().Q;  // column vector, output of the final layer in the 1st subnet
		const auto& n2 = _subnets[1].back().Q;
		return n1 + n2.cwiseProduct(u);
	}

	Eigen::VectorXd NARMAL2Network::backwardPass(const Eigen::MatrixXd & X, const Eigen::VectorXd & u,
		const Eigen::VectorXd& y, double reg)
	{
		auto yp = forwardPass(X, u);
		auto n = X.rows();
		// compute the gradient of the loss on n1 and n2 (output layer in each subnet)
		Eigen::MatrixXd dyp = (yp - y) / n;
		_subnets[0].back().dQ = dyp;
		_subnets[1].back().dQ = dyp.cwiseProduct(u);
		// now backpropagate from the output layer until the 1st hidden layer in each subnet
		for (auto & subnet : _subnets)
		{
			for (int i = int(subnet.size()) - 1; i >= 1; i--)
			{
				auto& layer = subnet[i];
				auto& preLayer = subnet[i - 1];
				layer.dP = layer.dQ.cwiseProduct(layer.dactivate(layer.P));
				layer.dW = preLayer.Q.transpose() * layer.dP + reg * layer.W;
				layer.db = layer.dP.colwise().sum();
				if (i - 1 > 0)  // no need to compute this for the input layer
					preLayer.dQ = layer.dP * layer.W.transpose();
			}
		}
		return yp;
	}


	void NARMAL2Network::initializeWeightsRandomly()
	{
		for (auto & subnet : _subnets)
		{
			for (int i = 1; i < subnet.size(); ++i)
			{
				int m = subnet[i - 1].nNeurons, n = subnet[i].nNeurons;
				subnet[i].W = Eigen::MatrixXd::Random(m, n) * 2;
				subnet[i].vW = Eigen::MatrixXd::Zero(subnet[i - 1].nNeurons, subnet[i].nNeurons);
				subnet[i].b = Eigen::RowVectorXd::Random(n) * 2;
				subnet[i].vb = Eigen::RowVectorXd::Zero(subnet[i].nNeurons);
			}
		}
	}

	void NARMAL2Network::naiveSGD(double eta)
	{
		assert(eta > 0);
		for (auto & subnet : _subnets)
			for (int i = 1; i < subnet.size(); i++)
			{
				subnet[i].W -= eta * subnet[i].dW;
				subnet[i].b -= eta * subnet[i].db;
			}
	}

	void NARMAL2Network::momentumSGD(double eta, double gamma)
	{
		assert(eta > 0 && gamma > 0);
		for (auto & subnet : _subnets)
			for (int i = 1; i < subnet.size(); i++)
			{
				subnet[i].vW = gamma * subnet[i].vW - eta * subnet[i].dW;
				//std::cout << "\tvW: " << subnet[i].vW.cwiseAbs().maxCoeff() << std::endl;
				subnet[i].W += subnet[i].vW;
				subnet[i].vb = gamma * subnet[i].vb - eta * subnet[i].db;
				subnet[i].b += subnet[i].vb;
				//std::cout << "\tvb: " << subnet[i].vW.cwiseAbs().maxCoeff() << std::endl;
			}
	}

	double NARMAL2Network::computeLoss(const Eigen::VectorXd & y, const Eigen::VectorXd & yp, double reg)
	{
		auto n = y.rows();
		auto dataLoss = (y - yp).array().square().sum() / n;
		double regLoss = 0;
		for (auto & subnet : _subnets)
			for (int i = 1; i < subnet.size(); i++)  // skip the input layer
			{
				regLoss += reg * subnet[i].W.squaredNorm();
			}
		return (dataLoss + regLoss) / 2;
	}
}
