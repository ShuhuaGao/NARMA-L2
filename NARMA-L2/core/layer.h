/**
 * @file layer.h
 * @author Gao Shuhua
 * @date 10/22/2018
 * @brief Representing a layer in the network.
 */

#pragma once

#include <Eigen/Dense>
#include <functional>

namespace narmal2
{
	/**
	 * A struct representing a layer
	 */
	struct Layer
	{
		Eigen::MatrixXd P;	///< input matrix to neurons (after weighted sum)
		Eigen::MatrixXd Q;	///< output matrix of neurons (after activation function)
		Eigen::MatrixXd dP;	///< derivatives on P (same size as P)
		Eigen::MatrixXd dQ;	///< derivatives on Q	(same size as Q)

		Eigen::MatrixXd W;	///< weight matrix
		Eigen::RowVectorXd b;	///< bias vector
		Eigen::MatrixXd dW; ///< derivatives on W
		Eigen::RowVectorXd db; ///< derivatives on b

		Eigen::MatrixXd vW;  ///< previous update matrix to support SGD with momentum
		Eigen::RowVectorXd vb; ///< previous update vector to support SGD with momentum

		int nNeurons; ///< number of neurons

		std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> activate;  ///< activation function
		std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> dactivate; ///< derivative of the activation function

		Layer(std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> activate,
			std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> dactivate, int nNeurons)
		{
			this->activate = activate;
			this->dactivate = dactivate;
			this->nNeurons = nNeurons;
		}

		Layer()
		{}

		Layer(int nNeurons)
		{
			this->nNeurons = nNeurons;
		}
	};
}

