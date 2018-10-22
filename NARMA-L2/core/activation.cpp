/**
 * @file activation.cpp
 * @author Gao Shuhua
 * @date 10/22/2018
 * @brief File containing activation functions, including Tanh, ReLu and identity.
 */


#include "activation.h"

namespace narmal2
{
	Eigen::MatrixXd tanh(const Eigen::MatrixXd & x)
	{
		const auto& ax = x.array();
		auto ex = ax.exp();
		auto emx = (-ax).exp();
		return ((ex - emx) / (ex + emx)).matrix();
	}

	Eigen::MatrixXd dtanh(const Eigen::MatrixXd & x)
	{
		const auto& ax = x.array();
		auto ex = ax.exp();
		auto emx = (-ax).exp();
		return (4 / (ex + emx).square()).matrix();
	}

	Eigen::MatrixXd relu(const Eigen::MatrixXd & x)
	{
		return x.array().unaryExpr([](double v) {return v < 0 ? 0 : v; }).matrix();
	}

	Eigen::MatrixXd drelu(const Eigen::MatrixXd & x)
	{
		return x.array().unaryExpr([](double v) {return v < 0 ? 0 : 1; }).matrix();
	}

	Eigen::MatrixXd identity(const Eigen::MatrixXd & x)
	{
		return x;
	}
	Eigen::MatrixXd didentity(const Eigen::MatrixXd & x)
	{
		Eigen::MatrixXd ans(x.rows(), x.cols());
		ans.setOnes();
		return ans;
	}
}
