/**
 * @file activation.h
 * @author Gao Shuhua
 * @date 10/22/2018
 * @brief File containing activation functions, including Tanh, ReLu and identity.
 */

#pragma once
#include <Eigen/Dense>

namespace narmal2
{
	/**
	 * Compute tanh element-wisely.
	 * @param x input matrix
	 * @return a matrix of the same size as x
	 */
	Eigen::MatrixXd tanh(const Eigen::MatrixXd& x);

	/**
	 * Compute the element-wise derivative of tanh.
	 * @param x input matrix
	 * @return derivative matrix, whose size is the same as x
	 */
	Eigen::MatrixXd dtanh(const Eigen::MatrixXd& x);

	/**
	 * Compute ReLu element-wisely.
	 * @param x input matrix
	 * @return a matrix of the same size as x
	 */
	Eigen::MatrixXd relu(const Eigen::MatrixXd& x);

	/**
	 * Compute the element-wise derivative of ReLU.
	 * @param x input matrix
	 * @return derivative matrix, whose size is the same as x
	 */
	Eigen::MatrixXd drelu(const Eigen::MatrixXd& x);

	/**
	 * Compute the identity function outputs, i.e., the output is identical to the input.
	 * @param x input matrix
	 * @return a matrix of the same size as x
	 */
	Eigen::MatrixXd identity(const Eigen::MatrixXd& x);

	/**
	 * Compute the element-wise derivative of the identity function, which is always 1.
	 * @param x input matrix
	 * @return a all-one derivative matrix, whose size is the same as x
	 */
	Eigen::MatrixXd didentity(const Eigen::MatrixXd& x);
}