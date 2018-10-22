/**
 * @file statistic.h
 * @author Gao Shuhua
 * @date 10/22/2018
 * @brief Classes to record the statistics during training.
 */

#pragma once
#include <vector>

namespace narmal2
{
	/**
	 * A structure to record the statistics during training to help tune parameters.
	 */
	struct Statistics
	{
		std::vector<double> trainLossHistory;	///< loss history of the training set during training
		std::vector<double> valLossHistory; ///< loss history of the validation set during training
	};
}
