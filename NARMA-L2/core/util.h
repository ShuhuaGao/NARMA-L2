/**
 * @file util.h
 * @author Gao Shuhua
 * @date 10/22/2018
 * @brief Some utility functions.
 */

#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <cctype>

namespace narmal2
{
	template<typename Derived>
	void writeCSV(const std::string& file, const Eigen::MatrixBase<Derived>& data)
	{
		auto os = std::ofstream(file);
		for (int i = 0; i < data.rows(); i++)
		{
			for (int j = 0; j < data.cols(); j++)
			{
				os << data(i, j) << ",";
			}
			os << std::endl;
		}
		os.close();
	}

	template<typename T>
	void writeCSV(const std::string& file, const std::vector<T>& data)
	{
		auto os = std::ofstream(file);
		for (int i = 0; i < data.size(); i++)
		{
			os << data[i] << std::endl;
			os << std::endl;
		}
		os.close();
	}

	// turn all characters in a string to upper case
	inline std::string capitalize(const std::string& str)
	{
		auto str2 = str;
		std::transform(str2.begin(), str2.end(), str2.begin(),
			[](char c) {return std::toupper(c); });
		return str2;
	}

}
