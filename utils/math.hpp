#pragma once
#include <vector>
#include <iostream>
#include <algorithm>

#include "random.h"


namespace zutil
{
	template <typename T>
	inline std::vector<T> get_unique_vec(const std::vector<T>& vec)
	{
		std::vector<T> uniq_vec;
		for (size_t i = 0; i < vec.size(); i++)
			if (std::find(uniq_vec.begin(), uniq_vec.end(), vec[i]) == uniq_vec.end())
				uniq_vec.push_back(vec[i]);
		return uniq_vec;
	}

	inline void check_equal(std::vector<float>& vec0, std::vector<float>& vec1, float error = 0.01)
	{
		if (vec0.size() != vec1.size()) std::cout << "!!!!! Not match in size";
		for (size_t i = 0; i < vec0.size(); i++)
		{
			if (std::abs(vec0[i] - vec1[i]) > 0.1 * (std::abs(vec0[i]) + std::abs(vec1[i])))
			{
				if (std::abs(vec0[i] - vec1[i]) >= 10 * error)
					std::cout << "!!!!! Not match level 3 " ;
				else if (std::abs(vec0[i] - vec1[i]) >= error)
					std::cout << "!!!!! Not match level 2 ";
				else
					std::cout << "!!!!! Not match level 1 ";
				std::cout << vec0[i] << " != " << vec1[i] << " ";
			}
		}
		std::cout << std::endl;
	}

	inline std::vector<std::vector<size_t>> split_vector(size_t total_size, size_t piece_size)
	{
		std::vector<std::vector<size_t>> splitted_vectors(1);
		if (piece_size == 0)
			return splitted_vectors;
		for (size_t i = 0; i < total_size; i++)
		{
			if (splitted_vectors[splitted_vectors.size() - 1].size() == piece_size)
			{
				splitted_vectors.push_back(std::vector<size_t>());
			}
			splitted_vectors[splitted_vectors.size() - 1].push_back(i);
		}
		return splitted_vectors;
	}

	inline float sum(const std::vector<float> xs, size_t until = 0)
	{
		if (until == 0)
			until = xs.size();
		float xs_sum = 0;
		for (size_t i = 0; i < until; i++)
			xs_sum += xs[i];
		return xs_sum;
	}

	inline float mean(const std::vector<float> xs, size_t until = 0)
	{
		return sum(xs, until) / until;
	}

	inline float standard_deviation(const std::vector<float> xs)
	{
		float se_sum = 0;
		// float xs_mean = mean(xs);
		float xs_mean = mean(xs, 10);
		for (size_t i = 0; i < xs.size(); i++)
			se_sum += std::pow(xs[i] - xs_mean, 2);
		return std::sqrt(se_sum / xs.size());
	}


	inline float approximate_tanh(float x)
	{
		return std::tanh(x);
		// x = 0.3 * x;
		// return x - x * x * x / 3;
	}

	inline  float approximate_tanh_diff(float x)
	{
		return 1 - std::tanh(x) * std::tanh(x);
		// x = 0.3 * x;
		// return 1 - x * x;
	}
}