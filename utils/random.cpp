#include <iostream>
#include <random>
#include <iterator>
#include <algorithm>
#include "random.h"


zutil::Random::Random()
{
	std::cout << "Random initialize..." << std::endl;
	rng = std::mt19937(std::random_device()());
	std::cout << "Random initialized" << std::endl;
}

zutil::Random::Random(int seed)
{
	std::cout << "Random initialize... seed=" << seed << std::endl;
	rng = std::mt19937(seed);
	std::cout << "Random initialize..." << std::endl;
}

float zutil::Random::uniform(float low, float high)
{
	float unit_uniform = float(rng()) / float(rng.max());
	return unit_uniform * (high - low) + low;
}

int zutil::Random::uniform(int low, int high)
{
	std::uniform_int_distribution<int> dist(low, high);
	return dist(rng);
}

uint32_t zutil::Random::random_u32()
{
	return uint32_t(rng());
}

std::vector<size_t> zutil::Random::choice(size_t n_total, size_t n_samples)
{
	std::vector<size_t> original_numbers = std::vector<size_t>(n_total);
	for (size_t i = 0; i < n_total; i++) original_numbers[i] = i;
	std::vector<size_t> selected_numbers;
	if (n_total == 0 || n_samples == 0) return selected_numbers;
	std::sample(original_numbers.begin(), original_numbers.end(), std::back_inserter(selected_numbers), n_samples, rng);
	return selected_numbers;
}

zutil::Random zutil::rand(1926);