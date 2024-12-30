#include <iostream>
#include "random.h"
#include "ios.hpp"


int main()
{
	std::cout << "Hello, world!" << std::endl;
	std::cout << "Test random uniform" << std::endl;
	for (size_t i = 0; i < 10; i++)
	{
		std::cout << zutil::rand.uniform(0, 10) << " ";
	}

	std::cout << "Test print bits" << std::endl;
	zutil::print_bits({ 1, 0, 32, 0xffffffff });
}