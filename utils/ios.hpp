#include <iostream>
#include <vector>


namespace zutil
{
	template <class T>
	inline void print_vector(std::vector<T> vec, std::string prefix = "")
	{
		std::cout << prefix << " ";
		for (size_t i = 0; i < vec.size(); i++) std::cout << vec[i] << ",";
		std::cout << std::endl;
	}


	inline void print_bits(std::vector<uint32_t> uints, std::string prefix = "", int start = 0, int end = 0)
	{
		std::cout << prefix << " ";
		if (end == 0) end = uints.size() - 1;
		// Notice that, the vector is in the reverse order.
		for (int i = end; i >= start; i--)
		{
			uint32_t x = uints[i];
			for (int j = 31; j >= 0; j--)
			{
				std::cout << (x >> j) % 2;
			}
			std::cout << " ";
		}
		std::cout << std::endl;
	}

	inline void wait_keydown()
	{
		std::cout << "======================\n Press enter to continue...: ";
		std::cin.get();
	}
}