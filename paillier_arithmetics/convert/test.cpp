#include <iostream>
#include "convert.h"
#include "../../utils/ios.hpp"



int main()
{
	std::cout << "Start convert test..." << std::endl;

	{
		std::cout << "Test big bumber mod" << std::endl;
		std::cout << "100 % 100 = " << BigNumber(100) % BigNumber(100) << std::endl;
	}

	zutil::wait_keydown();

	{
		std::cout << "Test get_bit_in_vec \n===============" << std::endl;
		std::vector<uint32_t> vec = { 0b10001001, 0b11110000 };

		std::cout << "Original bits: " << std::endl;
		zutil::print_bits(vec);

		for (size_t i = 0; i < vec.size() * 32; i++)
			std::cout << phez::get_bit_in_vec(vec, i);
		std::cout << std::endl;
	}

	zutil::wait_keydown();

	{
		std::cout << "Test shift \n===============" << std::endl;
		std::vector<uint32_t> vec = { 1, 12, 0xff, 3 };
		std::cout << "raw: ";
		zutil::print_bits(vec);

		std::cout << ">> from 0 to 50:" << std::endl;
		for (int i = 0; i <= 50; i++)
		{
			zutil::print_bits(phez::shift_u32_vector(vec, i));
		}

		zutil::wait_keydown();

		std::cout << "<< from 0 to 50:" << std::endl;
		for (int i = 0; i <= 50; i++)
		{
			zutil::print_bits(phez::shift_u32_vector(vec, -i));
		}

		zutil::wait_keydown();

		std::cout << ">> from 0 to 50 with sign_position = 64:" << std::endl;
		for (int i = 0; i <= 50; i++)
		{
			zutil::print_bits(phez::shift_u32_vector(vec, i, 64));
		}

		zutil::wait_keydown();

		std::cout << ">> from 0 to 33 with sign_position = 34:" << std::endl;
		for (int i = 0; i <= 33; i++)
		{
			zutil::print_bits(phez::shift_u32_vector(vec, i, 34));
		}

		zutil::wait_keydown();
	}

	phez::Converter converter(23, 90, 10, 6);

	{
		std::cout << "======================\n Test convert floats" << std::endl;
		std::vector<float> floats = { 0.1, 2, -3, 1314, -9999, 0.0003, -0.0001 };
		zutil::print_vector(floats, "Original");

		phez::PackedBigNumber packed_bn = converter.pack(floats);
		std::vector<float> unpacked_floats = converter.to_floats(packed_bn);
		zutil::print_vector(unpacked_floats, "Unpacked");

		for (int i = 1; i <= 2; i++)
		{
			std::cout << "Multiplication level " << i << std::endl;
			phez::PackedBigNumber packed_bn_high_level = converter.reduce_multiplication_level(packed_bn, -i);
			phez::PackedBigNumber packed_bn_recovered = converter.reduce_multiplication_level(packed_bn_high_level, i);
			std::vector<float> unpacked_floats_recovered = converter.to_floats(packed_bn_recovered);
			zutil::print_vector(unpacked_floats_recovered, "Unpacked");
		}
	}

	{
		std::cout << "======================\n Test generate modulus" << std::endl;
		for (size_t i = 31; i <= 33; i++)
			std::cout << converter.generate_modulus(i) << std::endl;
	}

	{
		std::cout << "======================\n Test negate floats" << std::endl;
		std::vector<float> floats = { 0.1, 2, -3, 1314, -9999, 0.0003, -0.0001 };
		zutil::print_vector(floats, "Original");

		phez::PackedBigNumber packed_bn = converter.negate(converter.pack(floats));
		std::vector<float> unpacked_floats = converter.to_floats(packed_bn);
		zutil::print_vector(unpacked_floats, "Unpacked");

		for (int i = 1; i <= 2; i++)
		{
			std::cout << "Multiplication level " << i << std::endl;
			phez::PackedBigNumber packed_bn_high_level = converter.reduce_multiplication_level(packed_bn, -i);
			phez::PackedBigNumber packed_bn_recovered = converter.reduce_multiplication_level(packed_bn_high_level, i);
			std::vector<float> unpacked_floats_recovered = converter.to_floats(packed_bn_recovered);
			zutil::print_vector(unpacked_floats_recovered, "Unpacked");
		}
	}

	zutil::wait_keydown();

	{
		std::cout << "=========================\n Test generating random mask...: " << std::endl;
		
		for (int i = 30; i <= 34; i ++)
		{
			{
				std::cout << i << " bits random number" << std::endl;
				std::cout << "ensure_positive = false" << std::endl;
				phez::PackedBigNumber random_bns = converter.random_mask(10, i);
0;
				std::vector<std::vector<uint32_t>> u32s_pieces = converter.to_u32s_pieces(random_bns);
				for (size_t j = 0; j < u32s_pieces.size(); j++)
					zutil::print_bits(u32s_pieces[j]);
			}

			{
				std::cout << "ensure_positive = true" << std::endl;
				phez::PackedBigNumber random_bns = converter.random_mask(10, i, true);
				std::vector<std::vector<uint32_t>> u32s_pieces = converter.to_u32s_pieces(random_bns);
				for (size_t j = 0; j < u32s_pieces.size(); j++)
					zutil::print_bits(u32s_pieces[j]);

			}

			zutil::wait_keydown();
		}
	}

	zutil::wait_keydown();
	return 0;
}
