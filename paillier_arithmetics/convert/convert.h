#pragma once

#include <math.h>
#include <stdlib.h>
#include <vector>

// #include <ipcl/bignum.h>
#include "/home/qjm/code/PaillierMatrix/ipcl/include/ipcl/bignum.h" 

namespace phez {
	class ConversionException : std::exception
	{
	public:
		ConversionException(const std::string& _message);
		const char* what();
	private:
		std::string message;
	};

	int get_bit_in_vec(std::vector <uint32_t>& u32s, size_t index);
	void set_bit_in_vec(std::vector <uint32_t>& u32s, size_t index, int val);

	std::vector<uint32_t> shift_u32_vector(std::vector <uint32_t>& u32s, int right_offset, int sign_bit_position = -1);

	class Converter;
	
	class PackedBigNumber
	{
	public:
		size_t n_elements;
		std::vector<BigNumber> packed_bns;

		PackedBigNumber();
		PackedBigNumber(size_t _n_elements, std::vector<BigNumber> _packed_bns);
		PackedBigNumber operator+(PackedBigNumber bn);
		PackedBigNumber operator-(PackedBigNumber bn);
		PackedBigNumber operator*(BigNumber bn);
		PackedBigNumber operator%(BigNumber bn);
	};


	class Converter
	{
	public:
		int precision_bits;
		/*
		* When float converts to uint64, first, it is multiplied by 2^precison_bits, then round to an int64, then cast into uint64.
		*/

		size_t slot_size; 
		/*
		* The original slot for encode float numbers
		* For example, if slot_size = 1, original value is stored in one uint32. The negative value x is represented by 2^32 - x.
		*/

		size_t slot_buffer_u32size;
		/*
		* The total reserved number of uint32s.
		* For example. computing x * y, where x and y < 2^32, then its range is 0 ~ 2^64.
		*/


		size_t n_slots;
		BigNumber bignum_modulus;
		BigNumber bignum_max;

		double scale;
		double word_scale;
		Converter();
		Converter(size_t _precision_bits, size_t slot_size = 32, size_t _slot_buffer_u32size = 3, size_t n_slots = 20);

		BigNumber float_to_bignum(float value, bool make_positive = true);
		float bignum_to_float(const BigNumber& bn);

		std::vector<uint32_t> to_u32s(const PackedBigNumber& packed_bn);
		std::vector<BigNumber> to_bignums(const PackedBigNumber& packed_bn);
		std::vector<std::vector<uint32_t>> to_u32s_pieces(const PackedBigNumber& packed_bn);
		std::vector<float> to_floats(const PackedBigNumber& packed_bn);

		PackedBigNumber pack(const std::vector<uint32_t>& u32s);
		PackedBigNumber pack(const std::vector<std::vector<uint32_t>>& u32s_pieces);
		PackedBigNumber pack(const std::vector<BigNumber>& bignums);
		PackedBigNumber pack(const std::vector<float>& values);

		BigNumber reduce_multiplication_level(const BigNumber& bignum, int level = 1);
		PackedBigNumber reduce_multiplication_level(const PackedBigNumber& packed_bn, int level = 1);

		std::vector<PackedBigNumber> shatter(const PackedBigNumber& packed_bn);
		PackedBigNumber merge(const std::vector<PackedBigNumber>& packed_bn);


		BigNumber generate_modulus(size_t n_bits);
		BigNumber negate(const BigNumber& bn, size_t max_bits = 0);
		PackedBigNumber negate(const PackedBigNumber& packed_bn, size_t max_bits = 0);
		PackedBigNumber elemwise_add(PackedBigNumber& packed_bn, BigNumber& bn);

		PackedBigNumber random_mask(size_t n_elements, size_t max_bits, bool ensure_positive = false);

		size_t get_packed_nbytes(const PackedBigNumber& packed_bn);
	};
}
