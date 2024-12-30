#include <iostream>

#include "convert.h"
#include "../../utils/random.h"

using namespace phez;

ConversionException::ConversionException(const std::string& _message)
{
	message = _message;
}

const char* ConversionException::what()
{
	return message.c_str();
}



int phez::get_bit_in_vec(std::vector <uint32_t>& u32s, size_t index)
{
	if (index >= 32 * u32s.size()) return -1;
	size_t offset = index % 32;
	return u32s[index / 32] << (32 - offset - 1) >> (32 - offset - 1) >> offset;
}


void phez::set_bit_in_vec(std::vector <uint32_t>& u32s, size_t index, int val)
{
	std::vector<uint32_t> bit_masks = {
		0x00000001, 0x00000002, 0x00000004, 0x00000008, 0x00000010, 0x00000020, 0x00000040, 0x00000080,
		0x00000100, 0x00000200, 0x00000400, 0x00000800, 0x00001000, 0x00002000, 0x00004000, 0x00008000,
		0x00010000, 0x00020000, 0x00040000, 0x00080000, 0x00100000, 0x00200000, 0x00400000, 0x00800000,
		0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000,
	};
	size_t u32_idx = index / 32;
	size_t offset = index % 32;
	if (val == 0) 
		u32s[u32_idx] &= ~bit_masks[offset];
	if (val == 1) 
		u32s[u32_idx] |= bit_masks[offset];
}


std::vector<uint32_t> phez::shift_u32_vector(std::vector<uint32_t>& u32s, int right_offset, int sign_bit_position)
{
	/*
	* When performing right-shift, the sign_bit_position is ignored.
	*/
	std::vector<uint32_t> shifted_vec(u32s.size(), 0);
	if (right_offset < 0)
	{
		int left_offset = - right_offset;
		for (size_t i = 0; i < u32s.size() * 32 - left_offset; i++)
			set_bit_in_vec(shifted_vec, i + left_offset, get_bit_in_vec(u32s, i));
	}

	else if (right_offset > 0)
	{
		if (sign_bit_position != -1)
		{
			if (sign_bit_position == 0)
				throw ConversionException("shift_u32_vector: sign_bit_position cannot be 0");

			for (int i = sign_bit_position - 1; i >= right_offset; i--)
				set_bit_in_vec(shifted_vec, i - right_offset, get_bit_in_vec(u32s, i));

			for (int i = sign_bit_position - 1; i >= sign_bit_position - right_offset; i--)
				set_bit_in_vec(shifted_vec, i, get_bit_in_vec(u32s, sign_bit_position));

			for (int i = u32s.size() * 32 - 1; i >= sign_bit_position; i--)
				set_bit_in_vec(shifted_vec, i, get_bit_in_vec(u32s, i));
		}
		else // There is no sign bit
		{
			for (int i = u32s.size() * 32 - 1; i >= right_offset; i--)
				set_bit_in_vec(shifted_vec, i - right_offset, get_bit_in_vec(u32s, i));
		}
	}
	else // offset = 0, simply copy the original vector
	{
		std::copy(u32s.begin(), u32s.end(), shifted_vec.begin());
	}
	return shifted_vec;
}



PackedBigNumber::PackedBigNumber() { n_elements = 0; }

PackedBigNumber::PackedBigNumber(size_t _n_elements, std::vector<BigNumber> _packed_bns) :
	n_elements(_n_elements), packed_bns(_packed_bns) {}

PackedBigNumber phez::PackedBigNumber::operator+(PackedBigNumber bn)
{
	std::vector<BigNumber> sum_bns(packed_bns.size());
	for (size_t i = 0; i < packed_bns.size(); i++)
	{
		sum_bns[i] = packed_bns[i] + bn.packed_bns[i];
	}
	return PackedBigNumber(n_elements, sum_bns);
}

PackedBigNumber phez::PackedBigNumber::operator-(PackedBigNumber bn)
{
	std::vector<BigNumber> sub_bns(packed_bns.size());
	for (size_t i = 0; i < packed_bns.size(); i++)
	{
		sub_bns[i] = packed_bns[i] - bn.packed_bns[i];
	}
	return PackedBigNumber(n_elements, sub_bns);
}

PackedBigNumber phez::PackedBigNumber::operator*(BigNumber bn)
{
	std::vector<BigNumber> prod_bns(packed_bns.size());
	for (size_t i = 0; i < packed_bns.size(); i++)
	{
		prod_bns[i] = packed_bns[i] * bn;
	}
	return PackedBigNumber(n_elements, prod_bns);
}

PackedBigNumber phez::PackedBigNumber::operator%(BigNumber bn)
{
	std::vector<BigNumber> prod_bns(packed_bns.size());
	for (size_t i = 0; i < packed_bns.size(); i++)
	{
		prod_bns[i] = packed_bns[i] % bn;
	}
	return PackedBigNumber(n_elements, prod_bns);
}



phez::Converter::Converter()
{
	Converter(23, 2, 4, 15);
}

phez::Converter::Converter(size_t _precision_bits, size_t _slot_size, size_t _slot_buffer_u32size, size_t _n_slots) :
	precision_bits(_precision_bits), slot_size(_slot_size), slot_buffer_u32size(_slot_buffer_u32size), n_slots(_n_slots)
{
	// Initialize the slot value modulus
	size_t n_u32s = slot_size / 32 + 1;
	std::vector<uint32_t> u32s_size(n_u32s, 0);
	set_bit_in_vec(u32s_size, slot_size, 1);
	bignum_modulus = BigNumber(&u32s_size[0], n_u32s);

	// Initialize the slot buffer biggest value
	bignum_max = (bignum_modulus + 1) / 2;

	scale = pow(2.0, precision_bits);
	word_scale = pow(2.0, 32);

	// std::cout << "Converter initialized, with modulus:" << bignum_modulus << " max:" << bignum_max << std::endl;
}


BigNumber phez::Converter::float_to_bignum(float value, bool make_positive)
{
	/*
	*	For anly float value, first convert it into 64-bit int, then convert it to BigNumber
	*/
	uint32_t u32s[2];
	float abs_val = abs(value);
	double encoded_val = double(abs_val) * scale;
	uint64_t u64_val = uint64_t(encoded_val);
	u32s[0] = uint32_t(u64_val & 0xffffffff);
	u32s[1] = uint32_t(u64_val >> 32);
	BigNumber bignum;
	if (make_positive)
	{
		bignum = BigNumber(&u32s[0], 2, ippBigNumPOS);
		if (value < 0) bignum = bignum_modulus - bignum;
	}
	else
	{
		bignum = BigNumber(&u32s[0], 2, value >= 0 ? IppsBigNumPOS : IppsBigNumNEG);
	}
	return bignum;
}
float phez::Converter::bignum_to_float(const BigNumber& bn)
{
	/*
	* We only consider the lowest 64-bit of the BigNumber
	*/
	BigNumber remnant = bn % bignum_modulus;
	bool is_neg = false;
	if (remnant > bignum_max)
	{
		is_neg = true;
		remnant = bignum_modulus - remnant;
	}

	std::vector<uint32_t> u32s;
	remnant.num2vec(u32s);

	float val_abs;
	if (u32s.size() == 1)
		val_abs = float((double(u32s[0])) / scale);
	else
		val_abs = float((double(u32s[0]) + double(u32s[1]) * word_scale) / scale);
	return is_neg ? -val_abs : val_abs;
}


std::vector<uint32_t> phez::Converter::to_u32s(const PackedBigNumber& packed_bn)
{
	std::vector<uint32_t> u32s;
	for (size_t i = 0; i < packed_bn.packed_bns.size(); i++)
	{
		size_t n_merged = std::min(n_slots, packed_bn.n_elements - i * n_slots);
		packed_bn.packed_bns[i].num2vec(u32s);
		u32s.resize((i * n_slots + n_merged) * slot_buffer_u32size, 0);
	}
	if (u32s.size() != packed_bn.n_elements * slot_buffer_u32size)
		throw ConversionException("to_u32s: error, vector length not match.");

	return u32s;
}

std::vector<BigNumber> phez::Converter::to_bignums(const PackedBigNumber& packed_bn)
{
	std::vector<uint32_t> u32s = to_u32s(packed_bn);
	std::vector<BigNumber> shattered_bignums;
	for (size_t i = 0; i < packed_bn.n_elements; i++)
		shattered_bignums.push_back(BigNumber(&u32s[0] + i * slot_buffer_u32size, slot_buffer_u32size));
	return shattered_bignums;
}

std::vector<std::vector<uint32_t>> phez::Converter::to_u32s_pieces(const PackedBigNumber& packed_bn)
{
	std::vector<uint32_t> u32s = to_u32s(packed_bn);
	std::vector<std::vector<uint32_t>> u32s_pieces;
	for (size_t i = 0; i < packed_bn.n_elements; i++)
		u32s_pieces.push_back(std::vector<uint32_t>(u32s.begin() + i * slot_buffer_u32size, u32s.begin() + (i + 1) * slot_buffer_u32size));
	return u32s_pieces;
}

std::vector<float> phez::Converter::to_floats(const PackedBigNumber& packed_bn)
{
	std::vector<BigNumber> bignums = to_bignums(packed_bn);
	std::vector<float> floats;
	for (size_t i = 0; i < packed_bn.n_elements; i++)
		floats.push_back(bignum_to_float(bignums[i]));
	return floats;
}


PackedBigNumber phez::Converter::pack(const std::vector<uint32_t>& u32s)
{
	if (u32s.size() % slot_buffer_u32size != 0)
		throw ConversionException("pack: u32s length not a multiple of slot_buffer_u32size");

	size_t n_elems = u32s.size() / slot_buffer_u32size;
	size_t n_bignums = u32s.size() / (n_slots * slot_buffer_u32size) + size_t(u32s.size() % (n_slots * slot_buffer_u32size) > 0);
	std::vector<BigNumber> bignums;
	for (size_t i = 0; i < n_bignums; i++)
	{
		size_t n_u32s = std::min(n_slots * slot_buffer_u32size, (u32s.size() - i * n_slots * slot_buffer_u32size));
		bignums.push_back(BigNumber(&u32s[i * n_slots * slot_buffer_u32size], n_u32s));
	}
	return PackedBigNumber(n_elems, bignums);
}


PackedBigNumber phez::Converter::pack(const std::vector<std::vector<uint32_t>>& u32s_pieces)
{
	std::vector<uint32_t> u32s;
	for (size_t i = 0; i < u32s_pieces.size(); i++)
		u32s.insert(u32s.end(), u32s_pieces[i].begin(), u32s_pieces[i].end());
	return pack(u32s);
}

PackedBigNumber phez::Converter::pack(const std::vector<BigNumber>& bignums)
{
	std::vector<std::vector<uint32_t>> u32s_pieces;
	for (size_t i = 0; i < bignums.size(); i++)
	{
		std::vector<uint32_t> u32s_piece;
		bignums[i].num2vec(u32s_piece);
		if (u32s_piece.size() > slot_buffer_u32size)
			throw ConversionException("pack: BigNumber exceeds slot buffer.");
		if (bignums[i] < BigNumber::Zero())
			throw ConversionException("pack: Cannot pack negative BigNumber.");

		u32s_piece.resize(slot_buffer_u32size, 0);
		u32s_pieces.push_back(u32s_piece);
	}
	return pack(u32s_pieces);
}

PackedBigNumber phez::Converter::pack(const std::vector<float>& floats)
{
	std::vector<BigNumber> bignums;
	for (size_t i = 0; i < floats.size(); i++)
		bignums.push_back(float_to_bignum(floats[i]));
	return pack(bignums);
}


BigNumber phez::Converter::reduce_multiplication_level(const BigNumber& bignum, int level)
{
	std::vector<uint32_t> u32s;
	bignum.num2vec(u32s);
	if (u32s.size() > slot_buffer_u32size) throw ConversionException("reduce_multiplication_level: BigNumber exceeds slot buffer");
	u32s.resize(slot_buffer_u32size, 0);
	if (level < 0)
		u32s = shift_u32_vector(u32s, level * precision_bits);
	else if (level > 0)
		u32s = shift_u32_vector(u32s, level * precision_bits, slot_size - 1);

	return BigNumber(&u32s[0], slot_buffer_u32size);
}


PackedBigNumber phez::Converter::reduce_multiplication_level(const PackedBigNumber& packed_bn, int level)
{
	std::vector<BigNumber> elems = to_bignums(packed_bn);
	for (size_t i = 0; i < elems.size(); i++)
		elems[i] = reduce_multiplication_level(elems[i], level);
	return pack(elems);
}

std::vector<PackedBigNumber> phez::Converter::shatter(const PackedBigNumber& packed_bn)
{
	if (n_slots != 1) throw ConversionException("shatter: n_slots must be 1");
	std::vector<PackedBigNumber> shattered_pbs;
	for (size_t i = 0; i < packed_bn.n_elements; i++)
		shattered_pbs.push_back(PackedBigNumber(1, { packed_bn.packed_bns[1] }));
	return shattered_pbs;
}

PackedBigNumber phez::Converter::merge(const std::vector<PackedBigNumber>& pbs)
{
	if (n_slots != 1) throw ConversionException("merge: n_slots must be 1");
	std::vector<BigNumber> bns;
	for (size_t i = 0; i < pbs.size(); i++)
		bns.push_back(pbs[i].packed_bns[0]);
	return pack(bns);
}


BigNumber phez::Converter::generate_modulus(size_t n_bits)
{
	/*
	* Generate the number 2^n_bits, whose length is n_bits + 1
	*/
	size_t n_uint32s = n_bits / 32 + 1;
	std::vector<uint32_t> modular_data(n_uint32s, 0);
	set_bit_in_vec(modular_data, n_bits, 1);
	return BigNumber(&modular_data[0], n_uint32s);
}


BigNumber phez::Converter::negate(const BigNumber& bn, size_t modlus_bits)
{
	if (modlus_bits == 0) modlus_bits = slot_size;
	BigNumber modulus = generate_modulus(modlus_bits);
	return (modulus - bn) % modulus;
}


PackedBigNumber phez::Converter::negate(const PackedBigNumber& packed_bn, size_t modulus_bits)
{
	if (modulus_bits == 0) modulus_bits = slot_size;
	BigNumber modulus = generate_modulus(modulus_bits);
	// Here we use 2^(max_bits) - 1 - n to represent the negation
	std::vector<BigNumber> elems = to_bignums(packed_bn);
	std::vector<BigNumber> negation(packed_bn.n_elements);
	for (size_t i = 0; i < packed_bn.n_elements; i++)
		negation[i] = (modulus - elems[i]) % modulus;
	return pack(negation);
}

PackedBigNumber phez::Converter::elemwise_add(PackedBigNumber& packed_bn, BigNumber& bn)
{
	std::vector<BigNumber> broadcast_bn(packed_bn.n_elements, bn);
	return packed_bn + pack(broadcast_bn);
}

PackedBigNumber phez::Converter::random_mask(size_t n_elements, size_t n_bits, bool ensure_positive)
{
	/*
	*  ensure_positive: makes the lastbit always be 1
	*/
	if (n_bits <= 3)
		throw ConversionException("random_mask: n_bits must be larger than 3");

	std::vector<std::vector<uint32_t>> random_pieces;

	for (size_t i = 0; i < n_elements; i++)
	{
		size_t n_uint32s = (n_bits / 32) + (n_bits % 32 != 0);
		if (n_uint32s > slot_buffer_u32size) throw ConversionException("random_mask: n_bits too large.");

		std::vector<uint32_t> u32s(slot_buffer_u32size, 0);
		for (size_t j = 0; j < n_uint32s; j++)
		{
			u32s[j] = zutil::rand.random_u32();
		}
		for (size_t j = n_bits; j < n_uint32s * 32; j++)
			set_bit_in_vec(u32s, j, 0);
		if (ensure_positive)
			set_bit_in_vec(u32s, n_bits - 1, 1);
		random_pieces.push_back(u32s);
	}

	return pack(random_pieces);
}


size_t phez::Converter::get_packed_nbytes(const PackedBigNumber& packed_bn)
{
	return packed_bn.packed_bns.size() * packed_bn.packed_bns[0].BitSize() / 8;
}