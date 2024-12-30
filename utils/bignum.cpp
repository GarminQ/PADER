#include <vector>

#include "bignum.h"


size_t zutil::get_bit_length(const BigNumber& bn)
{
	std::vector<uint32_t> u32s;
	bn.num2vec(u32s);
	uint32_t last_u32 = u32s[u32s.size() - 1];
	size_t last_bit_position = 0;
	while (last_u32 != 0)
	{
		last_u32 >>= 1;
		last_bit_position += 1;
	}
	return (u32s.size() - 1) * 32 + last_bit_position;
}