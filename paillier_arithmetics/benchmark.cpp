// #include <ipcl/ipcl.hpp>
#include "/home/qjm/code/PaillierMatrix/ipcl/include/ipcl/ipcl.hpp"

#include "arithmetics.h"
#include "../utils/time_counter.hpp"
#include "../utils/ios.hpp"


int main()
{
	zutil::TimeCounter time_counter;
	ipcl::KeyPair keyPair = ipcl::generateKeypair(2048);
	phez::EncryptionScheme scheme(keyPair.pub_key, keyPair.priv_key, 3, 12, 5, 25);

	for (size_t i = 0; i < 10; i++)
	{
		phez::PackedBigNumber packed_random_bns = scheme.converter.random_mask(10000, 128);

		time_counter.start();
		phez::PackedCiphertext packed_ct = scheme.encrypt(packed_random_bns);
		std::cout << "Encrypt 10000 random numbers: " << time_counter.tick() / 1000 << "s" << std::endl;


		time_counter.start();
		phez::PackedBigNumber packed_pt = scheme.decrypt(packed_ct);
		std::cout << "Decrypt 10000 random numbers: " << time_counter.tick() / 1000 << "s" << std::endl;
	}
	return 0;
}