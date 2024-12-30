#include <omp.h>
// #include <ipcl/ipcl.hpp>
#include "/home/qjm/code/PaillierMatrix/ipcl/include/ipcl/ipcl.hpp"



int main()
{
	ipcl::KeyPair keyPair = ipcl::generateKeypair(2048);
	std::cout << "N=" << *keyPair.pub_key.getN() << std::endl;
	std::cout << "N^2=" << *keyPair.pub_key.getNSQ() << std::endl;
	std::vector<BigNumber> bignums;
	for (size_t i = 0; i < 100; i++)
	{
		bignums.push_back(BigNumber(uint32_t(i + 1)));
	}

	for (size_t t = 0; t < 100; t++)
	{
		std::cout << "Start to encrypt 100 BigNumbers ==== time " << t << std::endl;
		BigNumber N = *keyPair.pub_key.getN();
		//BigNumber moded = (N - bignums[i]) % N;
		BigNumber moded = N - bignums[t];
#pragma omp parallel for
		for (size_t i = 0; i < 100; i++)
		{
			//std::cout << "N - moded" << *keyPair.pub_key.getN() - moded << std::endl;
			//std::cout << "moded" << moded << std::endl;
			keyPair.pub_key.encrypt(ipcl::PlainText(moded));
		}
		std::cout << "Encrypted 100 BigNumbers" << std::endl;
	}
}