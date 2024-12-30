// #include <ipcl/ipcl.hpp>
# include "/home/qjm/code/PaillierMatrix/ipcl/include/ipcl/ipcl.hpp"

#include "arithmetics.h"
#include "../utils/time_counter.hpp"
#include "../utils/ios.hpp"


int main()
{
	zutil::TimeCounter timecounter;
	std::cout << "Test key generation (2048) ==========" << std::endl;
	timecounter.start();
	ipcl::KeyPair keyPair = ipcl::generateKeypair(2048);
	std::cout << "Private key: P=" << *keyPair.priv_key.getP() << ", Q=" << *keyPair.priv_key.getQ() << ", N=" << *keyPair.priv_key.getN() << std::endl;
	std::cout << "Key generation time: " << timecounter.tick() << "ms" << std::endl;

	phez::EncryptionScheme scheme(keyPair.pub_key, keyPair.priv_key, 90, 12, 5, 25);

	{
		std::cout << "Test encryption and decryption of float values ======" << std::endl;
		timecounter.start();
		std::vector<float> original_floats = { 1, -1, 2, -2, 0.89, -0.98};
		phez::PackedCiphertext encrypted_floats = scheme.encrypt(original_floats);
		std::cout << "Encryption time: " << timecounter.tick() << "ms" << std::endl;

		timecounter.start();
		std::vector<float> decrypted_floats = scheme.decrypt_to_floats(encrypted_floats);
		std::cout << "Dncryption time: " << timecounter.tick() << "ms" << std::endl;
		zutil::print_vector(decrypted_floats, "Decrypted");
	}

	{
		std::cout << "Test ciphertext add ciphertext ======" << std::endl;
		std::vector<float> floats1 = { 1.0, -1.0, 2.0, -2.0, 99, -99, 0.111, -0.222 };
		std::vector<float> floats2 = { -1, 1.23, 2.0, -2.0, 0.111, -0.222, 23, -99 };
		phez::PackedCiphertext c1 = scheme.encrypt(floats1);
		phez::PackedCiphertext c2 = scheme.encrypt(floats2);
		timecounter.start();
		phez::PackedCiphertext c_sum = scheme.add_cc(c1, c2);
		std::cout << "Ciphertext addition time: " << timecounter.tick() << "ms" << std::endl;
		zutil::print_vector(scheme.decrypt_to_floats(c_sum), "Decrypted");
	}

	{
		std::cout << "Test ciphertext mul plaintext [n times 1] =======" << std::endl;
		std::vector<float> floats = { 1.0, -1.0, 2.0, -2.0, 99, -99, -0.111, float(1) / 1145141 };
		phez::PackedCiphertext c = scheme.encrypt(floats);
		float multiplier = 0.031416;

		timecounter.start();
		phez::PackedCiphertext c_mul = scheme.mul_cp(c, multiplier);
		std::cout << "Ciphertext mul plaintext time: " << timecounter.tick() << "ms" << std::endl;

		zutil::print_vector(scheme.decrypt_to_floats(c_mul, 1), "Decrypted");
	}

	{
		std::cout << "Test ciphertext mul plaintext [1 times n] =======" << std::endl;
		std::vector<float> floats4 = { 1.0, -1.0, 2.0, -2.0, 99, -99, -0.111, float(1) / 1145141 };
		phez::PackedCiphertext c = scheme.encrypt(3.01);
		timecounter.start();
		phez::PackedCiphertext c_mul = scheme.mul_cp(c, floats4);
		std::cout << "Ciphertext mul plaintext time: " << timecounter.tick() << "ms" << std::endl;

		zutil::print_vector(scheme.decrypt_to_floats(c_mul, 1), "Decrypted");
	}
}
