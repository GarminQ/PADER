// PaillierMatrix.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。

#pragma once

#include <iostream>
#include <string>
#include <exception>

// TODO: 在此处引用程序需要的其他标头。
// #include <ipcl/ipcl.hpp>
#include "/home/qjm/code/PaillierMatrix/ipcl/include/ipcl/ipcl.hpp"

#include "convert/convert.h"

// #define PSEUDO_MODE

namespace phez
{
	class ArithmeticException : std::exception
	{
	public:
		ArithmeticException(const std::string& _message);
		const char* what();
	private:
		std::string message;
	};

#ifndef PSEUDO_MODE
	class PackedCiphertext
	{
	public:
		size_t n_elements;
		ipcl::CipherText ciphertext;

		PackedCiphertext();
		PackedCiphertext(size_t _n_elements, ipcl::CipherText _ciphertext);
	};

	class EncryptionScheme
	{
	public:
		ipcl::PrivateKey privKey;
		ipcl::PublicKey pubKey;
		BigNumber modulus;
		Converter converter;
		EncryptionScheme();
		EncryptionScheme(ipcl::PublicKey _pubKey, ipcl::PrivateKey _privKey, size_t _slot_size, size_t _slot_buffer_u32size, size_t _n_slots, size_t precision_bits = 8);
		PackedCiphertext encrypt(BigNumber bn);
		PackedCiphertext encrypt(float fVal);
		PackedCiphertext encrypt(const std::vector<float>& fVals);
		PackedCiphertext encrypt(const std::vector<BigNumber>& bignums);
		PackedCiphertext encrypt(const PackedBigNumber& packed_bns);

		PackedBigNumber decrypt(const PackedCiphertext& cVal);

		std::vector<BigNumber> decrypt_to_bignumbers(const PackedCiphertext& cVal, size_t multiplication_level = 0);
		std::vector<float> decrypt_to_floats(const PackedCiphertext& cVal, size_t multiplication_level = 0);

		PackedBigNumber add_pp(const PackedBigNumber& p0, const PackedBigNumber& p1);
		PackedCiphertext add_cc(const PackedCiphertext& c0, const PackedCiphertext& c1);
		PackedCiphertext mul_cp(const PackedCiphertext& ct, float pt);
		PackedCiphertext mul_cp(const PackedCiphertext& ct, const BigNumber& pt);
		PackedCiphertext mul_cp(const PackedCiphertext& ct, const std::vector<float>& pt);
		PackedCiphertext mul_cp(const PackedCiphertext& ct, const PackedBigNumber& packed_bn);

		size_t get_ciphertext_nbytes(const PackedCiphertext& ct);

		PackedBigNumber negate_on_ring(const PackedBigNumber& packed_bn);
		BigNumber random_on_ring();
		PackedBigNumber random_on_ring(const PackedCiphertext& ref);

		std::vector<PackedCiphertext> shatter(const PackedCiphertext& pc);
		PackedCiphertext merge(const std::vector<PackedCiphertext>& pcs);
	};

#else
	typedef PackedBigNumber PackedCiphertext;

	class EncryptionScheme
	{
	public:
		ipcl::PrivateKey privKey;
		ipcl::PublicKey pubKey;
		BigNumber modulus;
		Converter converter;
		EncryptionScheme();
		EncryptionScheme(ipcl::PublicKey _pubKey, ipcl::PrivateKey _privKey, size_t _slot_size, size_t _slot_buffer_u32size, size_t _n_slots, size_t precision_bits = 8);
		PackedCiphertext encrypt(BigNumber bn);
		PackedCiphertext encrypt(float fVal);
		PackedCiphertext encrypt(const std::vector<float>& fVals);
		PackedCiphertext encrypt(const std::vector<BigNumber>& bignums);
		PackedCiphertext encrypt(const PackedBigNumber& packed_bns);

		PackedBigNumber decrypt(const PackedCiphertext& cVal);

		std::vector<BigNumber> decrypt_to_bignumbers(const PackedCiphertext& cVal, size_t multiplication_level = 0);
		std::vector<float> decrypt_to_floats(const PackedCiphertext& cVal, size_t multiplication_level = 0);

		PackedBigNumber add_pp(const PackedBigNumber& p0, const PackedBigNumber& p1);
		PackedCiphertext add_cc(const PackedCiphertext& c0, const PackedCiphertext& c1);
		PackedCiphertext mul_cp(const PackedCiphertext& ct, float pt);
		PackedCiphertext mul_cp(const PackedCiphertext& ct, const BigNumber& pt);
		PackedCiphertext mul_cp(const PackedCiphertext& ct, const std::vector<float>& pt);
		PackedCiphertext mul_cp(const PackedCiphertext& ct, const PackedBigNumber& packed_bn);

		size_t get_ciphertext_nbytes(const PackedCiphertext& ct);

		PackedBigNumber negate_on_ring(const PackedBigNumber& packed_bn);
		BigNumber random_on_ring();
		PackedBigNumber random_on_ring(const PackedCiphertext& ref);

		std::vector<PackedCiphertext> shatter(const PackedCiphertext& pc);
		PackedCiphertext merge(const std::vector<PackedCiphertext>& pcs);
	};
#endif
}