// PaillierMatrix.cpp: 定义应用程序的入口点。
//

// #include <ipcl/bignum.h>
#include "/home/qjm/code/PaillierMatrix/ipcl/include/ipcl/bignum.h"

#include "convert/convert.h"
#include "arithmetics.h"



using namespace phez;



ArithmeticException::ArithmeticException(const std::string& _message)
{
	message = _message;
}

const char* ArithmeticException::what()
{
	return message.c_str();
}

#ifndef PSEUDO_MODE

PackedCiphertext::PackedCiphertext() { n_elements = 0; }

PackedCiphertext::PackedCiphertext(size_t _n_elements, ipcl::CipherText _ciphertext) :
	n_elements(_n_elements), ciphertext(_ciphertext) {}



EncryptionScheme::EncryptionScheme(){}

EncryptionScheme::EncryptionScheme(ipcl::PublicKey _pubKey, ipcl::PrivateKey _privKey, 
	size_t _slot_size, size_t _slot_buffer_u32size, size_t _n_slots, size_t precision_bits):
	pubKey(_pubKey), privKey(_privKey),converter(Converter(precision_bits, _slot_size, _slot_buffer_u32size, _n_slots))
{
	modulus = *_pubKey.getN();
}

PackedCiphertext EncryptionScheme::encrypt(BigNumber bn)
{
	ipcl::CipherText ct = pubKey.encrypt(ipcl::PlainText(bn));
	return PackedCiphertext(1, ct);
}

PackedCiphertext EncryptionScheme::encrypt(float fVal)
{
	ipcl::CipherText ct = pubKey.encrypt(ipcl::PlainText(converter.float_to_bignum(fVal)));
	return PackedCiphertext(1, ct);
}

PackedCiphertext EncryptionScheme::encrypt(const std::vector<float>& fVals)
{
	return encrypt(converter.pack(fVals));
}

PackedCiphertext EncryptionScheme::encrypt(const std::vector<BigNumber>& bignums)
{
	return encrypt(converter.pack(bignums));
}

PackedCiphertext EncryptionScheme::encrypt(const PackedBigNumber& packed_bn)
{
	ipcl::CipherText ct = pubKey.encrypt(ipcl::PlainText(packed_bn.packed_bns));
	return PackedCiphertext(packed_bn.n_elements, ct);
}

PackedBigNumber EncryptionScheme::decrypt(const PackedCiphertext& cVal)
{
	std::vector<BigNumber> bignums = privKey.decrypt(cVal.ciphertext).getTexts();
	return PackedBigNumber(cVal.n_elements, bignums);
}

std::vector<BigNumber> EncryptionScheme::decrypt_to_bignumbers(const PackedCiphertext& cVal, size_t multiplication_level)
{
	PackedBigNumber plaintext = converter.reduce_multiplication_level(decrypt(cVal), multiplication_level);
	return converter.to_bignums(plaintext);
}

std::vector<float> EncryptionScheme::decrypt_to_floats(const PackedCiphertext& cVal, size_t multiplication_level)
{
	PackedBigNumber plaintext = converter.reduce_multiplication_level(decrypt(cVal), multiplication_level);
	return converter.to_floats(plaintext);
}

PackedBigNumber EncryptionScheme::add_pp(const PackedBigNumber& p0, const PackedBigNumber& p1)
{
	if (p0.n_elements != p1.n_elements) throw ArithmeticException("Number of elements not match.");
	return (PackedBigNumber(p0) + PackedBigNumber(p1)) % *pubKey.getN();
}

PackedCiphertext EncryptionScheme::add_cc(const PackedCiphertext& c0, const PackedCiphertext& c1)
{
	if (c0.n_elements != c1.n_elements) throw ArithmeticException("Number of elements not match.");
	return PackedCiphertext(c0.n_elements, c0.ciphertext + c1.ciphertext);
}

PackedCiphertext phez::EncryptionScheme::mul_cp(const PackedCiphertext& ct , float pt)
{
	return PackedCiphertext(ct.n_elements, ct.ciphertext * ipcl::PlainText(converter.float_to_bignum(pt)));
}

PackedCiphertext phez::EncryptionScheme::mul_cp(const PackedCiphertext& ct, const BigNumber& pt)
{
	return PackedCiphertext(ct.n_elements, ct.ciphertext * ipcl::PlainText(pt));
}

PackedCiphertext phez::EncryptionScheme::mul_cp(const PackedCiphertext& ct, const std::vector<float>& pt)
{
	PackedBigNumber packed_bn = converter.pack(pt);
	return mul_cp(ct, packed_bn);
}

PackedCiphertext phez::EncryptionScheme::mul_cp(const PackedCiphertext& ct, const PackedBigNumber& packed_bn)
{
	if (ct.n_elements != packed_bn.n_elements && ct.n_elements != 1 && packed_bn.n_elements != 1) throw ArithmeticException("Number of elements not match");
	ipcl::PlainText pt(packed_bn.packed_bns);
	if (ct.n_elements == 1 && packed_bn.n_elements != 1)
	{
		std::vector<ipcl::CipherText> ciphertexts(packed_bn.packed_bns.size());
		for (size_t i = 0; i < packed_bn.packed_bns.size(); i++)
		{
			ciphertexts[i] = ct.ciphertext * ipcl::PlainText(pt.getElement(i));
		}
		std::vector<BigNumber> ct_bns;
		for (size_t i = 0; i < ciphertexts.size(); i++)
			ct_bns.push_back(ciphertexts[i].getElement(0));
		ipcl::CipherText prod_ct(pubKey, ct_bns);
		return PackedCiphertext(packed_bn.n_elements, prod_ct);
	}
	else 
	{
		return PackedCiphertext(ct.n_elements, ct.ciphertext * pt);
	}
}


size_t phez::EncryptionScheme::get_ciphertext_nbytes(const PackedCiphertext& ct)
{
	return ct.ciphertext.getSize() * pubKey.getBits() / 8;
}


PackedBigNumber phez::EncryptionScheme::negate_on_ring(const PackedBigNumber& packed_bn)
{
	std::vector<BigNumber> negated_bignums;
	for (size_t i = 0; i < packed_bn.packed_bns.size(); i++)
		negated_bignums.push_back((modulus - packed_bn.packed_bns[i]) % modulus);
	return PackedBigNumber(packed_bn.n_elements, negated_bignums);
}

BigNumber phez::EncryptionScheme::random_on_ring()
{
	return ipcl::getRandomBN(pubKey.getBits() + 10) % modulus;
}

PackedBigNumber phez::EncryptionScheme::random_on_ring(const PackedCiphertext& ref)
{
	std::vector<BigNumber> random_bns;
	for (size_t i = 0; i < ref.ciphertext.getSize(); i++)
		random_bns.push_back(random_on_ring());
	return phez::PackedBigNumber(ref.n_elements, random_bns);
}

std::vector<PackedCiphertext> phez::EncryptionScheme::shatter(const PackedCiphertext& pc)
{
	if (converter.n_slots != 1) throw ArithmeticException("shatter: converter.n_slots must be 1");
	std::vector<PackedCiphertext> pcs;
	for (size_t i = 0; i < pc.n_elements; i++)
		pcs.push_back(PackedCiphertext(1, pc.ciphertext.getCipherText(i)));
	return pcs;
}

PackedCiphertext phez::EncryptionScheme::merge(const std::vector<PackedCiphertext>& pcs)
{
	std::vector<BigNumber> ciphertext_bns;
	for (size_t i = 0; i < pcs.size(); i++)
		ciphertext_bns.push_back(pcs[i].ciphertext.getElement(i));
	return PackedCiphertext(pcs.size(), ipcl::CipherText(pubKey, ciphertext_bns));
}

#else
EncryptionScheme::EncryptionScheme() {}

EncryptionScheme::EncryptionScheme(ipcl::PublicKey _pubKey, ipcl::PrivateKey _privKey,
	size_t _slot_size, size_t _slot_buffer_u32size, size_t _n_slots, size_t precision_bits) :
	pubKey(_pubKey), privKey(_privKey), converter(Converter(precision_bits, _slot_size, _slot_buffer_u32size, _n_slots))
{
	modulus = *_pubKey.getN();
}

PackedCiphertext EncryptionScheme::encrypt(BigNumber bn)
{
	return converter.pack({ bn });
}

PackedCiphertext EncryptionScheme::encrypt(float fVal)
{
	return converter.pack(std::vector<float>{ fVal });
}

PackedCiphertext EncryptionScheme::encrypt(const std::vector<float>& fVals)
{
	return encrypt(converter.pack(fVals));
}

PackedCiphertext EncryptionScheme::encrypt(const std::vector<BigNumber>& bignums)
{
	return encrypt(converter.pack(bignums));
}

PackedCiphertext EncryptionScheme::encrypt(const PackedBigNumber& packed_bn)
{
	return packed_bn;
}

PackedBigNumber EncryptionScheme::decrypt(const PackedCiphertext& cVal)
{
	return cVal;
}

std::vector<BigNumber> EncryptionScheme::decrypt_to_bignumbers(const PackedCiphertext& cVal, size_t multiplication_level)
{
	PackedBigNumber plaintext = converter.reduce_multiplication_level(decrypt(cVal), multiplication_level);
	return converter.to_bignums(plaintext);
}

std::vector<float> EncryptionScheme::decrypt_to_floats(const PackedCiphertext& cVal, size_t multiplication_level)
{
	PackedBigNumber plaintext = converter.reduce_multiplication_level(decrypt(cVal), multiplication_level);
	return converter.to_floats(plaintext);
}

PackedBigNumber EncryptionScheme::add_pp(const PackedBigNumber& p0, const PackedBigNumber& p1)
{
	if (p0.n_elements != p1.n_elements) throw ArithmeticException("Number of elements not match.");
	return (PackedBigNumber(p0) + PackedBigNumber(p1)) % *pubKey.getN();
}

PackedCiphertext EncryptionScheme::add_cc(const PackedCiphertext& c0, const PackedCiphertext& c1)
{
	return add_pp(c0, c1);
}

PackedCiphertext phez::EncryptionScheme::mul_cp(const PackedCiphertext& ct, float pt)
{
	return (PackedBigNumber(ct) * converter.float_to_bignum(pt)) % *pubKey.getN();
}

PackedCiphertext phez::EncryptionScheme::mul_cp(const PackedCiphertext& ct, const BigNumber& pt)
{
	return PackedBigNumber(ct) * ipcl::PlainText(pt);
}

PackedCiphertext phez::EncryptionScheme::mul_cp(const PackedCiphertext& ct, const std::vector<float>& pt)
{
	PackedBigNumber packed_bn = converter.pack(pt);
	return mul_cp(ct, packed_bn);
}

PackedCiphertext phez::EncryptionScheme::mul_cp(const PackedCiphertext& ct, const PackedBigNumber& packed_bn)
{
	if (ct.n_elements != packed_bn.n_elements && ct.n_elements != 1 && packed_bn.n_elements != 1) throw ArithmeticException("Number of elements not match");
	if (ct.n_elements == 1 && packed_bn.n_elements != 1)
		return PackedBigNumber(packed_bn) * ct.packed_bns[0];
	else
		return PackedBigNumber(ct) * packed_bn.packed_bns[0];
}


size_t phez::EncryptionScheme::get_ciphertext_nbytes(const PackedCiphertext& ct)
{
	return ct.n_elements * pubKey.getBits() / 8;
}


PackedBigNumber phez::EncryptionScheme::negate_on_ring(const PackedBigNumber& packed_bn)
{
	std::vector<BigNumber> negated_bignums;
	for (size_t i = 0; i < packed_bn.packed_bns.size(); i++)
		negated_bignums.push_back((modulus - packed_bn.packed_bns[i]) % modulus);
	return PackedBigNumber(packed_bn.n_elements, negated_bignums);
}

BigNumber phez::EncryptionScheme::random_on_ring()
{
	return ipcl::getRandomBN(pubKey.getBits() + 10) % modulus;
}

PackedBigNumber phez::EncryptionScheme::random_on_ring(const PackedCiphertext& ref)
{
	std::vector<BigNumber> random_bns;
	for (size_t i = 0; i < ref.packed_bns.size(); i++)
		random_bns.push_back(random_on_ring());
	return phez::PackedBigNumber(ref.n_elements, random_bns);
}

std::vector<PackedCiphertext> phez::EncryptionScheme::shatter(const PackedCiphertext& pc)
{
	if (converter.n_slots != 1) throw ArithmeticException("shatter: converter.n_slots must be 1");
	std::vector<PackedCiphertext> pcs;
	for (size_t i = 0; i < pc.n_elements; i++)
		pcs.push_back(PackedBigNumber(1, { pc.packed_bns[i] }));
	return pcs;
}

PackedCiphertext phez::EncryptionScheme::merge(const std::vector<PackedCiphertext>& pcs)
{
	std::vector<BigNumber> ciphertext_bns;
	for (size_t i = 0; i < pcs.size(); i++)
		ciphertext_bns.push_back(pcs[i].packed_bns[i]);
	return PackedBigNumber(pcs.size(), ciphertext_bns);
}


#endif