#include <ipcl/ipcl.hpp>
#include "../../paillier_arithmetics/convert/convert.h"
#include "../../paillier_arithmetics/arithmetics.h"
#include "../../utils/random.h"
#include "../../utils/time_counter.hpp"


// void mat_mul_vec(size_t dim1, size_t dim2)
// {
// 	std::vector<std::vector<float>> matrix_columns(dim2);
// 	std::vector<float> vec;
// 	for (size_t i = 0; i < dim2; i++)
// 		for (size_t j = 0; j < dim1; j++)
// 			matrix_columns[i].push_back(zutil::rand.uniform(-1.0f, 1.0f));
// 	for (size_t i = 0; i < dim2; i++)
// 		vec.push_back(zutil::rand.uniform(-1.0f, 1.0f));

// 	std::vector<float> desired_result;
// 	for (size_t i = 0; i < dim1; i++)
// 	{
// 		float elem = 0;
// 		for (size_t j = 0; j < dim2; j++)
// 			elem += matrix_columns[j][i] * vec[i];
// 		desired_result.push_back(elem);
// 	}

// 	ipcl::KeyPair keyPair = ipcl::generateKeypair(2048);
// 	phez::EncryptionScheme scheme(keyPair.pub_key, keyPair.priv_key, 3, 12, 5, 25);

// 	zutil::time_counter.start();
// 	std::vector<phez::PackedCiphertext> packed_mat_cols(dim2);
// 	for (size_t i = 0; i < dim2; i++)
// 		packed_mat_cols[i] = 


// }