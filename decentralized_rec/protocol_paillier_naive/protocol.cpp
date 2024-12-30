#include <ipcl/ipcl.hpp>
#include <omp.h>

#include "protocol.h"
#include "../../utils/random.h"
#include "../../utils/ios.hpp"
#include "../../utils/math.hpp"
#include "../../utils/time_counter.hpp"
#include "../../paillier_arithmetics/arithmetics.h"


const int debug_level = 0;

drec::paillier0::TrainProtocol::TrainProtocol(drec::Recommender* _recommender, std::vector<drec::Client>* _clients, size_t key_size,
	float _learning_rate, float _reg_u, float _reg_i, float _reg_s, 
	size_t _n_items_per_iter, size_t _n_friends_per_iter, size_t _n_aggregation, std::string optimizer_type) : 
	recommender(_recommender), clients(_clients),
	learning_rate(_learning_rate), reg_u(_reg_u), reg_i(_reg_i), reg_s(_reg_s), 
	n_items_per_iter(_n_items_per_iter), n_friends_per_iter(_n_friends_per_iter), n_aggregation(_n_aggregation)
{
	ipcl::KeyPair keyPair = ipcl::generateKeypair(key_size);
	scheme = phez::EncryptionScheme(keyPair.pub_key, keyPair.priv_key, 3, 12, 1, 25);  // One ciphertext only contain one number

	item_update_buffer.resize(recommender->item_embeddings.size());

	raw_protocol = new drec::raw::TrainProtocol(_recommender, _clients, _learning_rate, _reg_u, _reg_i, _reg_s, _n_items_per_iter, _n_friends_per_iter, _n_aggregation, optimizer_type);

	n_rounds = 0;
	client_inbound_sizes.resize(_clients->size(), 0);
	client_comm_rounds.resize(_clients->size(), 0);
	client_outbound_sizes.resize(_clients->size(), 0);


	// ===== Debug_level 2 ======
	if (debug_level >= 2)
	{
		std::cout << "#Items: " << recommender->item_embeddings.size() << std::endl;
		std::cout << "#Users: " << clients->size() << std::endl;
		std::cout << "Learning rate: " << learning_rate << std::endl;
		std::cout << "Regularization (user, item, social): " << reg_u << "," << reg_i << "," << reg_s << std::endl;
	}

	size_t n_items = recommender->item_embeddings.size();
	size_t n_users = clients->size();
	size_t embedding_dim = recommender->item_embeddings[0].size();

	// Initialize the caches
	cached_vs_ct.resize(n_items);
	cached_social_us_ct.resize(n_users);
	cached_social_neg_us_ct.resize(n_users);

	// Initialize the optimizer
	if (optimizer_type == "momentum") optimizer = new drec::MomentumRecOptimizer(n_items, n_users, embedding_dim, learning_rate, 0.9f);
	else optimizer = new drec::RecOptimizer(n_items, n_users, embedding_dim, learning_rate);
	for (size_t i = 0; i < n_users; i++) optimizer->user_embeddings[i] = &clients->at(i).embedding;
	for (size_t i = 0; i < n_items; i++) optimizer->item_embeddings[i] = &recommender->item_embeddings[i];

}

void drec::paillier0::TrainProtocol::run_one_iteration(size_t client_id, std::vector<size_t>& selected_item_indices, std::vector<size_t>& selected_friend_indices)
{
	drec::TimeCounter tc;
	float ciphertext_size = scheme.pubKey.getBits() / 8;  // The byte-length of one ciphertext
	// ----------------------------------------------------  Client  ------------------------------------------------------------------


	// ===== Debug level 1
	if (debug_level >= 1) std::cout << "================ Iteration begins, #items: " << selected_item_indices.size() << std::endl;


	Client& client = clients->at(client_id);
	// Client selects item indices, sends to the server
	if (client.item_ratings.size() == 0) return;

	// ===== Open the debug mode
	if (debug_level >= 2) raw_protocol->iter_compute_grads(client_id, selected_item_indices, selected_friend_indices);


	std::vector<size_t> selected_item_ids(selected_item_indices.size());

	for (size_t i = 0; i < selected_item_indices.size(); i++) selected_item_ids[i] = client.item_ratings[selected_item_indices[i]].first;

	// ==== Debug_level 2 ====
	if (debug_level >= 2) print_vector("Selected item indices", selected_item_ids);

	client_outbound_sizes[client_id] += selected_item_ids.size() * 4; // 4 bytes for each indices is enough.
	client_comm_rounds[client_id] += 1;


	// ----------------------------------------------------  Server  -------------------------------------------------------------------
	size_t embedding_dim = client.embedding.size();
	size_t n_items = selected_item_ids.size();

	// The recommender gets the indices and encrypt them, in item order.
	// Format v11 v12 v13
	//		  v21 v22 v23 ...
	std::vector<std::vector<float>> vs_pt;
	vs_pt.resize(n_items);
	for (size_t i = 0; i < n_items; i++)
	{
		vs_pt[i].resize(embedding_dim);
		for (size_t j = 0; j < embedding_dim; j++) 
			vs_pt[i][j] = recommender->item_embeddings[selected_item_ids[i]][j];
	}

	// Basic protocol to compute [UV - R]
#pragma region BasicUVsubR


	// ======================================================================================================
	// Recommender system encrypts the embeddings
	std::vector<std::vector<phez::PackedCiphertext>> vs_ct(n_items);

	// ===== Debug time =====
	if (debug_level >= 1) tc.start();
#pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
	{
		size_t item_id = selected_item_ids[i];
		// Check whether the encrypted item embedding is already in the cache
		if (cached_vs_ct[item_id].empty())
		{
			cached_vs_ct[item_id].resize(embedding_dim);
			for (size_t j = 0; j < embedding_dim; j++)
			{
				cached_vs_ct[item_id][j] = scheme.encrypt(vs_pt[i][j]);
			}
		}
		vs_ct[i] = cached_vs_ct[item_id];
	}
	if (debug_level >= 1) std::cout << "(omp) encrypt vs_ct: " << tc.tick() / 1000 << "s" << std::endl;



	// Recommender system sends the embeddings to client
	client_inbound_sizes[client_id] += n_items * embedding_dim * ciphertext_size;
	client_comm_rounds[client_id] += 1;


	// -------------------------------------------------------  Client  ----------------------------------------------
	// 
	// Client computes embedding element-wise product (U^TV),
	// Format u1v11 u1v21 u1v31
	//        u2v12 u2v22 u2v32...
	std::vector<std::vector<phez::PackedCiphertext>> uv_elems_ct(n_items);
	// ===== Debug time =====
	if (debug_level >= 1) tc.start();
	// #pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
	{
		uv_elems_ct[i].resize(embedding_dim);
		for (size_t j = 0; j < embedding_dim; j++)
		{
			uv_elems_ct[i][j] = scheme.mul_cp(vs_ct[i][j], client.embedding[j]);
		}
	}
	if (debug_level >= 1) std::cout << "mul_cp uv_elems_ct: " << tc.tick() / 1000 << "s" << std::endl;


	// Sum along the vertical axis to get [UV1, UV2, ...]
	std::vector<phez::PackedCiphertext> uv_ct(n_items);
	for (size_t i = 0; i < n_items; i++)
		uv_ct[i] = uv_elems_ct[i][0];
	for (size_t i = 0; i < n_items; i++)
	{
		for (size_t j = 1; j < embedding_dim; j++)
			uv_ct[i] = scheme.add_cc(uv_ct[i], uv_elems_ct[i][j]);
	}

	// Create ratings
	std::vector<float> item_ratings(n_items);
	for (size_t i = 0; i < n_items; i++) item_ratings[i] = client.item_ratings[selected_item_indices[i]].second;


	// ======= Debug_level 2 =====
	if (debug_level >= 2)
	{
		std::vector<float> debug__uv_pt(n_items);
		for (size_t i = 0; i < n_items; i++)
			debug__uv_pt[i] = scheme.decrypt_to_floats(uv_ct[i], 1)[0];
		print_vector("Item ratings", item_ratings);
		print_vector("UV product", debug__uv_pt);
		print_vector("UV product real", raw_protocol->idata.uvs);
		check_equal(debug__uv_pt, raw_protocol->idata.uvs);
	}

	// Client masks embedding product, big-length is 2 * slot size * 32 since it is applied on 1 multiplication level numbers

	std::vector<BigNumber> random_bns(n_items);
	std::vector<BigNumber> r_mask_bns = scheme.converter.random_mask(n_items, scheme.converter.slot_size * 32 * 2, true);
	std::vector<BigNumber> masked_neg_r_bns(n_items);
	std::vector<BigNumber> item_rating_l1_bns(n_items);

	for (size_t i = 0; i < n_items; i++) item_rating_l1_bns[i] = scheme.converter.reduce_multiplication_level_raw_bn(scheme.converter.float_to_bignum(item_ratings[i], false), -1);


	for (size_t i = 0; i < n_items; i++) masked_neg_r_bns[i] = r_mask_bns[i] - item_rating_l1_bns[i];

	std::vector<phez::PackedCiphertext> masked_neg_r_ct(n_items);
#pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
		masked_neg_r_ct[i] = scheme.encrypt(masked_neg_r_bns[i]);

	std::vector<phez::PackedCiphertext> masked_uv_sub_r_ct(n_items);
#pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
		masked_uv_sub_r_ct[i] = scheme.add_cc(uv_ct[i], masked_neg_r_ct[i]);
	// Get MASK + UV - R
	// Format(masked) [UV1 - R1, UV2 - R2, ...]


	// Client send the masked product to the recommender

	client_outbound_sizes[client_id] += n_items * ciphertext_size;
	client_comm_rounds[client_id] += 1;

	// -------------------------------------------  Server  ------------------------------------------------

	// The server performs further computation.
	// First, the server get all masked UV - R.
	// Multiplication level is changed to 0
	std::vector<BigNumber> masked_uv_sub_r_pt(n_items);
	for (size_t i = 0; i < n_items; i++)
		masked_uv_sub_r_pt[i] = scheme.converter.unpack_to_bignumbers(scheme.decrypt(masked_uv_sub_r_ct[i]))[0];

#pragma endregion 
	//BasicUVsubR
#pragma region ComputeUGrad
	// =====================================================================
	// This is for compute user gradients

	// Compute sum[masked (UV - R) * V, dim=0] * V' (Weighted V)
	// Format
	// (UV1 - R1)V1 + (UV2 - R2)V2 + ...
	// Multiplication level = 2 (already reaches max, as we assume slot_buffer = S * 2, hence we needs to reduce it)


	// -------------------------------------------  Server  --------------------------------------------------------
	std::vector<BigNumber> masked_gu_v_pt(embedding_dim, BigNumber::Zero());
	std::vector<phez::PackedCiphertext> masked_uv_sub_r_elems_ct(n_items);

	// ====== Debug time =====
	if (debug_level >= 1) tc.start();
#pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
	{
		masked_uv_sub_r_elems_ct[i] = scheme.encrypt(masked_uv_sub_r_pt[i]);
	}
	if (debug_level >= 1) std::cout << "(omp) encrypt masked_uv_sub_r_elems_ct: " << tc.tick() / 1000 << "s" << std::endl;

	// ===== Debug time =====
	if (debug_level >= 1) tc.start();
	for (size_t i = 0; i < n_items; i++)
	{
		for (size_t j = 0; j < embedding_dim; j++)
			masked_gu_v_pt[j] += masked_uv_sub_r_pt[i] * scheme.converter.float_to_bignum(vs_pt[i][j]);
	}

	std::vector<phez::PackedCiphertext> masked_gu_v_ct(embedding_dim);
	for (size_t i = 0; i < embedding_dim; i++)
		masked_gu_v_ct[i] = scheme.encrypt(masked_gu_v_pt[i]);

	// Send the ciphertext of weighted_v (can be interpreted as Gu_1, gradient w.r.t. U, first portion (computed by the item embedding)
	client_inbound_sizes[client_id] += embedding_dim * ciphertext_size;
	// Send the masked_uv_sub_r_elems_ct to clients.
	client_inbound_sizes[client_id] += n_items * ciphertext_size;
	client_comm_rounds[client_id] += 1;

	// -------------------------------------------- Client ---------------------------------------------------------------
	// The client have to unmask this ciphertext
	// Hence, it have to subtract -Mask * V1 (this shall performed by plus 2^(32 * 3) - mask * V?
	std::vector<BigNumber> neg_r_mask_bns;
	for (size_t i = 0; i < r_mask_bns.size(); i++) neg_r_mask_bns.push_back(scheme.converter.negate(r_mask_bns[i], 2 * 32 * scheme.converter.slot_size));

	std::vector<phez::PackedCiphertext> neg_mask_dot_v_ct(embedding_dim);
	for (size_t i = 0; i < embedding_dim; i++)
		neg_mask_dot_v_ct[i] = scheme.mul_cp(vs_ct[0][i], neg_r_mask_bns[0]);


	std::vector<std::vector<phez::PackedCiphertext>> neg_r_mask_dot_vs_ct(n_items);
	// ====== Debug time =====
	if (debug_level >= 1) tc.start();
	// #pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
	{
		neg_r_mask_dot_vs_ct[i].resize(embedding_dim);
		for (size_t j = 0; j < embedding_dim; j++)
			neg_r_mask_dot_vs_ct[i][j] = scheme.mul_cp(vs_ct[i][j], neg_r_mask_bns[i]);
	}
	if (debug_level >= 1) std::cout << "mul_cp neg_r_mask_dot_vs_ct: " << tc.tick() / 1000 << "s" << std::endl;

#pragma omp parallel for
	for (size_t j = 0; j < embedding_dim; j++)
		for (size_t i = 1; i < n_items; i++)
			neg_mask_dot_v_ct[j] = scheme.add_cc(neg_mask_dot_v_ct[j], neg_r_mask_dot_vs_ct[i][j]);

	// Client obtains (UV - R) * V by subtracting the mask.
	// This yields Gu_1, the gradient brought by the item embeddings (multiplication level = 2)
	for (size_t i = 0; i < embedding_dim; i++)
		masked_gu_v_ct[i] = scheme.add_cc(masked_gu_v_ct[i], neg_mask_dot_v_ct[i]);


	// ================ Debug_level 2 ==========================
	if (debug_level >= 2)
	{
		std::vector<float> debug__gu_v(embedding_dim);
		for (size_t i = 0; i < embedding_dim; i++)
			debug__gu_v[i] = scheme.decrypt_to_floats(masked_gu_v_ct[i], 2)[0];
		print_vector("Gu from V", debug__gu_v);
		print_vector("Gu from V real", raw_protocol->idata.gu_v);
		check_equal(debug__gu_v, raw_protocol->idata.gu_v);

	}

	// Add friend's embeddings (weighted) to Gu_1, then obtains the complete user gradient.
	// (UV - R) * V + reg_s * sum (U' - U)
	size_t n_friends = selected_friend_indices.size();
	if (debug_level >= 1) std::cout << "#Friends: " << n_friends<< std::endl;

	std::vector<phez::PackedCiphertext> gs_friend_ct(n_friends);

	if (cached_social_us_ct[client_id].empty())
	{
		std::vector<float> self_emb(embedding_dim);
		for (size_t j = 0; j < embedding_dim; j++) self_emb[j] = reg_s * client.embedding[j] * n_friends;
		std::vector<BigNumber> self_emb_pt(embedding_dim);
		for (size_t j = 0; j < embedding_dim; j++)
		{
			self_emb_pt[j] = scheme.converter.float_to_bignum(self_emb[j]);
			self_emb_pt[j] = scheme.converter.reduce_multiplication_level_raw_bn(self_emb_pt[j], -1);
		}
		for (size_t j = 0; j < embedding_dim; j++)
		{
			cached_social_us_ct[client_id].resize(embedding_dim);
			cached_social_us_ct[client_id][j] = scheme.encrypt(self_emb_pt[j]);
		}
	}

	// ===== Debug time =====
	if (debug_level >= 1) tc.start();
// #pragma omp parallel for
	for (size_t i = 0; i < n_friends; i++)
	{
		std::vector<float> neg_friend_emb(embedding_dim);
		size_t friend_id = client.friend_trusts[selected_friend_indices[i]].first;
		// float trust_score = client.friend_trusts[i].second;
		for (size_t j = 0; j < embedding_dim; j++)
		{
			neg_friend_emb[j] = - clients->at(friend_id).embedding[j] * reg_s;
		}
		
		std::vector<BigNumber> neg_friend_emb_pt(embedding_dim);
		for (size_t j = 0; j < embedding_dim; j++)
		{
			neg_friend_emb_pt[j] = scheme.converter.float_to_bignum(neg_friend_emb[j]);
			neg_friend_emb_pt[j] = scheme.converter.reduce_multiplication_level_raw_bn(neg_friend_emb_pt[j], -1);
		}
		// First, check for the caches
		if (cached_social_neg_us_ct[friend_id].empty())
		{
			cached_social_neg_us_ct[friend_id].resize(embedding_dim);
#pragma omp parallel for
			for (size_t j = 0; j < embedding_dim; j++)
				cached_social_neg_us_ct[friend_id][j] = scheme.encrypt(neg_friend_emb_pt[j]);
		}
	}
	if (debug_level >= 1) std::cout << "encrypt gs_friend_ct: " << tc.tick() / 1000 << "s" << std::endl;


	// ===== Debug time =====
	if (debug_level >= 1) tc.start();
	std::vector<phez::PackedCiphertext> sum_gs_friend_ct = cached_social_us_ct[client_id];

	for (size_t i = 0; i < n_friends; i++)
	{
		size_t friend_id = client.friend_trusts[selected_friend_indices[i]].first;
		for (size_t j = 0; j < embedding_dim; j++)
			sum_gs_friend_ct[j] = scheme.add_cc(sum_gs_friend_ct[j], cached_social_neg_us_ct[client.friend_trusts[selected_friend_indices[i]].first][j]);
	}

	for (size_t i = 0; i < embedding_dim; i++)
		sum_gs_friend_ct[i] = scheme.mul_cp(sum_gs_friend_ct[i], float(2) / n_friends);
	if (debug_level >= 1) std::cout << "add_cc gs_friend_ct (non-omp): " << tc.tick() / 1000 << "s" << std::endl;

	// ===== Debug level 2 =====
	if (debug_level >= 2)
	{
		// This is for debug: to check that whether weighted_v_ct is correct
		std::vector<float> debug__g_friends(embedding_dim);
		for (size_t i = 0; i < embedding_dim; i++) 
			debug__g_friends[i] = scheme.decrypt_to_floats(sum_gs_friend_ct[i], 2)[0];
		print_vector("Gu from friend embeddings", debug__g_friends);
		print_vector("Gu from friend embeddings real", raw_protocol->idata.gu_s);
		check_equal(debug__g_friends, raw_protocol->idata.gu_s);
	}


	// Notice that, now we are computing the opposite number of the gradient!!!
	for (size_t i = 0; i < embedding_dim; i++)
		masked_gu_v_ct[i] = scheme.add_cc(masked_gu_v_ct[i], sum_gs_friend_ct[i]);

	// Add mask to gradients. Consider that the multiplication level = 2, and the modular is 2^64 scale, the mask shall be 2^96 scale
	std::vector<BigNumber> g_mask_bns = scheme.converter.random_mask(embedding_dim, scheme.converter.slot_size * 32 * 3);
	
	//std::vector<phez::PackedCiphertext> masked_gu_v_ct(embedding_dim);
	for (size_t i = 0; i < embedding_dim; i++)
		masked_gu_v_ct[i] = scheme.add_cc(masked_gu_v_ct[i], scheme.encrypt(g_mask_bns[i]));

	// User send the masked Gu to the server
	client_outbound_sizes[client_id] += embedding_dim * ciphertext_size;
	client_comm_rounds[client_id] += 1;

	// ------------------------------------------------  Server  -----------------------------------------------------------------
	// Server decrypts the masked gradient, then send it back to the client
	
	std::vector<BigNumber> masked_gu_pt(embedding_dim);
	for (size_t i = 0; i < embedding_dim; i++)
		masked_gu_pt[i] = scheme.decrypt_to_bignumbers(masked_gu_v_ct[i])[0];


	client_inbound_sizes[client_id] += embedding_dim * ciphertext_size;
	client_comm_rounds[client_id] += 1;

	// ------------------------------------------------  Client  -------------------------------------------------------------------


	std::vector<BigNumber> gu_pt(embedding_dim);
	for (size_t i = 0; i < embedding_dim; i++)
		gu_pt[i] = masked_gu_pt[i] - g_mask_bns[i];
	std::vector<float> gu_float(embedding_dim);
	for (size_t i = 0; i < embedding_dim; i++)
		gu_float[i] = scheme.converter.bignum_to_float(gu_pt[i], 2);

	// ====== Debug_level 2 ======
	if (debug_level >= 2)
	{
		print_vector("Gu complete", gu_float);
		print_vector("Gu complete real", raw_protocol->idata.gu);
		check_equal(gu_float, raw_protocol->idata.gu);
	}



#pragma endregion
	//	ComputeUGrad


#pragma region ComputeIGrad

	// --------------------------------------------------- Client ---------------------------------------------------------------------

	// Notice that we already set the item_ratings_bn to multiplication level 1.
	std::vector<phez::PackedCiphertext> uv_sub_r_elems_ct(n_items);
	std::vector<phez::PackedCiphertext> neg_r_mask_elems_ct(n_items);

	// ===== Debug time
	if (debug_level >= 1) tc.start();
#pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
	{
		neg_r_mask_elems_ct[i] = scheme.encrypt(neg_r_mask_bns[i]);
		uv_sub_r_elems_ct[i] = scheme.add_cc(masked_uv_sub_r_elems_ct[i], neg_r_mask_elems_ct[i]);
	}
	if (debug_level >= 1) std::cout << "add_cc uv_sub_r_elemes_ct: " << tc.tick() / 1000 << "s" << std::endl;

	std::vector<std::vector<phez::PackedCiphertext>> i_grad_elems_ct(n_items);

	// ===== Debug time
	if (debug_level >= 1) tc.start();
#pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
	{
		i_grad_elems_ct[i].resize(embedding_dim);
		for (size_t j = 0; j < embedding_dim; j++)
			i_grad_elems_ct[i][j] = scheme.mul_cp(uv_sub_r_elems_ct[i], client.embedding[j]);
	}

	if (debug_level >= 1) std::cout << "mul_cp i_grad_elems_ct: " << tc.tick() / 1000 << "s" << std::endl;
	// [[(UV1 - R)u1, (UV1 - R)u2, ...],
	//	[(UV2 - R)u1, (UV2 - R)u2, ...],
	//  ...]  Each row is a packed ciphertext



	// Client send the masked_i_grads and the sum of mask to server (hence the server only knows the aggregated i grad
	client_outbound_sizes[client_id] += embedding_dim * n_items * ciphertext_size;
	client_comm_rounds[client_id] += 1;

	// -------------------------------------------  Server/Perhaps a intermediate layer  ----------------------------------------------------
	// For privacy concerns, late update shall be performed
	// ===== Debug time
	if (debug_level >= 1) tc.start();
// #pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
	{
		size_t item_id = selected_item_ids[i];
		item_update_buffer[item_id].push_back(i_grad_elems_ct[i]);
		if (item_update_buffer[item_id].size() == n_aggregation)
		{
			std::vector<phez::PackedCiphertext> aggregated_i_grad_ct = item_update_buffer[item_id][0];
			for (size_t j = 1; j < n_aggregation; j++)
			{
				for (size_t k = 0; k < embedding_dim; k++)
					aggregated_i_grad_ct[k] = scheme.add_cc(aggregated_i_grad_ct[k], item_update_buffer[item_id][j][k]);
			}
			std::vector<float> aggregated_i_grad_pt(embedding_dim);
			for (size_t j = 0; j < embedding_dim; j++)
				aggregated_i_grad_pt[j] = scheme.decrypt_to_floats(aggregated_i_grad_ct[j], 2)[0];
			// ========== Debug_level 2 ========================
			if (debug_level >= 2) print_vector("Aggregated V update", aggregated_i_grad_pt);

			std::vector<float> item_update(embedding_dim, 0);
			// *********************************************************
			// Update the item embedding!
			// *********************************************************
			for (size_t j = 0; j < embedding_dim; j++)
			{
			    if (j!=0) item_update[j] += -2 * reg_i * recommender->item_embeddings[item_id][j];
				item_update[j] += - 2 * aggregated_i_grad_pt[j];
			}
			optimizer->item_step(item_id, item_update);
			item_update_buffer[item_id].clear();
			cached_vs_ct[item_id].clear(); // Set the cache invalid
		}
	}
	if (debug_level >= 1) std::cout << "decrypt item updates: " << tc.tick() / 1000 << "s" << std::endl;


#pragma endregion
	// ComputeIGrad
	std::vector<float> client_update(embedding_dim, 0);
	// ********************************************************
	// Update the client embedding!
	// ********************************************************
	for (size_t i = 0; i < embedding_dim; i++)
	{
		if (i!=1) client_update[i] += -2 * reg_u * client.embedding[i];
		client_update[i] += - 2 * gu_float[i];
	}
	optimizer->user_step(client_id, client_update);
	// Set the cache invalid
	cached_social_us_ct[client_id].resize(0);
	cached_social_neg_us_ct[client_id].resize(0);


	// ====== Debug_level 2
	if (debug_level >= 2)
	{
		std::vector<float> debug__new_uvs;
		for (size_t i = 0; i < n_items; i++)
		{
			float new_uv = 0;
			for (size_t j = 0; j < embedding_dim; j++)
			{
				new_uv += client.embedding[j] * vs_pt[i][j];
			}
			debug__new_uvs.push_back(new_uv);
		}
		print_vector("New UV", debug__new_uvs);
	}
}


void drec::paillier0::TrainProtocol::run_one_client(size_t client_id)
{
	Client& client = clients->at(client_id);
	if (client.item_ratings.size() == 0) return;
	size_t embedding_dim = client.embedding.size();

	std::vector<std::vector<size_t>> splitted_item_indices = drec::split_vector(client.item_ratings.size(), 10);
	for (size_t i = 0; i < splitted_item_indices.size(); i++)
	{
		std::vector<size_t> friend_indices = drec::rand.choice(client.friend_trusts.size(), n_friends_per_iter);
		run_one_iteration(client_id, splitted_item_indices[i], friend_indices);
	}
}

void drec::paillier0::TrainProtocol::run_one_epoch()
{
	size_t n_ratings = 0;
	for (size_t i = 0; i < clients->size(); i++)
	{
		std::cout << "Run client: " << i << ", processed ratings: " << n_ratings << std::endl;
		std::cout << "Time elapsed: " << drec::time_counter.tick()/1000.0 << std::endl;
		run_one_client(i);
		n_ratings += clients->at(i).item_ratings.size();
	}
}
