// #include <ipcl/ipcl.hpp>
#include "/home/qjm/code/PaillierMatrix/ipcl/include/ipcl/ipcl.hpp"
#include <omp.h>

#include "protocol.h"

#include "../../paillier_arithmetics/arithmetics.h"

#include "../../utils/random.h"
#include "../../utils/ios.hpp"
#include "../../utils/math.hpp"
#include "../../utils/time_counter.hpp"
#include "../../utils/bignum.h"


// const int debug_level = 0;
const int debug_level = -1; // inference

drec::paillier1::TrainProtocol::TrainProtocol(drec::Recommender* _recommender, std::vector<drec::Client>* _clients, size_t key_size,
	float _learning_rate, float _reg_u, float _reg_i, float _reg_s, 
	size_t _n_items_per_iter, size_t _n_friends_per_iter, size_t _n_aggregation, std::string optimizer_type) : 
	recommender(_recommender), clients(_clients),
	embedding_dim(_recommender->item_embeddings[0].size()),
	learning_rate(_learning_rate), reg_u(_reg_u), reg_i(_reg_i), reg_s(_reg_s), 
	n_items_per_iter(_n_items_per_iter), n_friends_per_iter(_n_friends_per_iter), n_aggregation(_n_aggregation)
{
	ipcl::KeyPair keyPair = ipcl::generateKeypair(key_size);

	// scheme = phez::EncryptionScheme(keyPair.pub_key, keyPair.priv_key, 80, 8, 8, 23);
	scheme = phez::EncryptionScheme(keyPair.pub_key, keyPair.priv_key, 80, 8, 1, 23);
	// scheme = phez::EncryptionScheme(keyPair.pub_key, keyPair.priv_key, 55, 3, 21, 23); // inference

	item_update_buffer.resize(recommender->item_embeddings.size());

	raw_protocol = new drec::raw::TrainProtocol(_recommender, _clients, _learning_rate, _reg_u, _reg_i, _reg_s, _n_items_per_iter, _n_friends_per_iter, _n_aggregation, optimizer_type);

	n_rounds = 0;
	client_inbound_sizes.resize(_clients->size(), 0);
	client_comm_rounds.resize(_clients->size(), 0);
	client_outbound_sizes.resize(_clients->size(), 0);


	// ===== Debug_level 2 ======
	if (debug_level >= 2)
	{
		std::cout << "Paillier key generated. Key bits: " << keyPair.pub_key.getBits() << ", Pub key N: " << *keyPair.pub_key.getN() << std::endl;
		std::cout << "N bits=" << zutil::get_bit_length(*keyPair.pub_key.getN()) << std::endl;
		std::cout << "#Items: " << recommender->item_embeddings.size() << std::endl;
		std::cout << "#Users: " << clients->size() << std::endl;
		std::cout << "Learning rate: " << learning_rate << std::endl;
		std::cout << "Regularization (user, item, social): " << reg_u << "," << reg_i << "," << reg_s << std::endl;
	}

	size_t n_items = recommender->item_embeddings.size();
	size_t n_users = clients->size();


	// Initialize the caches
	cached_vs_ct.resize(n_items);
	cached_social_neg_us_ct.resize(n_users);

	// Initialize the optimizer
	if (optimizer_type == "momentum") optimizer = new drec::MomentumRecOptimizer(n_items, n_users, embedding_dim, learning_rate, 0.9f);
	else optimizer = new drec::RecOptimizer(n_items, n_users, embedding_dim, learning_rate);
	optimizer->initialize(*recommender, *clients);
}


void drec::paillier1::TrainProtocol::iter_fetch_data(size_t client_id, std::vector<size_t>& selected_item_indices, std::vector<size_t>& selected_friend_indices)
{
	idata.client_id = client_id;

	Client& client = clients->at(client_id);
	size_t n_items = selected_item_indices.size();

	// Client
	// ======================================================================
	// Client gets item_ids, rating, friend_ids, friend_trusts
	// and sends item ids to recommender
	idata.selected_item_ids.clear();
	idata.rs.clear();
	for (size_t i = 0; i < selected_item_indices.size(); i++)
	{
		idata.selected_item_ids.push_back(client.item_ratings[selected_item_indices[i]].first);
		idata.rs.push_back(client.item_ratings[selected_item_indices[i]].second);
	}
	if (debug_level >= 2) zutil::print_vector(idata.selected_item_ids, "Selected item ids");

	// client_outbound_sizes[client_id] += selected_item_indices.size() * 4; // 4 bytes for each indices is enough.
	// client_comm_rounds[client_id] += 1;

	idata.selected_friend_ids.clear();
	idata.selected_friend_trusts.clear();
	for (size_t i = 0; i < selected_friend_indices.size(); i++)
	{
		idata.selected_friend_ids.push_back(client.friend_trusts[i].first);
		idata.selected_friend_trusts.push_back(client.friend_trusts[i].second);
	}
}


void drec::paillier1::TrainProtocol::iter_compute_uv()
{
	Client& client = clients->at(idata.client_id);
	size_t n_items = idata.selected_item_ids.size();

	zutil::TimeCounter tc = zutil::time_counter;
	// Recommender 
	// ========================================================================================
	// 
	// The recommender gets the indices and encrypt them, in embedding dimension order. (The same dimension of different embeddings are packed together)
	// Format v11 v21 v31
	//		  v12 v22 v32 ...
	std::vector<std::vector<float>> vs_transposed_pt;
	vs_transposed_pt.resize(embedding_dim);
	for (size_t i = 0; i < embedding_dim; i++)
	{
		vs_transposed_pt[i].resize(n_items);
		for (size_t j = 0; j < n_items; j++)
			vs_transposed_pt[i][j] = recommender->item_embeddings[idata.selected_item_ids[j]][i];
	}


	std::vector<phez::PackedCiphertext> vs_transposed_ct(embedding_dim);

	if (debug_level >= 1) tc.start();
	#ifndef PSEUDO_MODE
	#pragma omp parallel for
	#endif
	for (size_t i = 0; i < embedding_dim; i++)
		vs_transposed_ct[i] = scheme.encrypt(vs_transposed_pt[i]);
	if (debug_level >= 1) std::cout << "(omp) encrypt vs_transposed_ct: " << tc.tick() / 1000 << "s" << std::endl;


	client_inbound_sizes[idata.client_id] += embedding_dim * scheme.get_ciphertext_nbytes(vs_transposed_ct[0]); // 4 bytes for each indices is enough.
	client_comm_rounds[idata.client_id] += 1;

	// Client
	// =========================================================
	// Client computes embedding element-wise product (U^TV),
	// Format u1v11 u1v21 u1v31
	//        u2v12 u2v22 u2v32...

	std::vector<phez::PackedCiphertext> uv_elems_ct(embedding_dim);
	if (debug_level >= 1) tc.start();
	#ifndef PSEUDO_MODE
	#pragma omp parallel for
	#endif
	for (size_t i = 0; i < embedding_dim; i++)
		uv_elems_ct[i] = scheme.mul_cp(vs_transposed_ct[i], client.embedding[i]);

	if (debug_level >= 1) std::cout << "(omp) mul_cp uv_elems_ct: " << tc.tick() / 1000 << "s" << std::endl;

	// Sum along the vertical axis to get [UV1, UV2, ...]
	idata.uv_ct = uv_elems_ct[0];
	for (size_t i = 1; i < embedding_dim; i++)
		idata.uv_ct = scheme.add_cc(idata.uv_ct, uv_elems_ct[i]);

	// ======= Check UV =====
	if (debug_level >= 2)
	{
		std::vector<float> debug__uv_pt = scheme.decrypt_to_floats(idata.uv_ct, 1);
		
		zutil::print_vector(debug__uv_pt, "UV product");
		zutil::print_vector(raw_protocol->idata.uvs, "UV product real");
		zutil::check_equal(debug__uv_pt, raw_protocol->idata.uvs);
	}
}


void drec::paillier1::TrainProtocol::iter_compute_gu_v()
{
	zutil::TimeCounter tc = zutil::time_counter;

	size_t n_items = idata.selected_item_ids.size();

	// Client
	// ==========================================================================================
	// Client computes masked UV - R and sends it to server
	phez::PackedBigNumber neg_rs_pt_l1 = scheme.converter.reduce_multiplication_level(scheme.converter.negate(scheme.converter.pack(idata.rs)), -1);
	phez::PackedBigNumber r_mask_pt = scheme.converter.random_mask(n_items, scheme.converter.slot_size * 2, true);
	phez::PackedBigNumber masked_neg_r_pt = scheme.add_pp(neg_rs_pt_l1, r_mask_pt);
	phez::PackedCiphertext masked_uv_sub_r_ct = scheme.add_cc(idata.uv_ct, scheme.encrypt(masked_neg_r_pt));


	client_outbound_sizes[idata.client_id] += scheme.get_ciphertext_nbytes(masked_uv_sub_r_ct);
	client_comm_rounds[idata.client_id] += 1;
	
	// Recommender
	// ===================================
	// The recommender gets the item embeddings, then compute masked (uv - r)v


	std::vector<std::vector<float>> vs_pt;
	vs_pt.resize(n_items);
	for (size_t i = 0; i < n_items; i++)
	{
		vs_pt[i].resize(embedding_dim);
		for (size_t j = 0; j < embedding_dim; j++) vs_pt[i][j] = recommender->item_embeddings[idata.selected_item_ids[i]][j];
	}

	std::vector<BigNumber> masked_uv_sub_r_bns = scheme.converter.to_bignums(scheme.decrypt(masked_uv_sub_r_ct));
	std::vector<BigNumber> masked_gu_v_bns(embedding_dim, BigNumber::Zero());
	for (size_t i = 0; i <masked_uv_sub_r_bns.size(); i++)
		for (size_t j = 0; j < embedding_dim; j++)
			masked_gu_v_bns[j] += masked_uv_sub_r_bns[i] * scheme.converter.float_to_bignum(vs_pt[i][j]);

	phez::PackedBigNumber masked_gu_v_pt = scheme.converter.pack(masked_gu_v_bns);
	phez::PackedCiphertext masked_gu_v_ct = scheme.encrypt(masked_gu_v_pt);

	// Recommender system encrypts the embedding
	std::vector<phez::PackedCiphertext> vs_ct(n_items);
	// ===== Debug time =====
	if (debug_level >= 1) tc.start();
	#ifndef PSEUDO_MODE
	#pragma omp parallel for
	#endif
	for (size_t i = 0; i < n_items; i++)
	{
		size_t item_id = idata.selected_item_ids[i];
		// Check whether the encrypted item embedding is already in the cache
		if (cached_vs_ct[item_id].n_elements == 0) cached_vs_ct[item_id] = scheme.encrypt(vs_pt[i]);
		vs_ct[i] = cached_vs_ct[item_id];
	}
	if (debug_level >= 1) std::cout << "(omp) encrypt vs_ct: " << tc.tick() / 1000 << "s" << std::endl;

	client_inbound_sizes[idata.client_id] += scheme.get_ciphertext_nbytes(masked_gu_v_ct); // 4 bytes for each indices is enough.
	client_inbound_sizes[idata.client_id] += n_items * scheme.get_ciphertext_nbytes(vs_ct[0]); // 4 bytes for each indices is enough.
	client_comm_rounds[idata.client_id] += 1;

	// Extra work: send masked_uv_sub_r_elems_ct to client for further use
	std::vector<phez::PackedCiphertext> masked_uv_sub_r_elems_ct(n_items);
	if (scheme.converter.n_slots != 1)
	{
		#ifndef PSEUDO_MODE
		#pragma omp parallel for
		#endif
		for (size_t i = 0; i < n_items; i++)
			masked_uv_sub_r_elems_ct[i] = scheme.encrypt({ masked_uv_sub_r_bns[i] });

		client_inbound_sizes[idata.client_id] += n_items * scheme.get_ciphertext_nbytes(masked_uv_sub_r_elems_ct[0]); // 4 bytes for each indices is enough.
	}

	// Client
	// =========================================================
	// Compute the reverse mask
	// Element-wise reverse mask
	std::vector<phez::PackedBigNumber> neg_r_mask_pt_elems;
	std::vector<BigNumber> r_mask_bns = scheme.converter.to_bignums(r_mask_pt);
	for (size_t i = 0; i < n_items; i++)
		neg_r_mask_pt_elems.push_back(scheme.converter.negate(scheme.converter.pack({ r_mask_bns[i] }), scheme.converter.slot_size));

	// Unmask gu_v
	idata.gu_v_ct = masked_gu_v_ct;
	for (size_t i = 0; i < n_items; i++)
		idata.gu_v_ct = scheme.add_cc(idata.gu_v_ct, scheme.mul_cp(vs_ct[i], neg_r_mask_pt_elems[i]));


	std::vector<phez::PackedCiphertext> neg_r_mask_ct_elems(n_items);
	if (debug_level >= 1) tc.start();
	#ifndef PSEUDO_MODE
	#pragma omp parallel for
	#endif
	for (size_t i = 0; i < n_items; i++)
		neg_r_mask_ct_elems[i] = scheme.encrypt(neg_r_mask_pt_elems[i]);
	if (debug_level >= 1) std::cout << "(omp) encrypt neg_r_mask_ct_elems: " << tc.tick() / 1000 << "s" << std::endl;

	// Extra work: Compute uv_sub_r_elems_ct
	if (scheme.converter.n_slots != 1)
	{
		idata.uv_sub_r_ct_elems.resize(n_items);
		if (debug_level >= 1) tc.start();
		for (size_t i = 0; i < n_items; i++)
			idata.uv_sub_r_ct_elems[i] = scheme.add_cc(masked_uv_sub_r_elems_ct[i], neg_r_mask_ct_elems[i]);
		if (debug_level >= 1) std::cout << "(-omp) add_cc uv_sub_r_ct_elems: " << tc.tick() / 1000 << "s" << std::endl;
	}
	else
	{
		idata.uv_sub_r_ct_elems = scheme.shatter(scheme.add_cc(idata.uv_ct, scheme.encrypt(neg_rs_pt_l1)));
	}
	// Check gu_v and uv_sub_r_elems
	if (debug_level >= 2)
	{
		std::vector<float> debug__gu_v = scheme.decrypt_to_floats(idata.gu_v_ct, 2);
		zutil::print_vector(debug__gu_v, "gu_v");
		zutil::print_vector(raw_protocol->idata.gu_v, "gu_v real");
		zutil::check_equal(debug__gu_v, raw_protocol->idata.gu_v);

		std::vector<float> debug__uv_sub_r;
		for (size_t i = 0; i < n_items; i++)
			debug__uv_sub_r.push_back(scheme.decrypt_to_floats(idata.uv_sub_r_ct_elems[i], 1)[0]);
		zutil::print_vector(debug__uv_sub_r, "uv_sub_r");
		zutil::print_vector(raw_protocol->idata.out_sub_r, "uv_sub_r real");
		zutil::check_equal(debug__uv_sub_r, raw_protocol->idata.out_sub_r);
	}

}

void drec::paillier1::TrainProtocol::iter_compute_gu_s()
{
	zutil::TimeCounter tc = zutil::time_counter;

	size_t n_friends = idata.selected_friend_ids.size();
	Client client = clients->at(idata.client_id);


	// Client
	// ===============================================================
	if (debug_level >= 1) std::cout << "#Friends: " << n_friends << std::endl;

	std::vector<phez::PackedCiphertext> gs_friend_ct(n_friends);
	if (debug_level >= 1) tc.start();
	#ifndef PSEUDO_MODE
	#pragma omp parallel for
	#endif
	for (size_t i = 0; i < n_friends; i++)
	{
		std::vector<float> neg_friend_emb(embedding_dim);
		size_t friend_id = idata.selected_friend_ids[i];
		// float trust_score = client.friend_trusts[i].second;
		for (size_t j = 0; j < embedding_dim; j++)
		{
			neg_friend_emb[j] = -clients->at(friend_id).embedding[j] * reg_s;
		}
		phez::PackedBigNumber neg_friend_emb_pt = scheme.converter.pack(neg_friend_emb);
		neg_friend_emb_pt = scheme.converter.reduce_multiplication_level(neg_friend_emb_pt, -1);

		// First, chec for the caches
		if (cached_social_neg_us_ct[friend_id].n_elements == 0) cached_social_neg_us_ct[friend_id] = scheme.encrypt(neg_friend_emb_pt);
	}
	if (debug_level >= 1) std::cout << "(omp) encrypt friend_emb: " << tc.tick() / 1000 << "s" << std::endl;


	if (debug_level >= 1) tc.start();
	idata.gu_s_friends_ct = cached_social_neg_us_ct[idata.selected_friend_ids[0]];
	for (size_t i = 1; i < n_friends; i++)  idata.gu_s_friends_ct = scheme.add_cc(idata.gu_s_friends_ct, cached_social_neg_us_ct[idata.selected_friend_ids[i]]);
	idata.gu_s_friends_ct = scheme.mul_cp(idata.gu_s_friends_ct, 2.0 / n_friends);
	if (debug_level >= 1) std::cout << "add_cc gu_s_ct: " << tc.tick() / 1000 << "s" << std::endl;

	idata.gu_s_self.clear();
	for (size_t i = 0; i < embedding_dim; i++)
		idata.gu_s_self.push_back(2 * reg_s * client.embedding[i]);
	// ===== Debug level 2 =====
	if (debug_level >= 2)
	{
		// This is for debug: to check that whether weighted_v_ct is correct
		std::vector<float> debug__gu_s_friend_pt = scheme.decrypt_to_floats(idata.gu_s_friends_ct, 2);
		std::vector<float> gu_s;
		for (size_t i = 0; i < embedding_dim; i++)
			gu_s.push_back(idata.gu_s_self[i] + debug__gu_s_friend_pt[i]);
		zutil::print_vector(gu_s, "gu_s");
		zutil::print_vector(raw_protocol->idata.gu_s, "gu_s real");
		zutil::check_equal(gu_s, raw_protocol->idata.gu_s);
	}
}



void drec::paillier1::TrainProtocol::iter_compute_gu()
{
	Client client = clients->at(idata.client_id);

	// Client	
	// ==============================
	phez::PackedCiphertext gu_ct = scheme.encrypt(scheme.converter.pack(std::vector<float>(embedding_dim, 0)));
	if (idata.selected_item_ids.size() != 0)
	{
		iter_compute_gu_v();
		gu_ct = scheme.add_cc(gu_ct, idata.gu_v_ct);
	}
	if (idata.selected_friend_ids.size() != 0)
	{
		iter_compute_gu_s();
		gu_ct = scheme.add_cc(gu_ct, idata.gu_s_friends_ct);
	}

	phez::PackedBigNumber gu_ring_mask_pt = scheme.random_on_ring(gu_ct);
	phez::PackedCiphertext gu_ring_mask_ct = scheme.encrypt(gu_ring_mask_pt);
	phez::PackedCiphertext masked_gu_ct = scheme.add_cc(gu_ct, scheme.encrypt(gu_ring_mask_pt));

	// User send the masked Gu to the server
	client_outbound_sizes[idata.client_id] += scheme.get_ciphertext_nbytes(masked_gu_ct);
	client_comm_rounds[idata.client_id] += 1;

	// Server
	// ===================================================================
	// Server decrypts the masked gradient, then send it back to the client
	phez::PackedBigNumber masked_gu_pt = scheme.decrypt(masked_gu_ct);

	client_inbound_sizes[idata.client_id] += scheme.converter.get_packed_nbytes(masked_gu_pt);
	client_comm_rounds[idata.client_id] += 1;

	// Client
	// ===================================================================
	phez::PackedBigNumber gu_pt = scheme.add_pp(masked_gu_pt, scheme.negate_on_ring(gu_ring_mask_pt));
	gu_pt = scheme.converter.reduce_multiplication_level(gu_pt, 2);

	idata.gu = scheme.converter.to_floats(gu_pt);

	if (idata.selected_friend_ids.size() != 0)
		for (size_t i = 0; i < embedding_dim; i++)
			idata.gu[i] += idata.gu_s_self[i];
	

	// Check gu
	if (debug_level >= 2)
	{
		std::vector<BigNumber> debug__gu_bns = scheme.converter.to_bignums(gu_pt);
		std::vector<size_t> debug__gu_bits;
		for (size_t i = 0; i < embedding_dim; i++)
			debug__gu_bits.push_back(zutil::get_bit_length(debug__gu_bns[i]));
		zutil::print_vector(debug__gu_bits, "gu bits");

		zutil::print_vector(idata.gu, "gu");
		zutil::print_vector(raw_protocol->idata.gu, "gu real");
		zutil::check_equal(idata.gu, raw_protocol->idata.gu);
	}
}


void drec::paillier1::TrainProtocol::iter_compute_gv()
{
	zutil::TimeCounter tc = zutil::time_counter;

	size_t n_friends = idata.selected_friend_ids.size();
	size_t n_items = idata.selected_item_ids.size();
	Client client = clients->at(idata.client_id);

	// Client
	// =========================================================================
	idata.gvs_ct.resize(n_items);

	if (debug_level >= 1) tc.start();
	#ifndef PSEUDO_MODE
	#pragma omp parallel for
	#endif
	for (size_t i = 0; i < n_items; i++) 
		idata.gvs_ct[i] = scheme.mul_cp(idata.uv_sub_r_ct_elems[i], client.embedding);
	if (debug_level >= 1) std::cout << "(omp) mul_cp gvs_ct: " << tc.tick() / 1000 << "s" << std::endl;

	// Check gv
	if (debug_level >= 2)
	{
		for (size_t i = 0; i < n_items; i++)
		{
			std::vector<float> debug__gv = scheme.decrypt_to_floats(idata.gvs_ct[i], 2);
			zutil::print_vector(debug__gv, "gv");
			zutil::print_vector(raw_protocol->idata.gvs[i], "gv real");
			zutil::check_equal(debug__gv, raw_protocol->idata.gvs[i]);

			std::vector<BigNumber> debug__gv_bns = scheme.decrypt_to_bignumbers(idata.gvs_ct[i]);
			std::vector<size_t> debug__gv_bits;
			for (size_t i = 0; i < embedding_dim; i++)
				debug__gv_bits.push_back(zutil::get_bit_length(debug__gv_bns[i]));
			zutil::print_vector(debug__gv_bits, "gv bits");
		}
	}

	// Client send the gu_v and the sum of mask to server (hence the server only knows the aggregated i grad)
	// It performs simultaneously with compute_gu, hence the communication round is not considered
	if (n_items > 0)
		client_outbound_sizes[idata.client_id] += n_items * scheme.get_ciphertext_nbytes(idata.gvs_ct[0]);

}

void drec::paillier1::TrainProtocol::update()
{
	zutil::TimeCounter tc = zutil::time_counter;
	size_t n_items = idata.selected_item_ids.size();
	Client client = clients->at(idata.client_id);

	// Client
	// ====================================================================
	// Update the client embedding
	std::vector<float> client_update(embedding_dim, 0);
	for (size_t i = 0; i < embedding_dim; i++)
	{
		if (i != 1) client_update[i] += -2 * reg_u * client.embedding[i];
		client_update[i] += -2 * idata.gu[i];
	}
	optimizer->user_step(idata.client_id, client_update);
	// Set the cache invalid
	cached_social_neg_us_ct[idata.client_id].n_elements = 0;

	// Recommender
	// ==============================================================================
	// Update the item updates
	if (debug_level >= 1) tc.start();
	for (size_t i = 0; i < n_items; i++)
	{
		size_t item_id = idata.selected_item_ids[i];
		item_update_buffer[item_id].push_back(idata.gvs_ct[i]);
		if (item_update_buffer[item_id].size() == n_aggregation)
		{
			phez::PackedCiphertext aggregated_gv_ct = item_update_buffer[item_id][0];
			for (size_t j = 1; j < n_aggregation; j++)
			{
				aggregated_gv_ct = scheme.add_cc(aggregated_gv_ct, item_update_buffer[item_id][j]);
			}
			std::vector<float> aggregated_gv = scheme.decrypt_to_floats(aggregated_gv_ct, 2);
			if (debug_level >= 2) zutil::print_vector(aggregated_gv, "Aggregated gv");

			std::vector<float> item_update(embedding_dim, 0);
			// *********************************************************
			// Update the item embedding!
			// *********************************************************
			for (size_t j = 0; j < embedding_dim; j++)
			{
				if (j != 0) item_update[j] += -2 * reg_i * recommender->item_embeddings[item_id][j];
				item_update[j] += -2 * aggregated_gv[j];
			}
			optimizer->item_step(item_id, item_update);
			item_update_buffer[item_id].clear();
			cached_vs_ct[item_id].n_elements = 0; // Set the cache invalid
		}
	}
	if (debug_level >= 1) std::cout << "decrypt item updates: " << tc.tick() / 1000 << "s" << std::endl;
}



void drec::paillier1::TrainProtocol::run_one_iteration(size_t client_id, std::vector<size_t>& selected_item_indices, std::vector<size_t>& selected_friend_indices)
{

	if (debug_level >= 2) raw_protocol->iter_compute_grads(client_id, selected_item_indices, selected_friend_indices);

	zutil::TimeCounter tc;
	tc.start();

	iter_fetch_data(client_id, selected_item_indices, selected_friend_indices);
	iter_compute_uv(); // inference only need the line
	// iter_compute_gu();
	// iter_compute_gv();
	// update();

	if (debug_level >= 1) std::cout << "Current iteration time: " << tc.tick() / 1000 << "s" << std::endl;
}


void drec::paillier1::TrainProtocol::run_one_client(size_t client_id)
{
	Client& client = clients->at(client_id);
	if (client.item_ratings.size() == 0) return; // ****************************

	std::vector<std::vector<size_t>> splitted_item_indices = zutil::split_vector(client.item_ratings.size(), n_items_per_iter);
	for (size_t i = 0; i < splitted_item_indices.size(); i++)
	{
		std::vector<size_t> friend_indices = zutil::rand.choice(client.friend_trusts.size(), n_friends_per_iter);
		run_one_iteration(client_id, splitted_item_indices[i], friend_indices);
	}
}

void drec::paillier1::TrainProtocol::run_one_epoch()
{
	size_t n_ratings = 0;
	zutil::TimeCounter tc;
	tc.start();
	for (size_t i = 0; i < clients->size(); i++)
	{
		if (debug_level >= 0)
		{
			std::cout << "==========================\n";
			std::cout << "Run client: " << i << ", processed ratings: " << n_ratings << std::endl;
			std::cout << "Time elapsed: " << tc.tick() / 1000.0 << std::endl;
			std::cout << "Client-Recommender communication: " << (zutil::mean(client_inbound_sizes, i + 1) + zutil::mean(client_outbound_sizes, i + 1)) / 1024 << std::endl;
		}
		run_one_client(i);
		n_ratings += clients->at(i).item_ratings.size();
	}
}