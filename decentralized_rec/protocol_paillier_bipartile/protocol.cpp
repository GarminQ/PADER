// #include <ipcl/ipcl.hpp>
#include "/home/qjm/code/PaillierMatrix/ipcl/include/ipcl/ipcl.hpp"

#include <omp.h>

#include "protocol.h"
#include "../../utils/random.h"
#include "../../utils/ios.hpp"
#include "../../utils/math.hpp"
#include "../../utils/time_counter.hpp"
#include "../../paillier_arithmetics/arithmetics.h"


const int debug_level = 0;
// const int debug_level = -1;

drec::paillier2::TrainProtocol::TrainProtocol(drec::Recommender* _recommender, std::vector<drec::Client>* _clients, size_t key_size,
	float _learning_rate, float _reg_u, float _reg_i, float _reg_s,
	size_t _n_items_per_iter, size_t _n_friends_per_iter, size_t _n_aggregation, std::string optimizer_type) :
	recommender(_recommender), clients(_clients),
	embedding_dim(_recommender->item_embeddings[0].size()),
	learning_rate(_learning_rate), reg_u(_reg_u), reg_i(_reg_i), reg_s(_reg_s), 
	n_items_per_iter(_n_items_per_iter), n_friends_per_iter(_n_friends_per_iter), n_aggregation(_n_aggregation)
{
	ipcl::KeyPair keyPair = ipcl::generateKeypair(key_size);
	// scheme = phez::EncryptionScheme(keyPair.pub_key, keyPair.priv_key, 56, 4, 16, 23);
	scheme = phez::EncryptionScheme(keyPair.pub_key, keyPair.priv_key, 56, 4, 1, 23);

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

	cached_vs_ct.resize(n_items);
	cached_vs_self_product_ct.resize(n_items);
	cached_vs_elems_ct.resize(n_items);

	cached_social_neg_us_ct.resize(n_users);

	// Initialize the optimizer
	if (optimizer_type == "momentum") optimizer = new drec::MomentumRecOptimizer(n_items, n_users, embedding_dim, learning_rate, 0.9f);
	else optimizer = new drec::RecOptimizer(n_items, n_users, embedding_dim, learning_rate);
	optimizer->initialize(*recommender, *clients);
}

void drec::paillier2::TrainProtocol::iter_fetch_data(size_t client_id, std::vector<size_t>& selected_item_indices, std::vector<size_t>& selected_friend_indices)
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

	// 	client_outbound_sizes[client_id] += selected_item_indices.size() * 4; // 4 bytes for each indices is enough.
	//  client_comm_rounds[client_id] += 1;

	idata.selected_friend_ids.clear();
	idata.selected_friend_trusts.clear();
	for (size_t i = 0; i < selected_friend_indices.size(); i++)
	{
		idata.selected_friend_ids.push_back(client.friend_trusts[i].first);
		idata.selected_friend_trusts.push_back(client.friend_trusts[i].second);
	}
}

void drec::paillier2::TrainProtocol::iter_compute_gu_v()
{
	zutil::TimeCounter tc;
	Client client = clients->at(idata.client_id);

	//===========================================
	// Recommender
	size_t n_items = idata.selected_item_ids.size();

	// The recommender compute the embedding self-product matrix
	// [V1V1, V1V2, ... ]
	// [V2V1, V2V2, ... ]
	// [V3V1, V3V2, ... ]

	std::vector<std::vector<float>> vs(n_items);
	for (size_t i = 0; i < n_items; i++)
		vs[i] = recommender->item_embeddings[idata.selected_item_ids[i]];


	std::vector<std::vector<std::vector<float>>> vs_self_product(n_items);
	for (size_t i = 0; i < n_items; i++)
	{
		std::vector<float> item_embedding = recommender->item_embeddings[idata.selected_item_ids[i]];
		vs_self_product[i].resize(embedding_dim);
		for (size_t j = 0; j < embedding_dim; j++)
		{
			vs_self_product[i][j].resize(embedding_dim);
			for (size_t k = 0; k < embedding_dim; k++)
				vs_self_product[i][j][k] = item_embedding[j] * item_embedding[k];
		}
	}

	tc.start();
	std::vector<phez::PackedCiphertext> vs_ct(n_items);
#pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
	{
		size_t item_id = idata.selected_item_ids[i];
		if (cached_vs_ct[item_id].n_elements == 0) cached_vs_ct[item_id] = scheme.encrypt(vs[i]);
		vs_ct[i] = cached_vs_ct[item_id];
	}

	std::vector<std::vector<phez::PackedCiphertext>> vs_self_product_ct(n_items);
#pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
	{
		size_t item_id = idata.selected_item_ids[i];
		if (cached_vs_self_product_ct[item_id].empty())
		{
			cached_vs_self_product_ct[item_id].resize(embedding_dim);
			for (size_t j = 0; j < embedding_dim; j++)
				cached_vs_self_product_ct[item_id][j] = scheme.encrypt(vs_self_product[i][j]);
		}
		vs_self_product_ct[i] = cached_vs_self_product_ct[item_id];
	}

	client_inbound_sizes[idata.client_id] += n_items * scheme.get_ciphertext_nbytes(vs_ct[0]);
	client_inbound_sizes[idata.client_id] += n_items * embedding_dim * scheme.get_ciphertext_nbytes(vs_self_product_ct[0][0]);
	client_comm_rounds[idata.client_id] += 1;

	// ===========================================
	// Client

	// Create ratings
	std::vector<BigNumber> neg_rs_ct(n_items);
	for (size_t i = 0; i < n_items; i++) neg_rs_ct[i] = scheme.converter.float_to_bignum(-idata.rs[i]);
	// Bipartile computation
	idata.gu_v_ct = scheme.encrypt(std::vector<float>(embedding_dim, 0));

	std::vector<phez::PackedCiphertext> neg_rVs(n_items);
	#pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
		neg_rVs[i] = scheme.mul_cp(vs_ct[i], -idata.rs[i]);
	for (size_t i = 0; i < n_items; i++)
		idata.gu_v_ct = scheme.add_cc(idata.gu_v_ct, neg_rVs[i]);

	std::vector<phez::PackedCiphertext> uvV_sum_i(embedding_dim);
	#pragma omp parallel for
	for (size_t j = 0; j < embedding_dim; j++)
	{
		uvV_sum_i[j] = scheme.mul_cp(vs_self_product_ct[0][j], client.embedding[j]);
		for (size_t i = 1; i < n_items; i++)
			uvV_sum_i[j] = scheme.add_cc(uvV_sum_i[j], scheme.mul_cp(vs_self_product_ct[i][j], client.embedding[j]));
	}

	for (size_t i = 0; i < embedding_dim; i++)
		idata.gu_v_ct = scheme.add_cc(idata.gu_v_ct, uvV_sum_i[i]);

	// Debug
	if (debug_level >= 2)
	{
		std::vector<float> debug__gu_v = scheme.decrypt_to_floats(idata.gu_v_ct, 1);
		raw_protocol->idata.gu_v;
		zutil::print_vector(debug__gu_v, "gu_v");
		zutil::print_vector(raw_protocol->idata.gu_v, "gu_v real");
		zutil::check_equal(debug__gu_v, raw_protocol->idata.gu_v);
	}
}

void drec::paillier2::TrainProtocol::iter_compute_gu_s()
{
	zutil::TimeCounter tc = zutil::time_counter;

	size_t n_friends = idata.selected_friend_ids.size();
	Client client = clients->at(idata.client_id);


	// Client
	// ===============================================================
	if (debug_level >= 1) std::cout << "#Friends: " << n_friends << std::endl;

	std::vector<phez::PackedCiphertext> gs_friend_ct(n_friends);
	if (debug_level >= 1) tc.start();
#pragma omp parallel for
	for (size_t i = 0; i < n_friends; i++)
	{
		std::vector<float> neg_friend_emb(embedding_dim);
		size_t friend_id = idata.selected_friend_ids[i];
		// float trust_score = client.friend_trusts[i].second;
		for (size_t j = 0; j < embedding_dim; j++)
			neg_friend_emb[j] = -clients->at(friend_id).embedding[j] * reg_s;
		phez::PackedBigNumber neg_friend_emb_pt = scheme.converter.pack(neg_friend_emb);

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
		std::vector<float> debug__gu_s_friend_pt = scheme.decrypt_to_floats(idata.gu_s_friends_ct, 1);
		std::vector<float> gu_s;
		for (size_t i = 0; i < embedding_dim; i++)
			gu_s.push_back(idata.gu_s_self[i] + debug__gu_s_friend_pt[i]);
		zutil::print_vector(gu_s, "gu_s");
		zutil::print_vector(raw_protocol->idata.gu_s, "gu_s real");
		zutil::check_equal(gu_s, raw_protocol->idata.gu_s);
	}
}
void drec::paillier2::TrainProtocol::iter_compute_gu()
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
	gu_pt = scheme.converter.reduce_multiplication_level(gu_pt, 1);
	idata.gu = scheme.converter.to_floats(gu_pt);

	if (idata.selected_friend_ids.size() != 0)
		for (size_t i = 0; i < embedding_dim; i++)
			idata.gu[i] += idata.gu_s_self[i];


	// Check gu
	if (debug_level >= 2)
	{
		zutil::print_vector(idata.gu, "gu");
		zutil::print_vector(raw_protocol->idata.gu, "gu real");
		zutil::check_equal(idata.gu, raw_protocol->idata.gu);
	}

}

void drec::paillier2::TrainProtocol::iter_compute_gv()
{
	Client client = clients->at(idata.client_id);
	size_t n_items = idata.selected_item_ids.size();


	// ===============================================================
	// Recommender
	std::vector<std::vector<phez::PackedCiphertext>> vs_elems_ct(n_items);
	#pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
	{
		size_t item_id = idata.selected_item_ids[i];
		if (cached_vs_elems_ct[item_id].empty())
		{
			cached_vs_elems_ct[item_id].resize(embedding_dim);
			if (scheme.converter.n_slots != 1)
				for (size_t j = 0; j < embedding_dim; j++)
					cached_vs_elems_ct[item_id][j] = scheme.encrypt(recommender->item_embeddings[item_id][j]);
			else
				cached_vs_elems_ct[item_id] = scheme.shatter(cached_vs_ct[item_id]);
		}
		vs_elems_ct[i] = cached_vs_elems_ct[item_id];
	}

	if (scheme.converter.n_slots != 1)
		client_inbound_sizes[idata.client_id] += n_items * embedding_dim * scheme.get_ciphertext_nbytes(vs_elems_ct[0][0]);

	// ===============================================================
	// Client

	std::vector<std::vector<float>> u_self_product(embedding_dim);
	for (size_t i = 0; i < embedding_dim; i++)
	{
		u_self_product[i].resize(embedding_dim);
		for (size_t j = 0; j < embedding_dim; j++)
			u_self_product[i][j] = client.embedding[i] * client.embedding[j];
	}

	std::vector<std::vector<float>> neg_rU(n_items);
	for (size_t i = 0; i < n_items; i++)
	{
		neg_rU[i].resize(embedding_dim);
		for (size_t j = 0; j < embedding_dim; j++)
			neg_rU[i][j] += -idata.rs[i] * client.embedding[j];
	}

	std::vector<phez::PackedCiphertext> neg_rU_ct(n_items);
	#pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
		neg_rU_ct[i] = scheme.encrypt(scheme.converter.reduce_multiplication_level(scheme.converter.pack(neg_rU[i]), -1));

	std::vector<phez::PackedCiphertext> uvU_ct(n_items);
	#pragma omp parallel for
	for (size_t i = 0; i < n_items; i++)
	{
		uvU_ct[i] = scheme.mul_cp(vs_elems_ct[i][0], u_self_product[0]);
		for (size_t j = 1; j < embedding_dim; j++)
			uvU_ct[i] = scheme.add_cc(uvU_ct[i], scheme.mul_cp(vs_elems_ct[i][j], u_self_product[j]));
	}

	idata.gvs_ct.resize(n_items);
	for (size_t i = 0; i < n_items; i++)
		idata.gvs_ct[i] = scheme.add_cc(neg_rU_ct[i], uvU_ct[i]);

	if (debug_level >= 2)
	{
		for (size_t i = 0; i < n_items; i++)
		{
			std::vector<float> debug__gv = scheme.decrypt_to_floats(idata.gvs_ct[i],1);
			zutil::print_vector(debug__gv, "gv");
			zutil::print_vector(raw_protocol->idata.gvs[i], "gv real");
			zutil::check_equal(debug__gv, raw_protocol->idata.gvs[i]);
		}
	}

	// Client send the gu_v and the sum of mask to server (hence the server only knows the aggregated i grad)
	// It performs simultaneously with compute_gu, hence the communication round is not considered
	client_outbound_sizes[idata.client_id] += n_items * scheme.get_ciphertext_nbytes(idata.gvs_ct[0]);
}



void drec::paillier2::TrainProtocol::update()
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
			std::vector<float> aggregated_gv = scheme.decrypt_to_floats(aggregated_gv_ct, 1);
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
			cached_vs_elems_ct[item_id].clear();
			cached_vs_self_product_ct[item_id].clear();
		}
	}
	if (debug_level >= 1) std::cout << "decrypt item updates: " << tc.tick() / 1000 << "s" << std::endl;
}



void drec::paillier2::TrainProtocol::run_one_iteration(size_t client_id, std::vector<size_t>& selected_item_indices, std::vector<size_t>& selected_friend_indices)
{
	if (debug_level >= 2) raw_protocol->iter_compute_grads(client_id, selected_item_indices, selected_friend_indices);

	zutil::TimeCounter tc;
	tc.start();
	
	iter_fetch_data(client_id, selected_item_indices, selected_friend_indices);
	iter_compute_gu();
	iter_compute_gv();
	update();

	if (debug_level >= 1) std::cout << "Current iteration time: " << tc.tick() / 1000 << "s" << std::endl;
}


void drec::paillier2::TrainProtocol::run_one_client(size_t client_id)
{
	Client& client = clients->at(client_id);
	if (client.item_ratings.size() == 0) return; //***********************

	std::vector<std::vector<size_t>> splitted_item_indices = zutil::split_vector(client.item_ratings.size(), n_items_per_iter);
	for (size_t i = 0; i < splitted_item_indices.size(); i++)
	{
		std::vector<size_t> friend_indices = zutil::rand.choice(client.friend_trusts.size(), n_friends_per_iter);
		run_one_iteration(client_id, splitted_item_indices[i], friend_indices);
	}
}

void drec::paillier2::TrainProtocol::run_one_epoch()
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