#pragma once

#include <vector>

#include "../recsys.h"
#include "../../paillier_arithmetics/arithmetics.h"
#include "../protocol_raw/protocol.h"
#include "../optimizer.h"

namespace drec::paillier2
{
	struct Idata
	{
		size_t client_id;

		std::vector<size_t> selected_item_ids;
		std::vector<size_t> selected_friend_ids;
		std::vector<float> selected_friend_trusts;

		std::vector<float> rs;

		phez::PackedCiphertext gu_v_ct;
		std::vector<float> gu_s_self;
		phez::PackedCiphertext gu_s_friends_ct;  // Gradient of user from social term
		std::vector<float> gu;
		std::vector<phez::PackedCiphertext> gvs_ct;      // Gradients of items
	};
	class TrainProtocol
	{
	public:
		std::vector<drec::Client>* clients;
		drec::Recommender* recommender;
		phez::EncryptionScheme scheme;

		drec::raw::TrainProtocol* raw_protocol;
		drec::RecOptimizer *optimizer;

		// This buffer is for the late update of item embedding.
		// Multiple gradients of one item update is 
		std::vector<std::vector<phez::PackedCiphertext>> item_update_buffer;

		std::vector<phez::PackedCiphertext> cached_vs_ct;
		std::vector<std::vector<phez::PackedCiphertext>> cached_vs_self_product_ct;
		std::vector<std::vector<phez::PackedCiphertext>> cached_vs_elems_ct;

		std::vector<phez::PackedCiphertext> cached_social_neg_us_ct;

		Idata idata;

		size_t embedding_dim;

		size_t n_items_per_iter;
		size_t n_friends_per_iter;
		size_t n_aggregation;

		size_t n_rounds;

		// Leanring rate for matrix factorization
		float learning_rate;
		// Regularizations on user/item/social components
		float reg_u;
		float reg_i;
		float reg_s;
		
		std::vector<float> client_inbound_sizes;
		std::vector<float> client_outbound_sizes;
		std::vector<size_t> client_comm_rounds;

		TrainProtocol(drec::Recommender* _recommender, std::vector<drec::Client>* _clients, size_t key_size,
			float _learning_rate, float _reg_u, float _reg_i, float _reg_s, 
			size_t _n_items_per_iter, size_t n_friends_per_iter, size_t _n_aggregation, std::string optimizer_type);

		void iter_fetch_data(size_t client_id, std::vector<size_t>& selected_item_indices, std::vector<size_t>& selected_friend_indices);
		void iter_compute_gu_s();
		void iter_compute_gu_v();
		void iter_compute_gu();
		void iter_compute_gv();
		void update();
		void run_one_iteration(size_t client_id, std::vector<size_t>& selected_item_indices, std::vector<size_t>& selected_friend_indices);
		void run_one_client(size_t client_id);
		void run_one_epoch();
	};
}