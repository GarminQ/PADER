#pragma once
#include <vector>
#include "../recsys.h"
#include "../optimizer.h"

namespace drec::raw
{
	struct Idata
	{
		size_t client_id;
		std::vector<size_t> selected_item_ids;
		std::vector<size_t> selected_friend_ids;
		std::vector<float> selected_friend_trusts;
		std::vector<float> rs;  // ratings
		std::vector<std::vector<float>> vs;  // item embeddngs
		std::vector<float> uvs;   // product of user embedding and item embeddings
		std::vector<float> outs;
		std::vector<float> out_sub_r;
		std::vector<float> g_uv;
		std::vector<float> gu_v;  // user gradient from item embeddings
		std::vector<float> gu_s;  // user gradient from friend embeddings
		std::vector<float> gu;  // complete user gradient
		std::vector<std::vector<float>> gvs;  // item gradients
	};


	class TrainProtocol
	{
	public:
		std::vector<drec::Client>* clients;
		drec::Recommender* recommender;

		// This buffer is for the late update of item embedding.
		// Multiple gradients of one item update is
		std::vector<std::vector<std::vector<float>>> item_update_buffer;
		size_t n_aggregation;
		size_t n_items_per_iter;
		size_t n_friends_per_iter;


		size_t n_rounds;

		// Leanring rate for matrix factorization
		float learning_rate;
		// Regularizations on user/item/social components
		float reg_u;
		float reg_i;
		float reg_s;
		
		bool use_approximate_tanh;

		Idata idata;

		drec::RecOptimizer* optimizer;

		TrainProtocol(drec::Recommender* _recommender, std::vector<drec::Client>* _clients,
			float _learning_rate, float _reg_u, float _reg_i, float _reg_s, size_t _n_items_per_iter, size_t _n_friends_per_iter, size_t _n_aggregation, 
			std::string optimizer_type, bool _use_approximate_tanh = false);

		void iter_compute_grads(size_t client_id, std::vector<size_t>& selected_indices, std::vector<size_t>& friend_indices);
		void update();
		void run_one_client(size_t client_id);
		void run_one_epoch();
	};
}