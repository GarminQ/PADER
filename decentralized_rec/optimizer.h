#pragma once

#include "recsys.h"
namespace drec
{

	class RecOptimizer
	{
	public:
		size_t n_users;
		size_t n_items;
		size_t embedding_dim;
		std::vector<std::vector<float>*> user_embeddings;
		std::vector<std::vector<float>*> item_embeddings;

		float learning_rate;

		RecOptimizer();

		RecOptimizer(size_t _n_items, size_t _n_users, size_t _embedding_dim, float _learning_rate);

		void initialize(Recommender& recommender, std::vector<Client>& clients);

		virtual void step(std::vector<std::vector<float>*>& embeddings, size_t vec_id, std::vector<float>& update, int no_grad = -1);
		virtual void user_step(size_t user_id, std::vector<float> user_update);
		virtual void item_step(size_t item_id, std::vector<float>& item_update);
	};


	class MomentumRecOptimizer : public RecOptimizer
	{
	public:
		float momentum;
		std::vector<std::vector<float>> user_momentum;
		std::vector<std::vector<float>> item_momentum;
		MomentumRecOptimizer(size_t _n_items, size_t _n_users, size_t _embedding_dim, float _learning_rate, float _momentum);

		virtual void step(std::vector<std::vector<float>*>& embeddings, std::vector<std::vector<float>>& emb_momentum,
			size_t vec_id, std::vector<float>& _update, int no_grad = -1);
		virtual void user_step(size_t user_id, std::vector<float> user_update);
		virtual void item_step(size_t item_id, std::vector<float>& item_update);
	};
}