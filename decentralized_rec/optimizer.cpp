#include "optimizer.h"

drec::RecOptimizer::RecOptimizer() {};

drec::RecOptimizer::RecOptimizer(size_t _n_items, size_t _n_users, size_t _embedding_dim, float _learning_rate) :
	n_items(_n_items), n_users(_n_users), embedding_dim(_embedding_dim), learning_rate(_learning_rate)
{
	user_embeddings.resize(n_users);
	item_embeddings.resize(n_items);
};

void drec::RecOptimizer::initialize(Recommender& recommender, std::vector<Client>& clients)
{
	for (size_t i = 0; i < clients.size(); i++)
		user_embeddings[i] = &clients[i].embedding;
	for (size_t i = 0; i < recommender.item_embeddings.size(); i++)
		item_embeddings[i] = &recommender.item_embeddings[i];
}

void drec::RecOptimizer::step(std::vector<std::vector<float>*>& embeddings, size_t vec_id, std::vector<float>& update, int no_grad)
{
	for (int i = 0; i < embedding_dim; i++)
	{
		if (i == no_grad) continue;
		embeddings[vec_id]->at(i) += learning_rate * update[i];
	}
}

void drec::RecOptimizer::user_step(size_t user_id, std::vector<float> user_update)
{
	step(user_embeddings, user_id, user_update, 0);
}

void drec::RecOptimizer::item_step(size_t item_id, std::vector<float>& item_update)
{
	step(item_embeddings, item_id, item_update, 1);
}

drec::MomentumRecOptimizer::MomentumRecOptimizer(size_t _n_items, size_t _n_users, size_t _embedding_dim, float _learning_rate, float _momentum) :
	RecOptimizer(_n_items, _n_users, _embedding_dim, _learning_rate), momentum(_momentum)
{
	user_momentum.resize(n_users);
	item_momentum.resize(n_items);
}

void drec::MomentumRecOptimizer::step(std::vector<std::vector<float>*>& embeddings, std::vector<std::vector<float>>& emb_momentum,
	size_t vec_id, std::vector<float>& _update, int no_grad)
{
	std::vector<float> update = _update;
	if (emb_momentum[vec_id].size() != 0)
	{
		for (size_t i = 0; i < embedding_dim; i++) update[i] += momentum * emb_momentum[vec_id][i];
	}
	RecOptimizer::step(embeddings, vec_id, update, no_grad);
	emb_momentum[vec_id] = update;
}

void drec::MomentumRecOptimizer::user_step(size_t user_id, std::vector<float> user_update)
{
	step(user_embeddings, user_momentum, user_id, user_update, 0);
}

void drec::MomentumRecOptimizer::item_step(size_t item_id, std::vector<float>& item_update)
{
	step(item_embeddings, item_momentum, item_id, item_update, 1);
}