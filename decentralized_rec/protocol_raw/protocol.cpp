#include "protocol.h"
#include "../../utils/random.h"
#include "../../utils/math.hpp"
#include "../../utils/ios.hpp"

const int debug_level = 0;


drec::raw::TrainProtocol::TrainProtocol(drec::Recommender* _recommender, std::vector<drec::Client>* _clients,
	float _learning_rate, float _reg_u, float _reg_i, float _reg_s, size_t _n_items_per_iter, size_t _n_friends_per_iter, size_t _n_aggregation, std::string optimizer_type,
	bool _use_approximate_tanh) :
	recommender(_recommender), clients(_clients),
	learning_rate(_learning_rate), reg_u(_reg_u), reg_i(_reg_i), reg_s(_reg_s), 
	n_items_per_iter(_n_items_per_iter), n_friends_per_iter(_n_friends_per_iter), n_aggregation(_n_aggregation), use_approximate_tanh(_use_approximate_tanh)
{
	item_update_buffer.resize(recommender->item_embeddings.size());

	// Initialize the optimizer
	if (optimizer_type == "momentum")
		optimizer = new drec::MomentumRecOptimizer(recommender->item_embeddings.size(), clients->size(), 
			recommender->item_embeddings[0].size(), learning_rate, 0.9f);

	else
		optimizer = new drec::RecOptimizer(recommender->item_embeddings.size(), clients->size(),
			recommender->item_embeddings[0].size(), learning_rate);

	optimizer->initialize(*recommender, *clients);
}

void drec::raw::TrainProtocol::iter_compute_grads(size_t client_id, std::vector<size_t>& item_indices, std::vector<size_t>& friend_indices)
{
	Client& client = clients->at(client_id);
	size_t n_items = item_indices.size();
	size_t embedding_dim = client.embedding.size();

	idata.client_id = client_id;

	idata.selected_item_ids.clear();
	for (size_t i = 0; i < n_items; i++)
		idata.selected_item_ids.push_back(client.item_ratings[item_indices[i]].first);

	// Create ratings
	idata.rs.clear();
	for (size_t i = 0; i < n_items; i++)
		idata.rs.push_back(client.item_ratings[item_indices[i]].second);

	idata.vs.clear();
	for (size_t i = 0; i < n_items; i++)
		idata.vs.push_back(recommender->item_embeddings[idata.selected_item_ids[i]]);


	// Compute the inner product between user embedding & item embedding
	idata.uvs.clear();
	for (size_t i = 0; i < n_items; i++)
	{
		float uv = 0;
		for (size_t j = 0; j < embedding_dim; j++)
			uv += client.embedding[j] * idata.vs[i][j];

		idata.uvs.push_back(uv);
	}

	idata.outs.clear();
	if (use_approximate_tanh)
		for (size_t i = 0; i < n_items; i++)
			idata.outs.push_back(zutil::approximate_tanh(idata.uvs[i]));
	else
		for (size_t i = 0; i < n_items; i++)
			idata.outs.push_back(idata.uvs[i]);

	// Debug
	if (debug_level >= 2) zutil::print_vector(idata.uvs, "UVs");

	idata.out_sub_r.clear();
	for (size_t i = 0; i < n_items; i++)
		idata.out_sub_r.push_back(idata.outs[i] - idata.rs[i]);

	idata.g_uv.clear();
	if (use_approximate_tanh)
		for (size_t i = 0; i < n_items; i++)
			idata.g_uv.push_back(idata.out_sub_r[i] * zutil::approximate_tanh_diff(idata.uvs[i]));
	else
		for (size_t i = 0; i < n_items; i++)
			idata.g_uv.push_back(idata.out_sub_r[i]);

	idata.gu_v.clear();
	idata.gu_v.resize(embedding_dim, 0);
	for (size_t i = 0; i < embedding_dim; i++)
		for (size_t j = 0; j < n_items; j++)
			idata.gu_v[i] += idata.out_sub_r[j] * idata.vs[j][i];

	idata.gu_s.clear();
	idata.gu_s.resize(embedding_dim, 0);
	size_t n_friends = friend_indices.size();


	idata.selected_friend_ids.clear();
	for (size_t i = 0; i < n_friends; i++)
	{
		idata.selected_friend_ids.push_back(client.friend_trusts[i].first);
		idata.selected_friend_trusts.push_back(client.friend_trusts[i].second);
	}

	if (n_friends > 0)
	{
		for (size_t i = 0; i < embedding_dim; i++)
		{
			for (size_t j = 0; j < n_friends; j++)
			{
				size_t k = friend_indices[j];
				size_t friend_id = idata.selected_friend_ids[j];
				size_t friend_trust = idata.selected_friend_trusts[j];
				idata.gu_s[i] += 2 * reg_s * friend_trust * (client.embedding[i] - clients->at(friend_id).embedding[i]);
			}
			idata.gu_s[i] /= float(n_friends);
		}
	}

	idata.gu.clear();
	idata.gu.resize(embedding_dim, 0);
	for (size_t i = 0; i < embedding_dim; i++)
		idata.gu[i] = idata.gu_v[i] + idata.gu_s[i];

	idata.gvs.resize(n_items);
	for (size_t i = 0; i < n_items; i++)
	{
		idata.gvs[i].clear();
		idata.gvs[i].resize(embedding_dim, 0);
		for (size_t j = 0; j < embedding_dim; j++)
			idata.gvs[i][j] = (idata.uvs[i] - idata.rs[i]) * client.embedding[j];
	}
}

void drec::raw::TrainProtocol::update()
{
	Client& client = clients->at(idata.client_id);
	size_t embedding_dim = client.embedding.size();
	size_t n_items = idata.selected_item_ids.size();

	std::vector<float> client_update(embedding_dim, 0);
	// Client update
	for (size_t i = 0; i < embedding_dim; i++)
	{
		client_update[i] += -2 * idata.gu[i];
		if (i != 1) // We don't put regularization on user bias! 
			client_update[i] += -2 * reg_u * client.embedding[i];
	}
	optimizer->user_step(idata.client_id, client_update);


	// Server check item update
	for (size_t i = 0; i < n_items; i++)
	{
		size_t item_id = idata.selected_item_ids[i];
		item_update_buffer[item_id].push_back(idata.gvs[i]);
		// Aggregate
		if (item_update_buffer[item_id].size() == n_aggregation)
		{
			std::vector<float> aggregated_gv(embedding_dim, 0);
			std::vector<float> item_update(embedding_dim, 0);
			for (size_t j = 0; j < n_aggregation; j++)
				for (size_t k = 0; k < embedding_dim; k++)
					aggregated_gv[k] += item_update_buffer[item_id][j][k];

			for (size_t j = 0; j < embedding_dim; j++)
			{
				item_update[j] += -2 * aggregated_gv[j];
				if (j != 0) // We don't put regularization on item bias!
					item_update[j] += -2 * reg_i * recommender->item_embeddings[item_id][j];
			}
			optimizer->item_step(item_id, item_update);


			item_update_buffer[item_id].clear();
		}
	}
}

void drec::raw::TrainProtocol::run_one_client(size_t client_id)
{
	Client& client = clients->at(client_id);

	std::vector<std::vector<size_t>> splitted_item_indices = zutil::split_vector(client.item_ratings.size(), n_items_per_iter);
	for (size_t i = 0; i < splitted_item_indices.size(); i++)
	{
		std::vector<size_t> friend_indices = zutil::rand.choice(client.friend_trusts.size(), n_friends_per_iter);
		iter_compute_grads(client_id, splitted_item_indices[i], friend_indices);
		update();
	}

}


void drec::raw::TrainProtocol::run_one_epoch()
{
	for (size_t i = 0; i < clients->size(); i++)
		run_one_client(i);
}