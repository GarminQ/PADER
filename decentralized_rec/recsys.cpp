#include "../utils/random.h"

#include "recsys.h"


drec::Client::Client(size_t _id, size_t embedding_dim) : id(_id)
{
	embedding.resize(embedding_dim);
	float scale = std::sqrt(float(embedding_dim));
	for (size_t i = 0; i < embedding_dim; i++)
	{
		embedding[i] = zutil::rand.uniform(-1.F, 1.F) / scale;
	}
	embedding[0] = 1;
}

drec::Recommender::Recommender()
{
}

drec::Recommender::Recommender(size_t n_items, size_t embedding_dim)
{
	item_embeddings.resize(n_items);
	float scale = std::sqrt(float(embedding_dim));
	for (size_t i = 0; i < n_items; i++)
	{
		item_embeddings[i].resize(embedding_dim);
		for (size_t j = 0; j < embedding_dim; j++)
		{
			item_embeddings[i][j] = zutil::rand.uniform(-1.F, 1.F) / scale;
		}
		item_embeddings[i][1] = 1;
	}
}
