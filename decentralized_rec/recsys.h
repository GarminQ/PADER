#pragma once

#include <iostream>
#include <vector>


namespace drec 
{
	typedef std::pair<size_t, float> IndexedValue;
	class Client
	{
	public:
		size_t id;
		std::vector<IndexedValue> item_ratings;
		std::vector<IndexedValue> friend_trusts;

		std::vector<float> embedding;
		Client(size_t _id, size_t embedding_dim);
	};

	class Recommender
	{
	public:
		std::vector<std::vector<float>> item_embeddings;
		Recommender();
		Recommender(size_t n_items, size_t embedding_dim);
	};
}