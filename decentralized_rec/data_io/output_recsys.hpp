#pragma once

#include "../recsys.h"
#include "data_utils.hpp"

namespace drec
{
	inline void output_clients(std::vector<drec::Client>& clients, std::string filepath)
	{
		std::ofstream output_file(filepath);
		for (drec::Client client : clients)
		{
			for (size_t i = 0; i < client.embedding.size(); i++)
			{
				output_file << client.embedding[i] << ((i == client.embedding.size() - 1) ? "\n" : " ");
			}
		}
		output_file.close();
	}

	inline void output_recommender(drec::Recommender& recommender, std::string filepath)
	{
		std::ofstream output_file(filepath);
		for (size_t i = 0; i < recommender.item_embeddings.size(); i++)
		{
			std::vector<float> emb = recommender.item_embeddings[i];
			for (size_t j = 0; j < emb.size(); j++)
			{
				output_file << emb[j] << ((j == emb.size() - 1) ? "\n" : " ");
			}
		}
	}
}