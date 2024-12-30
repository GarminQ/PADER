#pragma once

#include "../recsys.h"
#include "data_utils.hpp"


namespace drec
{
	std::tuple<std::vector<Client>, Recommender> create_from_triples(
		std::vector<UserTriple>& uirs, std::vector<UserTriple>& trusts, size_t n_users, size_t n_items, float rating_mean, float rating_std,
		size_t embedding_dim = 32);

	bool load_clients_embedding(std::vector<Client>& clients, std::string filename);

	bool save_clients_embedding(std::vector<Client>& clients, std::string filename);

	bool load_item_embeddings(Recommender recommeder, std::string filename);

	bool save_item_embeddings(Recommender recommender, std::string filename);
}