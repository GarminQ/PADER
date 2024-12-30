#include "../recsys.h"
#include "data_utils.hpp"
#include "create_recsys.h"

namespace drec
{
	std::tuple<std::vector<Client>, Recommender> create_from_triples(
		std::vector<UserTriple>& uirs, std::vector<UserTriple>& trusts, size_t n_users, size_t n_items, float rating_mean, float rating_std,
		size_t embedding_dim)
	{
		std::vector<Client> clients;
		clients.reserve(n_users);
		for (size_t i = 0; i < n_users; i++)
		{
			clients.push_back(Client(i, embedding_dim));
		}

		for (size_t i = 0; i < uirs.size(); i++)
		{
			clients[uirs[i].user].item_ratings.push_back(IndexedValue(uirs[i].item, (uirs[i].rating - rating_mean) / rating_std));
		}
		for (size_t i = 0; i < trusts.size(); i++)
		{
			if (trusts[i].user < clients.size() && trusts[i].item < clients.size())
				clients[trusts[i].user].friend_trusts.push_back(IndexedValue(trusts[i].item, trusts[i].rating));
		}

		Recommender recommender(n_items, embedding_dim);
		return std::make_tuple(clients, recommender);
	}

	bool load_clients_embedding(std::vector<Client>& clients, std::string filename)
	{
		std::ifstream file(filename);
		if (file.bad()) return false;
		try
		{
			for (size_t i = 0; i < clients.size(); i++)
			{
				for (size_t j = 0; j < clients[i].embedding.size(); j++)
				{
					file >> clients[i].embedding[j];
				}

			}
		}
		catch (const std::ifstream::failure& e)
		{
			return false;
		}
		return true;
	}

	bool save_clients_embedding(std::vector<Client>& clients, std::string filename)
	{
		std::ofstream file(filename);
		if (file.bad()) return false;
		for (size_t i = 0; i < clients.size(); i++)
		{
			Client client = clients[i];
			for (size_t j = 0; j < client.embedding.size() - 1; j++)
			{
				file << client.embedding[j] << " ";
			}
			file << client.embedding[client.embedding.size() - 1] << "\n";
		}
		file.close();
		return true;
	}

	bool load_item_embeddings(Recommender recommeder, std::string filename)
	{
		std::ifstream file(filename);
		try
		{
			for (size_t i = 0; i < recommeder.item_embeddings.size(); i++)
			{
				for (size_t j = 0; j < recommeder.item_embeddings[i].size(); j++)
				{
					file >> recommeder.item_embeddings[i][j];
				}

			}
		}
		catch (const std::ifstream::failure& e)
		{
			return false;
		}
		file.close();
		return true;
	}

	bool save_item_embeddings(Recommender recommender, std::string filename)
	{
		std::ofstream file(filename);
		if (file.bad()) return false;
		for (size_t i = 0; i < recommender.item_embeddings.size(); i++)
		{
			for (size_t j = 0; j < recommender.item_embeddings[i].size() - 1; j++)
			{
				file << recommender.item_embeddings[i][j] << " ";
			}
			file << recommender.item_embeddings[i][recommender.item_embeddings[i].size() - 1] << "\n";
		}
		file.close();
		return true;
	}
}