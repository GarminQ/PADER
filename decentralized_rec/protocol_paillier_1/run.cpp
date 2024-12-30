#include <sys/stat.h>
#include "../data_io/create_recsys.hpp"
#include "../data_io/output_recsys.hpp"
#include "protocol.h"
#include "../../utils/time_counter.hpp"


int main(int argc, char* argv[])
{
	std::string train_file = "/home/qjm/code/PaillierMatrix/data/datasets/epinions/ratings-train_ratio_0.9_seed_1926.txt";
	std::string trust_file = "/home/qjm/code/PaillierMatrix/data/datasets/epinions/trusts.txt";
	std::string output_folder = ".";

	// Epinions: rating mean 3.9917, items/users/ratings: 139739/49290/664824, rating std 1.2068, density 0.000097
	std::size_t n_users = 49290, n_items = 139739;
	float rating_mean = 3.9917, rating_std = 1.2068;

	size_t embedding_dim = 10;
	float learning_rate = 0.003;


	float reg_u = 1;
	float reg_i = 0.5;
	float reg_s = 3;
	size_t n_items_per_iter = 10;
	size_t n_friends_per_iter = 10;
	size_t n_agg = 5;

	for (int i = 1; i < argc; i++)
	{
		std::string arg(argv[i]);
		if (i == 1) output_folder = arg;
		if (i == 2) embedding_dim = std::stoi(arg);
		if (i == 3) learning_rate = std::stof(arg);
	}

	std::cout << "Current configuration: " << std::endl;
	std::cout << "Learning rate=" << learning_rate << std::endl;
	std::cout << "Embedding_dim=" << embedding_dim << std::endl;

	std::stringstream ss;
	ss << "model-d" << embedding_dim << "-regs" << reg_u << "_" << reg_i << "_" << reg_s << "-friends" << n_friends_per_iter << "-aggs" << n_agg;
	std::string folder_name = ss.str();
	mkdir(folder_name.c_str(), 0777);


	std::vector<drec::UserTriple> uirs = drec::read_triples_from_txt(train_file);
	std::vector<drec::UserTriple> trusts = drec::read_triples_from_txt(trust_file);

	drec::Recommender recommender;
	std::vector<drec::Client> clients;
	std::tie(clients, recommender) = drec::create_from_triples(uirs, trusts, n_users, n_items, rating_mean, rating_std, embedding_dim);
	drec::paillier1::TrainProtocol protocol(&recommender, &clients, 2048, learning_rate, reg_u, reg_i, reg_s, n_items_per_iter, n_friends_per_iter, n_agg, "momentum");


	drec::output_recommender(recommender, folder_name + "/item_embeddings_init.csv");
	drec::output_clients(clients, folder_name + "/user_embeddings_init.csv");

	drec::time_counter.start();
	for (size_t i = 0; i < 50; i++)
	{
		std::cout << "Epoch " << i << std::endl;
		protocol.run_one_epoch();
		drec::output_recommender(recommender, folder_name + "/item_embeddings_" + std::to_string(i) + ".csv");
		drec::output_clients(clients, folder_name + "/user_embeddings_" + std::to_string(i) + ".csv");
	}
}

/*
./protocol1Runner lr0.01 0.01
./protocol1Runner lr0.01 0.01
*/