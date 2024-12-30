#include <sstream>
#include <sys/stat.h>
#include <omp.h>
#include "data_io/create_recsys.h"
#include "data_io/output_recsys.hpp"
#include "data_io/data_utils.hpp"
#include "../utils/time_counter.hpp"
#include "../utils/random.h"
#include "evaluation/evaluation.h"

#include "protocol_raw/protocol.h"
//#include "protocol_paillier_naive/protocol.h"
#include "protocol_paillier_1/protocol.h"
#include "protocol_paillier_bipartile/protocol.h"

std::string prefix = "/home/qjm/code/PaillierMatrix/data/datasets/";

std::string dataset = "epinions";
std::string ratio = "0.9";

std::string train_file = prefix + dataset + "/ratings-train_ratio_" + ratio + "_seed_1926.txt";
std::string trust_file = prefix + dataset + "/trusts.txt";
std::string test_file = prefix + dataset + "/ratings-test_ratio_" + ratio + "_seed_1926.txt";
std::string output_folder = ".";

// Epinions-0.9
size_t n_users = 49290, n_items = 139739;
float rating_mean = 3.9917, rating_std = 1.2068;

// Douban-0.1
// size_t n_users = 2964, n_items = 39695;
// float rating_mean = 3.7441, rating_std = 0.9272;

// FilmTrust-0.9
// size_t n_users = 1509, n_items = 2072;
// float rating_mean = 3.0028, rating_std = 0.9187;

// Ciao-0.9
// size_t n_users = 2379, n_items = 16862;
// float rating_mean = 4.2218, rating_std = 1.0074;


// Parameters related to training
size_t embedding_dim = 16;
float learning_rate = 0.003;

size_t n_items_per_iter = 8; // 10
size_t n_friends_per_iter = 10;
size_t n_agg = 1; // 2

std::string optim_type = "sgd";
size_t key_size = 2048;

bool use_approximate_tanh = false;

std::vector<float> train(std::string protocol_name, drec::Recommender* recommender, std::vector<drec::Client>* clients, std::string folder_name,
	float reg_u, float  reg_i, float reg_s)
{
	folder_name = folder_name + "-" + protocol_name;
	mkdir(folder_name.c_str(), 0777);
	drec::RMSEEvaluator rmse_evaluator(test_file, clients, recommender, rating_mean, rating_std, use_approximate_tanh);
	drec::HREvaluator hr_evaluator(test_file, train_file, clients, recommender, 3);

	std::vector<float> rmses;
	std::vector<float> hr10s;
	// std::cout<<protocol_name<<std::endl;
	if (protocol_name == "natural")
	{
		drec::paillier1::TrainProtocol protocol(recommender, clients, key_size, learning_rate, reg_u, reg_i, reg_s, n_items_per_iter, n_friends_per_iter, n_agg, optim_type);


		//drec::output_recommender(*recommender, folder_name + "/item_embeddings_init.csv");
		//drec::output_clients(*clients, folder_name + "/user_embeddings_init.csv");

		for (size_t i = 0; i < 1; i++)
		{
			std::cout << "Epoch " << i << std::endl;
			protocol.run_one_epoch();
			rmses.push_back(rmse_evaluator.eval());
			std::cout << rmses.back() << std::endl;
			// hr10s.push_back(hr_evaluator.eval(10));
			// drec::output_recommender(*recommender, folder_name + "/item_embeddings_" + std::to_string(i) + ".csv");
			// drec::output_clients(*clients, folder_name + "/user_embeddings_" + std::to_string(i) + ".csv");
		}
	}
	else if (protocol_name == "bipartile")
	{
		drec::paillier2::TrainProtocol protocol(recommender, clients, key_size, learning_rate, reg_u, reg_i, reg_s, n_items_per_iter, n_friends_per_iter, n_agg, optim_type);

		//drec::output_recommender(*recommender, folder_name + "/item_embeddings_init.csv");
		//drec::output_clients(*clients, folder_name + "/user_embeddings_init.csv");

		for (size_t i = 0; i < 1; i++)
		{
			std::cout << "Epoch " << i << std::endl;
			protocol.run_one_epoch();
			rmse_evaluator.eval();
			drec::output_recommender(*recommender, folder_name + "/item_embeddings_" + std::to_string(i) + ".csv");
			drec::output_clients(*clients, folder_name + "/user_embeddings_" + std::to_string(i) + ".csv");
		}
	}
	else if (protocol_name == "raw")
	{
		drec::raw::TrainProtocol protocol(recommender, clients, learning_rate, reg_u, reg_i, reg_s, n_items_per_iter, n_friends_per_iter, n_agg, optim_type, use_approximate_tanh);

		//drec::output_recommender(*recommender, folder_name + "/item_embeddings_init.csv");
		//drec::output_clients(*clients, folder_name + "/user_embeddings_init.csv");

		for (size_t i = 0; i < 150; i++)
		{
			if (i % 10 == 0) std::cout << "Epoch " << i << std::endl;
			protocol.run_one_epoch();
			rmses.push_back(rmse_evaluator.eval());
			//hr10s.push_back(hr_evaluator.eval(10));
			//drec::output_recommender(*recommender, folder_name + "/item_embeddings_" + std::to_string(i) + ".csv");
			//drec::output_clients(*clients, folder_name + "/user_embeddings_" + std::to_string(i) + ".csv");
		}
		for (size_t i = 0; i < rmses.size(); i++)
			std::cout << rmses[i] << "\n";
	}
	else 
	{
		std::cout << "Invalid protocol name, exit" << std::endl;
	}
	std::stringstream save_name;
	save_name << dataset << ratio << "-" << protocol_name << "-" << reg_u << "_" << reg_i << "_" << reg_s << "ahe";
	drec::write_vector_to_csv(save_name.str() + "-rmse.csv", rmses);
	//drec::write_vector_to_csv(save_name.str() + "-hr10.csv", hr10s);
	return rmses;
}

int main(int argc, char* argv[])
{
	std::cout << "Current configuration: " << std::endl;
	std::cout << "Learning rate=" << learning_rate << std::endl;
	std::cout << "Embedding_dim=" << embedding_dim << std::endl;

	std::string protocol_name = "raw";
	// Input parameters
	if (argc >= 2)
	{
		protocol_name = argv[1];
	}
	std::stringstream ss;
	std::string folder_name = "tmp";
	
	std::vector<drec::UserTriple> uirs = drec::read_triples_from_txt(train_file);
	std::vector<drec::UserTriple> trusts = drec::read_triples_from_txt(trust_file);

	if(true)
	{
		protocol_name = "bipartile"; //natural
		float reg_u = 2, reg_i = 0.5, reg_s = 0.5;
		drec::Recommender recommender;
		std::vector<drec::Client> clients;
		std::tie(clients, recommender) = drec::create_from_triples(uirs, trusts, n_users, n_items, rating_mean, rating_std, embedding_dim);
		std::vector<float> rmses = train(protocol_name, &recommender, &clients, folder_name, reg_u, reg_i, reg_s);
	}


	std::vector<float> reg_us{ 0.5, 1, 2, 4, 8, 16 };
	std::vector<float> reg_is{ 0.25, 0.5, 1, 2, 4, 8 };
	std::vector<float> reg_ss{ 0, 0.25, 0.5, 1, 2, 4 };

	float uis_rmses[6][6][6];

	if (false)
	{
#pragma omp parallel for private(zutil::rand) collapse(3)
		for (size_t i = 0; i < 6; i++)
			for (size_t j = 0; j < 6; j++)
				for (size_t k = 0; k < 6; k++)
				{
					drec::Recommender recommender;
					std::vector<drec::Client> clients;
					std::tie(clients, recommender) = drec::create_from_triples(uirs, trusts, n_users, n_items, rating_mean, rating_std, embedding_dim);
					std::vector<float> rmses = train(protocol_name, &recommender, &clients, folder_name, reg_us[i], reg_is[j], reg_ss[k]);
					uis_rmses[i][j][k] = *std::min_element(rmses.begin(), rmses.end());
				}
		std::ofstream param_record("param_record-" + dataset + ratio + ".csv");
		param_record << "reg_u,reg_i,reg_s,rmse" << std::endl;
		for (size_t i = 0; i < 6; i++)
			for (size_t j = 0; j < 6; j++)
				for (size_t k = 0; k < 6; k++)
					param_record << reg_us[i] << "," << reg_is[j] << "," << reg_ss[k] << "," << uis_rmses[i][j][k] << std::endl;
		param_record.close();
	}
	return 0;
}

/*
export OMP_NUM_THREADS=25
cd ~/vs/PaillierMatrix/0e833b41-352d-4868-a46e-92d87872875d/out/build/Linux-GCC-Release/decentralized_rec
nohup ./protocolRun log.txt &
nohup ./protocolRun natural > natural.log &
nohup ./protocolRun bipartile > bipartile.log &
*/


