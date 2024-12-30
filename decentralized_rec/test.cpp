#include "recsys.h"
#include "data_io/create_recsys.h"
#include "../utils/time_counter.hpp"


int main()
{
	std::string root = "/home/zf/vs/PaillierMatrix/0e833b41-352d-4868-a46e-92d87872875d/src/data/datasets/";
	zutil::TimeCounter time_counter;
	

	std::cout << "Test reading triples =============== " << std::endl;
	time_counter.start();
	std::vector<drec::UserTriple> uirs = drec::read_triples_from_txt(root + "douban/ratings-train_ratio_0.1_seed_1926.txt");
	std::cout << "UIRs read, size: " << uirs.size() << std::endl;

	std::vector<drec::UserTriple> trusts = drec::read_triples_from_txt(root + "douban/trusts.txt");
	std::cout << "Trusts read, size: " << trusts.size() << std::endl;
	std::cout << "Time spent: " << time_counter.tick() / 1000 << " ms" << std::endl;

	std::cout << "Test create clients and servers =============" << std::endl;
	time_counter.start();
	std::vector<drec::Client> clients;
	drec::Recommender recommender;
	std::tie(clients, recommender) = drec::create_from_triples(uirs, trusts, 2964, 39695, 3.7441, 0.9272);
	std::cout << "Time spent: " << time_counter.tick() / 1000 << " ms" << std::endl;

	std::cout << "Test save embeddings =============" << std::endl;
	drec::save_clients_embedding(clients, "user_embeddings.txt");
	drec::save_item_embeddings(recommender, "item_embeddings.txt");
	std::cout << "Time spent: " << time_counter.tick() / 1000 << " ms" << std::endl;

	std::cout << "Tese load embeddings ========== " << std::endl;
	drec::load_clients_embedding(clients, "user_embeddings.txt");
	drec::load_item_embeddings(recommender, "item_embeddings.txt");
	std::cout << "Time spent: " << time_counter.tick() / 1000 << " ms" << std::endl;

}