#include "data_io/create_recsys.h"

#include "protocol_raw/protocol.h"
#include "protocol_paillier_1/protocol.h"
#include "protocol_paillier_bipartile/protocol.h"


#include "../utils/time_counter.hpp"
#include "../utils/math.hpp"
#include "../utils/ios.hpp"


size_t embedding_dim = 16;
float learning_rate = 0.003;

float reg_u = 1;
float reg_i = 0.5;
float reg_s = 3;
size_t n_items_per_iter = 100000;
size_t n_friends_per_iter = 100;
size_t n_agg = 1;  // 1

std::string optim_type = "sgd"; // momentum
size_t key_size = 2048;


int benchmark(size_t embedding_dim, size_t n_items, size_t n_friends, size_t n_repeat = 1, bool use_cache = true)
{
	std::vector<drec::UserTriple> uirs;
	std::vector<drec::UserTriple> trusts;
	for (size_t i = 0; i < n_items; i++)
		uirs.push_back(drec::UserTriple{ 0, i, float(i % 2 - 0.5) });
	for (size_t i = 1; i < n_friends + 1; i++)
		trusts.push_back(drec::UserTriple{ 0, i, 1.0f });
	drec::Recommender recommender;
	std::vector<drec::Client> clients;
	std::tie(clients, recommender) = drec::create_from_triples(uirs, trusts, 1 + n_friends, n_items, 0, 1, embedding_dim);


	typedef drec::paillier1::TrainProtocol Protocol;
	Protocol protocol(&recommender, &clients, key_size, learning_rate, reg_u, reg_i, reg_s, n_items_per_iter, n_friends_per_iter, n_agg, optim_type);

	std::vector<float> time_passed;
	for (size_t i = 0; i < n_repeat + 1; i++)
	{
		zutil::TimeCounter tc;
		tc.start();
		protocol.run_one_client(0);
		if (i != 0)
			time_passed.push_back(tc.tick() / 1000);
		if (!use_cache)
			protocol = Protocol(&recommender, &clients, key_size, learning_rate, reg_u, reg_i, reg_s, n_items_per_iter, n_friends_per_iter, n_agg, optim_type);

	}
	zutil::print_vector(time_passed, "time_passed");
	std::cout << "embedding_dim: " << embedding_dim << " n_items:" << n_items << " n_friends: " << n_friends << std::endl;
	// std::cout << "Mean time: " << zutil::mean(time_passed) << std::endl;
	// std::cout << "Std: " << zutil::standard_deviation(time_passed) << std::endl;
	std::cout << "Mean time: " << zutil::mean(time_passed, 1) << std::endl;
	std::cout << "Std: " << zutil::standard_deviation(time_passed) << std::endl;
	std::cout << "Comm. size: " << (protocol.client_inbound_sizes[0] + protocol.client_outbound_sizes[0]) / (1024 * (n_repeat + 1)) << std::endl;
	return 0;
}

int main()
{
	// benchmark(32, 8, 0); 

	// dim, n_items, n_friends
	int dim_values[] = {8, 16, 24, 32, 40};
	// int n_items_values[] = {1, 2, 4, 8, 16, 32, 64};
	int n_friends_values[] = {0, 2, 4, 6, 8, 10};

	// for (int dim : dim_values) {
    //     // benchmark(dim, 8, 10);
	// 	benchmark(dim, 8, 0); // inference
    // }

	int n_items_values[] = {
        1, 2, 4, 8, 16, 32, 64,
        128, 256, 512, 1024, 2048, 4096, 8192, 16384 // 2^0 to 2^14
    };
	for (int n_items : n_items_values) {
        // benchmark(8, n_items, 10);
		benchmark(8, n_items, 10); // inference
    }
	// for (int n_friends : n_friends_values) {
    //     benchmark(8, 8, n_friends);
    // }
}
// export OMP_NUM_THREADS=8
// nohup ./decentralized_rec/protocolBenchmark > benchmark.log &