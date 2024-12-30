#include "protocol_raw/protocol.h"
//#include "protocol_paillier_naive/protocol.h"
#include "protocol_paillier_1/protocol.h"
//#include "protocol_paillier_bipartile/protocol.h"

int main()
{
	drec::Recommender recommender(2, 3); // 2 items, dimension is 3
	recommender.item_embeddings[0] = { 1, 2, -1 };
	recommender.item_embeddings[1] = { 0, 0, 2 };

	drec::Client client0(0, 3);
	client0.item_ratings = { {0, 1}, {1, 2} };
	client0.embedding = { 0, 1, -1 };

	drec::Client client1(1, 3);
	client1.embedding = { 3, 0, 0 };

	client0.friend_trusts = { { 1, 1 } };

	std::vector clients{ client0, client1 };



	int key_size = 2048;
	float lr = 0.01;
	float reg_u = 0;
	float reg_i = 0;
	float reg_s = 0.01;
	size_t n_item_per_iter = 10;
	size_t n_friends_per_iter = 10;
	size_t n_agg = 5;
	std::string optim_type = "momentum";

	drec::paillier1::TrainProtocol protocol(&recommender, &clients, key_size, lr, reg_u, reg_i, reg_s, n_item_per_iter, n_friends_per_iter, n_agg, optim_type);
//	drec::paillier0::TrainProtocol protocol(&recommender, &clients, key_size, lr, reg_u, reg_i, reg_s, n_item_per_iter, n_friends_per_iter, n_agg, optim_type);
//	drec::paillier2::TrainProtocol protocol(&recommender, &clients, key_size, lr, reg_u, reg_i, reg_s, n_item_per_iter, n_friends_per_iter, n_agg, optim_type);
//	drec::raw::TrainProtocol protocol(&recommender, &clients, lr, reg_u, reg_i, reg_s, n_item_per_iter, n_friends_per_iter, n_agg, optim_type);

	// UV = (3, -2)
	// UV - R = (2, -4)
	// WeightedV (Gradient on U) = 2 * (1, 2, -1) - 4 * (0, 0, 2) = (2, 4, -10)
	// WeightedU (Gradient on V) = (2 and -4) * (0, 1, -1) = (0, 2, -2) and (0, -4, 4)
	for (size_t i = 0; i < 1000; i++)
	{
		protocol.run_one_client(0);
	}
	std::getchar();
	return 0;
}