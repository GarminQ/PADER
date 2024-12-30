#include <vector>
#include <string>
#include <set>
#include <algorithm>

#include "../recsys.h"
#include "../data_io/data_utils.hpp"
#include "../data_io/create_recsys.h"

namespace drec
{
	class RMSEEvaluator
	{
	private:
		std::vector<UserTriple> test_ratings;
		std::vector<UserTriple> train_ratings;
		std::vector<Client>* clients;
		Recommender* recommender;
		float rating_mean;
		float rating_std;
		bool use_approximate_tanh;
	public:
		RMSEEvaluator();
		RMSEEvaluator(std::string test_filepath3, std::vector<Client>* _clients, Recommender* _recommender, float _rating_mean, float _rating_std, bool _use_approximate_tanh = false);
		float eval();
	};



	class HREvaluator
	{
	private:
		std::vector<std::set<size_t>> test_targets;
		std::vector<std::set<size_t>> train_targets;
		std::vector<Client>* clients;
		Recommender* recommender;
		float positive_threshold;
		size_t n_items;
	public:
		HREvaluator();
		HREvaluator(std::string test_filepath, std::string training_filepath, std::vector<Client>* _clients, Recommender* _recommender, float _positive_threshold);
		float eval(size_t n_predictions);
	};
}