#include <math.h>
#include "../../utils/math.hpp"
#include "../../utils/random.h"

#include "evaluation.h"

namespace drec
{
	RMSEEvaluator::RMSEEvaluator()
	{
	}
	RMSEEvaluator::RMSEEvaluator(std::string test_filepath,
		std::vector<Client>* _clients, Recommender* _recommender, float _rating_mean, float _rating_std, bool _use_approximate_tanh):
		clients(_clients), recommender(_recommender), rating_mean(_rating_mean), rating_std(_rating_std), use_approximate_tanh(_use_approximate_tanh)
	{
		test_ratings = read_triples_from_txt(test_filepath);
	}

	float RMSEEvaluator::eval()
	{
		size_t embedding_dim = recommender->item_embeddings[0].size();
		double mse_sum = 0;
		for (size_t i = 0; i < test_ratings.size(); i++)
		{
			size_t user_id = test_ratings[i].user;
			size_t item_id = test_ratings[i].item;
			float rating = test_ratings[i].rating;
			
			float predicted_rating = 0;
			for (size_t j = 0; j < embedding_dim; j++)
				predicted_rating += recommender->item_embeddings[item_id][j] * clients->at(user_id).embedding[j];
			
			if (use_approximate_tanh) predicted_rating = zutil::approximate_tanh(predicted_rating);
			predicted_rating = rating_std * predicted_rating + rating_mean;
			mse_sum += pow(predicted_rating - rating, 2);
		}
		float rmse = sqrt(mse_sum / test_ratings.size());
		return rmse;
	}


	struct RatingComparator
	{
		bool operator()(std::tuple<size_t, float> r1, std::tuple<size_t, float> r2)
		{
			return std::get<1>(r1) < std::get<1>(r2);
		}
	} compare_rating;

	HREvaluator::HREvaluator(){}

	HREvaluator::HREvaluator(std::string test_filepath, std::string train_filepath,
		std::vector<Client>* _clients, Recommender* _recommender, float _positive_threshold) :
		clients(_clients), recommender(_recommender), positive_threshold(_positive_threshold)
	{

		train_targets.resize(clients->size());
		test_targets.resize(clients->size());
		auto test_ratings = read_triples_from_txt(test_filepath);
		auto train_ratings = read_triples_from_txt(train_filepath);
		for (size_t i = 0; i < train_ratings.size(); i++)
		{
			size_t user_id = train_ratings[i].user;
			size_t item_id = train_ratings[i].item;
			float rating = train_ratings[i].rating;
			if (rating >= positive_threshold)
				train_targets[user_id].emplace(item_id);
		}

		for (size_t i = 0; i < test_ratings.size(); i++)
		{
			size_t user_id = test_ratings[i].user;
			size_t item_id = test_ratings[i].item;
			float rating = test_ratings[i].rating;
			if (rating >= positive_threshold && train_targets[user_id].find(item_id) == train_targets[user_id].end())
				test_targets[user_id].emplace(item_id);
		}
		n_items = recommender->item_embeddings.size();
	}

	float HREvaluator::eval(size_t n_predictions)
	{
		size_t embedding_dim = recommender->item_embeddings[0].size();
		std::vector<std::set<size_t>> predicted_items(clients->size());
		

		std::vector<float> hit_ratios(clients->size(), 0);
		float valid_clients = 0;
#pragma omp parallel for
		for (size_t i = 0; i < clients->size(); i++)
		{
			if (test_targets[i].empty()) continue;
			valid_clients += 1;
			std::vector<std::tuple<size_t, float>> client_i_item_ratings;

			std::vector<size_t> candiate_sets;
			for (std::set<size_t>::iterator elem = test_targets[i].begin(); elem != test_targets[i].end(); elem++)
				candiate_sets.push_back(*elem);
			std::vector<size_t> random_samples = zutil::rand.choice(n_items, 100);
			candiate_sets.insert(candiate_sets.end(), random_samples.begin(), random_samples.end());

			for (size_t j = 0; j < candiate_sets.size(); j++)
			{
				size_t item_id = candiate_sets[j];
				float rating = 0;
				for (size_t k = 0; k < embedding_dim; k++)
					rating += clients->at(i).embedding[k] * recommender->item_embeddings[item_id][k];

				if (client_i_item_ratings.size() < n_predictions)
				{
					client_i_item_ratings.push_back({ item_id, rating });
				}
				else
				{
					std::vector<std::tuple<size_t, float>>::iterator min_rating = std::min_element(client_i_item_ratings.begin(), client_i_item_ratings.end(), compare_rating);
					if (rating > std::get<1>(*min_rating))
					{
						*min_rating = { item_id, rating };
					}
				}
			}

			size_t n_hit = 0;
			for (size_t j = 0; j < n_predictions; j++)
				if (test_targets[i].find(std::get<0>(client_i_item_ratings[j])) != test_targets[i].end())
				{
					n_hit += 1;
					break;
				}

			hit_ratios[i] = float(n_hit) / float(n_predictions);
		}

		float hit_ratio = zutil::sum(hit_ratios) / valid_clients;
		std::cout << hit_ratio << std::endl;
		return hit_ratio;
	}
}