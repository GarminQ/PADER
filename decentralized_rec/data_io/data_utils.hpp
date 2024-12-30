#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <tuple>
#include "../recsys.h"


namespace drec
{

	struct UserTriple
	{
		size_t user;
		size_t item;
		float rating;
	};


	inline std::vector<UserTriple> read_triples_from_txt(std::string path)
	{
		// UIRTuple (normalized), n_users, n_items, rating-mean, rating-std
		std::vector<UserTriple> triples;

		std::ifstream rating_stream(path);
		while (rating_stream.good())
		{
			UserTriple triple;
			rating_stream >> triple.user >> triple.item >> triple.rating;
			triples.push_back(triple);
		}
		rating_stream.close();
		return triples;
	}

	inline void write_vector_to_csv(std::string filename, std::vector<float> vec)
	{
		std::ofstream outfile(filename);
		for (size_t i = 0; i < vec.size(); i++)
			outfile << vec[i] << "\n";
		outfile.close();
	}
}