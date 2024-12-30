#pragma once
#include <random>
#include <iostream>


namespace zutil
{
    class Random
    {
    public:
        // initialize Mersennes' twister using rd to generate the seed
        std::mt19937 rng;
        Random();
        Random(int seed);
        float uniform(float low, float high);
        int uniform(int low, int high);
        std::vector<size_t> choice(size_t n_total, size_t n_samples);
        uint32_t random_u32();
    };
    extern Random rand;
}