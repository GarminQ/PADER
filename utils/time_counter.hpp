#include <iostream>
#include <chrono>


namespace zutil
{
	class TimeCounter
	{
	public:
		std::chrono::system_clock::time_point start_time;
		void start()
		{
			start_time = std::chrono::high_resolution_clock::now();
		}
		float tick()
		{
			auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
			uint64_t microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
				elapsed).count();
			return (double)microseconds / 1000;
		}
	};
	inline TimeCounter time_counter;
}