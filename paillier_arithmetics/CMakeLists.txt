# CMakeList.txt: PaillierMatrix 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)


find_package ( OpenMP )

set (IPCL_DIR "/home/qjm/code/PaillierMatrix/ipcl")

# 将源代码添加到此项目的可执行文件。
add_executable (arithmTest
				"test.cpp"
				"arithmetics.cpp" "arithmetics.h" 
				"convert/convert.cpp" "convert/convert.h" 
				"../utils/random.h" "../utils/random.cpp" "../utils/time_counter.hpp")


target_include_directories (arithmTest PUBLIC "${IPCL_DIR}/include/ipcl/" "${IPCL_DIR}/include/ipcl/ippcrypto/"  ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries (arithmTest "${IPCL_DIR}/lib/ipcl/libipcl.so.2.0.0" ${OpenMP_CXX_LIBRARIES})


add_executable (arithmBenchmark
				"benchmark.cpp"
				"arithmetics.cpp" "arithmetics.h" 
				"convert/convert.cpp" "convert/convert.h" 
				"../utils/random.h" "../utils/random.cpp" "../utils/time_counter.hpp")


target_include_directories (arithmBenchmark PUBLIC "${IPCL_DIR}/include/ipcl/" "${IPCL_DIR}/include/ipcl/ippcrypto/"  ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries (arithmBenchmark "${IPCL_DIR}/lib/ipcl/libipcl.so.2.0.0" ${OpenMP_CXX_LIBRARIES})



# TODO: 如有需要，请添加测试并安装目标。
