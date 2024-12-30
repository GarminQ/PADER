# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

project ("PaillierMatrix")

# 包含子项目。
include_directories(/home/qjm/code/PaillierMatrix/ipcl)

add_subdirectory ("utils")

add_subdirectory ("paillier_arithmetics")
add_subdirectory ("paillier_arithmetics/convert")

add_subdirectory ("decentralized_rec")

add_subdirectory ("toys")
