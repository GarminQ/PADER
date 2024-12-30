cmake_minimum_required (VERSION 3.8)


find_package ( OpenMP )

set (IPCL_DIR "/home/qjm/code/PaillierMatrix/ipcl")


add_executable (convertTest
				"convert.cpp" "convert.h" "test.cpp"
				"${CMAKE_SOURCE_DIR}/utils/random.cpp" "${CMAKE_SOURCE_DIR}/utils/random.h")


target_include_directories (convertTest PUBLIC "${IPCL_DIR}/include/ipcl/" "${IPCL_DIR}/include/ipcl/ippcrypto/" ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries (convertTest "${IPCL_DIR}/lib/ipcl/libipcl.so.2.0.0" ${IPPCP_LIB} ${OpenMP_CXX_LIBRARIES})