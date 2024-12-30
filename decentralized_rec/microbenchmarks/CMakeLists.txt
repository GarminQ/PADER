find_package ( OpenMP )

set (IPCL_DIR "/home/qjm/code/PaillierMatrix/ipcl")
set (IPPCP_LIB "/home/qjm/code/PaillierMatrix/ipcl/lib/ipcl/ippcrypto/libippcp.so.11")
set (ROOT ${CMAKE_SOURCE_DIR})

 
add_executable (vec_op_benchmarks 
				# Basic Libs 
				"${ROOT}/utils/random.h" "${ROOT}/utils/random.cpp" "${ROOT}/utils/time_counter.hpp"
				# Paillier libs
				"${ROOT}/paillier_arithmetics/arithmetics.cpp" "${ROOT}/paillier_arithmetics/arithmetics.h" 
				"${ROOT}/paillier_arithmetics/convert/convert.cpp" "${ROOT}/paillier_arithmetics/convert/convert.h" 
				"${ROOT}/utils/random.h" "${ROOT}/utils/random.cpp" "${ROOT}/utils/time_counter.hpp"

				"vector_operations.cpp")

target_include_directories (vec_op_benchmarks PUBLIC "${IPCL_DIR}/include" ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries (vec_op_benchmarks "${IPCL_DIR}/lib/ipcl/libipcl.so.2.0.0" ${IPPCP_LIB} ${OpenMP_CXX_LIBRARIES})