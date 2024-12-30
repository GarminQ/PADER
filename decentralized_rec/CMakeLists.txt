find_package ( OpenMP )

set (IPCL_DIR "/home/qjm/code/PaillierMatrix/ipcl")
set (IPPCP_LIB "/home/qjm/code/PaillierMatrix/ipcl/lib/ipcl/ippcrypto/libippcp.so.11")

set (ROOT ${CMAKE_SOURCE_DIR})

 



add_executable (protocolTest 
				"${ROOT}/decentralized_rec/test_protocol.cpp"

				# Basic Libs
				"${ROOT}/decentralized_rec/recsys.cpp" "${ROOT}/decentralized_rec/recsys.h"
				"${ROOT}/decentralized_rec/data_io/create_recsys.h" "${ROOT}/decentralized_rec/data_io/create_recsys.cpp"
				"${ROOT}/decentralized_rec/data_io/data_utils.hpp"
				"${ROOT}/decentralized_rec/evaluation/evaluation.h" "${ROOT}/decentralized_rec/evaluation/evaluation.cpp"
				"${ROOT}/decentralized_rec/optimizer.cpp" "${ROOT}/decentralized_rec/optimizer.h"

				"${ROOT}/decentralized_rec/protocol_raw/protocol.h" "${ROOT}/decentralized_rec/protocol_raw/protocol.cpp"
				"${ROOT}/decentralized_rec/protocol_paillier_1/protocol.h" "${ROOT}/decentralized_rec/protocol_paillier_1/protocol.cpp"
#				"${ROOT}/decentralized_rec/protocol_paillier_naive/protocol.h" "${ROOT}/decentralized_rec/protocol_paillier_naive/protocol.cpp"
				"${ROOT}/decentralized_rec/protocol_paillier_bipartile/protocol.h" "${ROOT}/decentralized_rec/protocol_paillier_bipartile/protocol.cpp"

				# Paillier libs
				"${ROOT}/paillier_arithmetics/arithmetics.cpp" "${ROOT}/paillier_arithmetics/arithmetics.h" 
				"${ROOT}/paillier_arithmetics/convert/convert.cpp" "${ROOT}/paillier_arithmetics/convert/convert.h" 
				"${ROOT}/utils/random.h" "${ROOT}/utils/random.cpp" "${ROOT}/utils/time_counter.hpp"
				 "evaluation/evaluation.cpp")

target_include_directories (protocolTest PUBLIC "${IPCL_DIR}/include/ipcl/" "${IPCL_DIR}/include/ipcl/ippcrypto/" ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries (protocolTest "${IPCL_DIR}/lib/ipcl/libipcl.so.2.0.0" ${IPPCP_LIB} ${OpenMP_CXX_LIBRARIES})
target_compile_options(protocolTest PRIVATE "${OpenMP_CXX_FLAGS}")


add_executable (protocolRun 
				"${ROOT}/decentralized_rec/run_protocol.cpp"

				# Basic Libs
				"${ROOT}/decentralized_rec/recsys.cpp" "${ROOT}/decentralized_rec/recsys.h"
				"${ROOT}/decentralized_rec/data_io/create_recsys.h" "${ROOT}/decentralized_rec/data_io/create_recsys.cpp"
				"${ROOT}/decentralized_rec/data_io/data_utils.hpp"
				"${ROOT}/decentralized_rec/evaluation/evaluation.h" "${ROOT}/decentralized_rec/evaluation/evaluation.cpp"
				"${ROOT}/decentralized_rec/optimizer.cpp" "${ROOT}/decentralized_rec/optimizer.h"

				"${ROOT}/decentralized_rec/protocol_raw/protocol.h" "${ROOT}/decentralized_rec/protocol_raw/protocol.cpp"
				"${ROOT}/decentralized_rec/protocol_paillier_1/protocol.h" "${ROOT}/decentralized_rec/protocol_paillier_1/protocol.cpp"
#				"${ROOT}/decentralized_rec/protocol_paillier_naive/protocol.h" "${ROOT}/decentralized_rec/protocol_paillier_naive/protocol.cpp"
				"${ROOT}/decentralized_rec/protocol_paillier_bipartile/protocol.h" "${ROOT}/decentralized_rec/protocol_paillier_bipartile/protocol.cpp"

				# Paillier libs
				"${ROOT}/paillier_arithmetics/arithmetics.cpp" "${ROOT}/paillier_arithmetics/arithmetics.h" 
				"${ROOT}/paillier_arithmetics/convert/convert.cpp" "${ROOT}/paillier_arithmetics/convert/convert.h" 
				"${ROOT}/utils/random.h" "${ROOT}/utils/random.cpp" "${ROOT}/utils/time_counter.hpp"
				"${ROOT}/utils/bignum.h" "${ROOT}/utils/bignum.cpp"
				 "evaluation/evaluation.cpp")

target_include_directories (protocolRun PUBLIC "${IPCL_DIR}/include/ipcl/" "${IPCL_DIR}/include/ipcl/ippcrypto/" ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries (protocolRun "${IPCL_DIR}/lib/ipcl/libipcl.so.2.0.0" ${IPPCP_LIB} ${OpenMP_CXX_LIBRARIES})
target_compile_options(protocolRun PRIVATE "${OpenMP_CXX_FLAGS}")


add_executable (protocolBenchmark
				"${ROOT}/decentralized_rec/benchmark.cpp"

				# Basic Libs
				"${ROOT}/decentralized_rec/recsys.cpp" "${ROOT}/decentralized_rec/recsys.h"
				"${ROOT}/decentralized_rec/data_io/create_recsys.h" "${ROOT}/decentralized_rec/data_io/create_recsys.cpp"
				"${ROOT}/decentralized_rec/data_io/data_utils.hpp"
				"${ROOT}/decentralized_rec/evaluation/evaluation.h" "${ROOT}/decentralized_rec/evaluation/evaluation.cpp"
				"${ROOT}/decentralized_rec/optimizer.cpp" "${ROOT}/decentralized_rec/optimizer.h"

				"${ROOT}/decentralized_rec/protocol_raw/protocol.h" "${ROOT}/decentralized_rec/protocol_raw/protocol.cpp"
				"${ROOT}/decentralized_rec/protocol_paillier_1/protocol.h" "${ROOT}/decentralized_rec/protocol_paillier_1/protocol.cpp"
#				"${ROOT}/decentralized_rec/protocol_paillier_naive/protocol.h" "${ROOT}/decentralized_rec/protocol_paillier_naive/protocol.cpp"
				"${ROOT}/decentralized_rec/protocol_paillier_bipartile/protocol.h" "${ROOT}/decentralized_rec/protocol_paillier_bipartile/protocol.cpp"

				# Paillier libs
				"${ROOT}/paillier_arithmetics/arithmetics.cpp" "${ROOT}/paillier_arithmetics/arithmetics.h" 
				"${ROOT}/paillier_arithmetics/convert/convert.cpp" "${ROOT}/paillier_arithmetics/convert/convert.h" 
				"${ROOT}/utils/random.h" "${ROOT}/utils/random.cpp" "${ROOT}/utils/time_counter.hpp"
				"${ROOT}/utils/bignum.h" "${ROOT}/utils/bignum.cpp"
				 "evaluation/evaluation.cpp")

target_include_directories (protocolBenchmark PUBLIC "${IPCL_DIR}/include/ipcl/" "${IPCL_DIR}/include/ipcl/ippcrypto/" ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries (protocolBenchmark "${IPCL_DIR}/lib/ipcl/libipcl.so.2.0.0" ${IPPCP_LIB} ${OpenMP_CXX_LIBRARIES})
target_compile_options(protocolBenchmark PRIVATE "${OpenMP_CXX_FLAGS}")
