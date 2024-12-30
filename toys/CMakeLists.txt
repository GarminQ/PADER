find_package ( OpenMP )

set (IPCL_DIR "/home/qjm/code/PaillierMatrix/ipcl")
set (IPPCP_LIB "/home/qjm/code/PaillierMatrix/ipcl/lib/ipcl/ippcrypto/libippcp.so.11")

set (ROOT ${CMAKE_SOURCE_DIR})
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable (encryption_test 
				"test_encryption.cpp")

target_include_directories (encryption_test PUBLIC "${IPCL_DIR}/include/ipcl/" "${IPCL_DIR}/include/ipcl/ippcrypto/" ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries (encryption_test "${IPCL_DIR}/lib/ipcl/libipcl.so.2.0.0" ${IPPCP_LIB} ${OpenMP_CXX_LIBRARIES})
target_compile_options(encryption_test PRIVATE "${OpenMP_CXX_FLAGS}")

message(STATUS "IPCL_DIR: ${IPCL_DIR}")
message(STATUS "Include paths: ${IPCL_DIR}/include/ipcl")
message(STATUS "Library path: ${IPCL_DIR}/lib/ipcl/libipcl.so.2.0.0")

message(STATUS "OpenMP Include Paths: ${OpenMP_CXX_INCLUDE_DIRS}")
message(STATUS "OpenMP Libraries: ${OpenMP_CXX_LIBRARIES}")

