include( CTest )
enable_testing()

include( FetchContent ) 
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/b479e7a3c161d7087113a05f8cb034b870313a55.zip
)
FetchContent_MakeAvailable( googletest )

add_executable( context_test context_test.cpp ../src/context.cpp )
add_executable( ckks_vector_test ckks_vector_test.cpp ../src/context.cpp ../src/ckks_vector.cpp )
add_executable( ckks_tensor_test ckks_tensor_test.cpp ../src/context.cpp ../src/ckks_vector.cpp ../src/ckks_tensor.cpp ../src/tensor.cpp )
target_link_libraries( context_test GTest::gtest_main )
target_link_libraries( ckks_vector_test GTest::gtest_main )
target_link_libraries( ckks_tensor_test GTest::gtest_main )
include( GoogleTest )
gtest_discover_tests( context_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests )
gtest_discover_tests( ckks_vector_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests )
gtest_discover_tests( ckks_tensor_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests )
