include( FetchContent ) 
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG f8d7d77c06936315286eb55f8de22cd23c188571 
)
FetchContent_MakeAvailable( googletest )
set( INSTALL_GTEST OFF )

add_executable( context_test context_test.cpp ../src/context.cpp )
add_executable( ckks_vector_test ckks_vector_test.cpp ../src/context.cpp ../src/ckks_vector.cpp )
add_executable( ckks_tensor_test ckks_tensor_test.cpp ../src/context.cpp ../src/ckks_vector.cpp ../src/ckks_tensor.cpp ../src/tensor.cpp )
add_executable( tensor_test tensor_test.cpp ../src/tensor.cpp )
add_custom_target( tests DEPENDS context_test ckks_vector_test ckks_tensor_test tensor_test )

target_link_libraries( context_test GTest::gtest_main spdlog::spdlog     ${OpenFHE_SHARED_LIBRARIES} )
target_link_libraries( ckks_vector_test GTest::gtest_main spdlog::spdlog ${OpenFHE_SHARED_LIBRARIES} )
target_link_libraries( ckks_tensor_test GTest::gtest_main spdlog::spdlog ${OpenFHE_SHARED_LIBRARIES} )
target_link_libraries( tensor_test GTest::gtest_main spdlog::spdlog      ${OpenFHE_SHARED_LIBRARIES} )

include( GoogleTest )
gtest_discover_tests( context_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests )
gtest_discover_tests( ckks_vector_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests )
gtest_discover_tests( ckks_tensor_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests )
gtest_discover_tests( tensor_test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests )

