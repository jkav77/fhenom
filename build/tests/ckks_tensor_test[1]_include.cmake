if(EXISTS "/Users/dangerginger/code/FHEnom/build/tests/ckks_tensor_test[1]_tests.cmake")
  include("/Users/dangerginger/code/FHEnom/build/tests/ckks_tensor_test[1]_tests.cmake")
else()
  add_test(ckks_tensor_test_NOT_BUILT ckks_tensor_test_NOT_BUILT)
endif()
