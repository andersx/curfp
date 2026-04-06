add_executable(test_ssfrk test_ssfrk.cu)
target_include_directories(test_ssfrk PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_ssfrk PRIVATE curfp CUDA::cudart CUDA::cublas)
add_test(NAME test_ssfrk COMMAND test_ssfrk)

add_executable(test_spftrf test_spftrf.cu)
target_include_directories(test_spftrf PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_spftrf PRIVATE curfp CUDA::cudart CUDA::cublas CUDA::cusolver)
add_test(NAME test_spftrf COMMAND test_spftrf)
