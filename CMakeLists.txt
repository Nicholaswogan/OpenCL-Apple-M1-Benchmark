cmake_minimum_required(VERSION "3.14")
project(hello_opencl LANGUAGES C)
find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()
add_executable(main main.c)
target_link_libraries(main ${OpenCL_LIBRARIES} ${OpenMP_C_LIBRARIES})
target_include_directories(main PUBLIC ${OpenMP_C_INCLUDE_DIRS})