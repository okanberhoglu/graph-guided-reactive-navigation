cmake_minimum_required(VERSION 3.18)

# GPU targets: Turing (75) and Ampere (86). The sm_86 binary also runs on Ada
# (8.9, e.g. RTX 40xx), and the compute_86 PTX is JIT-compiled on newer GPUs.
# CUDA 11.5 cannot target sm_89 directly. Override with
#   colcon build --cmake-args -DCMAKE_CUDA_ARCHITECTURES=<arch>
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 86)
endif()

project(robot_controller LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# host-compiler warnings only, nvcc does not accept these flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra> $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
endif()

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(messages REQUIRED)

add_executable(controller_node
  src/controller.cpp
  src/controller.cu
)

target_include_directories(controller_node PRIVATE
  include/${PROJECT_NAME}
)

target_link_libraries(controller_node CUDA::cudart)

ament_target_dependencies(controller_node
  rclcpp
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  messages
)

install(TARGETS controller_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
