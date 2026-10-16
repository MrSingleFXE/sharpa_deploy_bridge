cmake_minimum_required(VERSION 3.10)
project(sharpa_ic_tools LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(industrial_calibration REQUIRED COMPONENTS analysis target_finders_opencv)
find_package(boost_plugin_loader REQUIRED)
find_package(yaml-cpp REQUIRED)
add_executable(ic_extrinsic_cli src/ic_extrinsic_cli.cpp)
target_link_libraries(ic_extrinsic_cli PRIVATE industrial_calibration::industrial_calibration_analysis
                      industrial_calibration::industrial_calibration_target_finders_opencv
                      boost_plugin_loader::boost_plugin_loader yaml-cpp stdc++fs)
install(TARGETS ic_extrinsic_cli DESTINATION bin)
install(TARGETS ic_extrinsic_cli DESTINATION lib/${PROJECT_NAME})
