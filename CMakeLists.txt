cmake_minimum_required(VERSION 3.20)
project(VehicleDynamics VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Eigen3 (header-only linear algebra)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3)

# nlohmann/json (header-only JSON)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(vehicle_dynamics
    src/main.cpp
    src/path_utils.cpp
    src/car_tracking.cpp
    src/car_dynamic.cpp
    src/aircraft.cpp
    src/ship.cpp
)

target_include_directories(vehicle_dynamics PRIVATE src)

target_link_libraries(vehicle_dynamics PRIVATE
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

if(MSVC)
    target_compile_options(vehicle_dynamics PRIVATE /W4 /O2 /utf-8)
else()
    target_compile_options(vehicle_dynamics PRIVATE -Wall -Wextra -O2)
endif()
