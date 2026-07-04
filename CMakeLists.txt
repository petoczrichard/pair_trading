cmake_minimum_required(VERSION 3.18)

project(trading_core LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    trading_core
    src/bindings.cpp
)

target_include_directories(trading_core PRIVATE src)

target_compile_features(trading_core PRIVATE cxx_std_23)

if(MSVC)
    target_compile_options(trading_core PRIVATE /O2)
else()
    target_compile_options(trading_core PRIVATE -O3)
endif()

install(TARGETS trading_core LIBRARY DESTINATION .)
