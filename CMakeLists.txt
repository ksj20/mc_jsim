###############################################################################
# CMakeLists.txt — minimal yet flexible template for a small C++ project
###############################################################################
cmake_minimum_required(VERSION 3.20)   # Pick a version you know is installed
project(jtsim VERSION 1.0 LANGUAGES CXX) # abbreviation for JSON_TREE_SIMILARITY = jtsim

# ─── Build settings ──────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)             # or 20/23 as you prefer
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)          # no compiler-specific extensions

# Helpful warnings (GCC/Clang/MSVC automatically filtered)
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

# ─── Source / header lists ───────────────────────────────────────────────────
# Explicit lists are clearer and avoid accidental rebuilds on every file change
set(SRC_FILES
    src/main.cpp
    # src/foo.cpp
    # src/bar.cpp
)

set(HDR_FILES
    # include/foo.hpp
    # include/bar.hpp
)

# ─── Target(s) ───────────────────────────────────────────────────────────────
# Option A: everything goes straight into one executable
add_executable(${PROJECT_NAME} ${SRC_FILES} ${HDR_FILES})
target_include_directories(${PROJECT_NAME}
    PUBLIC               # “PUBLIC” lets downstream targets inherit the path
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Option B (uncomment if you want a reusable static/shared library)
# add_library(${PROJECT_NAME}_lib STATIC ${SRC_FILES} ${HDR_FILES})
# target_include_directories(${PROJECT_NAME}_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# add_executable(${PROJECT_NAME} src/main.cpp)
# target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)

# ─── Install rules (optional) ────────────────────────────────────────────────
# install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
# install(DIRECTORY include/ DESTINATION include)

# ─── Unit tests (optional) ───────────────────────────────────────────────────
# option(BUILD_TESTS "Build unit tests with CTest" ON)
# if(BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(test)   # put your tests in test/CMakeLists.txt
# endif()

# ─── Packaging (optional) ────────────────────────────────────────────────────
# include(CPack)            # Gives you `make package` / `cmake --build . --target package`
