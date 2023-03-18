include(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.fea_cmake.txt)
cmake_minimum_required (VERSION 3.25)
project(fea_ml VERSION 1.0.0 LANGUAGES CXX)

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GoogleTest)

# User options
option(FEA_ML_TESTS "Build and run tests." On)

# Download fea_cmake and update download script if need be.
fea_cmake_download(89ce8c2fcfc0247da8c27849552aea95ebcbaf05)
fea_import_fea_cmake_download()

# Use fea clang-format file.
fea_import_clangformat()

# Use fea gitignore file.
fea_import_gitignore()

# Output to predictable location.
fea_set_output_dir(${CMAKE_BINARY_DIR}/bin)

# C++20
set(CMAKE_CXX_STANDARD 20)

# Set fea_libs options.
if (${FEA_LIBS_LOCAL})
	set(FEA_TESTS On CACHE INTERNAL "")  # Forces the value
else()
	set(FEA_TESTS Off CACHE INTERNAL "")  # Forces the value
endif()
set(FEA_CPP20 On CACHE INTERNAL "")  # Forces the value

# Pull fea_libs.
fea_fetch_content(fea_libs
	https://github.com/p-groarke/fea_libs.git
	5d2a844b878b3bb9e020f96c37e163554f4f7187
)

# Pull and find our conan dependencies.
fea_pull_conan()
find_package(TBB CONFIG REQUIRED QUIET)


# Main Lib
set(INCLUDE_NAME fea_ml)
file(GLOB_RECURSE HEADER_FILES "include/${INCLUDE_NAME}/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.hpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include PREFIX "Header Files" FILES ${HEADER_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Source Files" FILES ${SOURCE_FILES})

add_library(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
fea_set_compile_options(${PROJECT_NAME} PRIVATE)
fea_release_profiling(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PUBLIC fea_libs)
target_link_libraries(${PROJECT_NAME} PRIVATE TBB::TBB)

# Interface
target_include_directories(${PROJECT_NAME} PUBLIC
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Install Package Configuration
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}_targets)
install(EXPORT ${PROJECT_NAME}_targets
	NAMESPACE ${PROJECT_NAME}::
	FILE ${PROJECT_NAME}-config.cmake
	DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}"
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/${INCLUDE_NAME}" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")


# Tests
if (${FEA_ML_TESTS})
	# enable_testing()

	# Tests external dependencies.
	find_package(GTest CONFIG REQUIRED QUIET)

	# Test Project
	set(TEST_NAME ${PROJECT_NAME}_tests)
	file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp" "tests/*.c" "tests/*.hpp" "tests/*.h" "tests/*.tpp")
	source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/tests PREFIX "Source Files" FILES ${TEST_SOURCES})

	add_executable(${TEST_NAME} ${TEST_SOURCES})
	fea_set_compile_options(${TEST_NAME} PRIVATE)
	target_link_libraries(${TEST_NAME} PRIVATE ${PROJECT_NAME} GTest::GTest)

	gtest_discover_tests(${TEST_NAME})
	# add_dependencies(${TEST_NAME} ${PROJECT_NAME})
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TEST_NAME})

	# Copy test data on build.
	set(DATA_IN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
	set(DATA_OUT_DIR ${BINARY_OUT_DIR}/tests_data)
	fea_copy_on_build_if_different(${TEST_NAME} ${DATA_IN_DIR} ${DATA_OUT_DIR})
endif()
