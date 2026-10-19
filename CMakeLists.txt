cmake_minimum_required(VERSION 3.30)
project(AkreonLinalg VERSION 0.1.0 LANGUAGES CXX)
option(BUILD_UNIT_TESTS "Build Catch2 unit tests" ${PROJECT_IS_TOP_LEVEL})
option(BUILD_FUNCTIONAL_TESTS "Build functional tests" ${PROJECT_IS_TOP_LEVEL})
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(akreon_linalg INTERFACE)
add_library(Akreon::Linalg ALIAS akreon_linalg)
set_target_properties(akreon_linalg PROPERTIES EXPORT_NAME Linalg)
target_compile_features(akreon_linalg INTERFACE cxx_std_26)
target_include_directories(akreon_linalg INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
add_library(akreon_linalg_test_support INTERFACE)
add_library(Akreon::LinalgTestSupport ALIAS akreon_linalg_test_support)
set_target_properties(akreon_linalg_test_support PROPERTIES EXPORT_NAME LinalgTestSupport)
target_link_libraries(akreon_linalg_test_support INTERFACE Akreon::Linalg)
target_include_directories(akreon_linalg_test_support INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test-support>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_DATADIR}/akreon-linalg/test-support>)
install(DIRECTORY test-support/ DESTINATION ${CMAKE_INSTALL_DATADIR}/akreon-linalg/test-support)

install(TARGETS akreon_linalg akreon_linalg_test_support EXPORT AkreonLinalgTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/${PROJECT_NAME})
install(EXPORT AkreonLinalgTargets NAMESPACE Akreon:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AkreonLinalg)
configure_package_config_file(cmake/AkreonLinalgConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/AkreonLinalgConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AkreonLinalg)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/AkreonLinalgConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/AkreonLinalgConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/AkreonLinalgConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AkreonLinalg)

if(PROJECT_IS_TOP_LEVEL AND (BUILD_UNIT_TESTS OR BUILD_FUNCTIONAL_TESTS))
  enable_testing()
  add_subdirectory(tests)
endif()
find_program(CLANG_FORMAT_BIN NAMES clang-format-22 clang-format)
if(PROJECT_IS_TOP_LEVEL AND CLANG_FORMAT_BIN)
  file(GLOB_RECURSE format_files CONFIGURE_DEPENDS include/*.hpp tests/*.cpp tests/*.hpp test-support/*.hpp examples/*.cpp)
  add_custom_target(clang_format_check COMMAND ${CLANG_FORMAT_BIN} --dry-run --Werror ${format_files} VERBATIM)
  add_custom_target(clang_format_fix COMMAND ${CLANG_FORMAT_BIN} -i ${format_files} VERBATIM)
endif()
