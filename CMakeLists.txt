# Minimum CMake version and project definition
cmake_minimum_required(VERSION 3.20)
project(granforgelib VERSION 0.1 LANGUAGES CXX)

# Set C++ standard to C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)


find_package(Threads REQUIRED)
find_package(Arrow REQUIRED)
find_package(ArrowCompute REQUIRED)
find_package(ArrowAcero REQUIRED)
find_package(ArrowSubstrait REQUIRED)
find_package(Parquet REQUIRED)
find_package(Protobuf REQUIRED)
find_package(spdlog REQUIRED)
find_package(Doxygen)

add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.12.0
)
FetchContent_MakeAvailable(json)

if(DOXYGEN_FOUND)
  FetchContent_Declare(
      doxygen-awesome-css
      URL https://github.com/jothepro/doxygen-awesome-css/archive/refs/tags/v2.4.1.zip
      # URL https://github.com/jothepro/doxygen-awesome-css/archive/refs/heads/main.zip
  )
  FetchContent_MakeAvailable(doxygen-awesome-css)
  FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR AWESOME_CSS_DIR)
  
  set(DOXYGEN_IN "${CMAKE_SOURCE_DIR}/Doxyfile.in")
  set(DOXYGEN_OUT "${CMAKE_BINARY_DIR}/Doxyfile")
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
else()
  message(STATUS "Doxygen not found. Documentation target 'doc' will be unavailable.")
endif()

add_subdirectory(src)

set(granforge_install_cmakedir "${CMAKE_INSTALL_LIBDIR}/cmake/granforge")

install(EXPORT granforgeTargets
  NAMESPACE granforge::
  DESTINATION "${granforge_install_cmakedir}"
)

configure_package_config_file(
  "${CMAKE_SOURCE_DIR}/cmake/granforgeConfig.cmake.in"
  "${CMAKE_BINARY_DIR}/granforgeConfig.cmake"
  INSTALL_DESTINATION "${granforge_install_cmakedir}"
)

write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/granforgeConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_BINARY_DIR}/granforgeConfig.cmake"
  "${CMAKE_BINARY_DIR}/granforgeConfigVersion.cmake"
  DESTINATION "${granforge_install_cmakedir}"
)
