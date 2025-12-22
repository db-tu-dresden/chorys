add_library(granforge INTERFACE)
target_include_directories(granforge INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(granforge INTERFACE
  Arrow::arrow_shared
  ArrowCompute::arrow_compute_shared
  ArrowAcero::arrow_acero_shared
  ArrowSubstrait::arrow_substrait_shared
  Parquet::parquet_shared
  spdlog::spdlog_header_only
)

add_library(granforge::granforge ALIAS granforge)
add_library(granforge::dbops ALIAS granforge)

install(TARGETS granforge EXPORT granforgeTargets)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(examples)
