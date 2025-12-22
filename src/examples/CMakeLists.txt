add_executable(granforge_demo
    granforge_demo.cpp
)
target_include_directories(granforge_demo PRIVATE ${Protobuf_INCLUDE_DIRS})
#if debug use - O0 -g else use -O3
target_compile_options(granforge_demo PRIVATE 
    $<$<CONFIG:Debug>:-O0 -g -march=native>
    $<$<CONFIG:Release>:-O3 -march=native>
)
target_compile_options(granforge_demo PRIVATE 
    -Wall -Wextra -pedantic -mavx2 -flax-vector-conversions)
target_link_libraries(granforge_demo 
    PRIVATE 
    granforge::dbops
    nlohmann_json::nlohmann_json 
    Arrow::arrow_shared
    ArrowCompute::arrow_compute_shared
    ArrowAcero::arrow_acero_shared
    ArrowSubstrait::arrow_substrait_shared
    Parquet::parquet_shared
    Threads::Threads 
    ${Protobuf_LIBRARIES}
    spdlog::spdlog_header_only

)
