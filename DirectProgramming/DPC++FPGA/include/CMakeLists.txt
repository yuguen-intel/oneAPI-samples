add_library( fpga_samples_shared_hdr INTERFACE )
target_sources( fpga_samples_shared_hdr INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/constexpr_math.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/exception_handler.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/memory_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/metaprogramming_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/onchip_memory_with_cache.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pipe_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rom_base.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/streaming_cholesky.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/streaming_cholesky_inversion.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/streaming_qrd.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/streaming_qri.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tuple.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/unrolled_loop.hpp
)
target_include_directories(fpga_samples_shared_hdr INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
