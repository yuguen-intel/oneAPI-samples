# Direct CMake to use icpx rather than the default C++ compiler/linker on Linux
# and icx-cl on Windows
if(UNIX)
    set(CMAKE_CXX_COMPILER icpx)
else() # Windows
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()

cmake_minimum_required (VERSION 3.7.2)

project(LoopUnroll CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(SOURCE_FILE src/loop_unroll.cpp)
set(TARGET_NAME loop_unroll)

set(EMULATOR_TARGET fpga_emu)
set(SIMULATOR_TARGET fpga_sim)
set(REPORT_TARGET report)
set(FPGA_TARGET fpga)

set(EMULATOR_OUTPUT_NAME ${TARGET_NAME}.${EMULATOR_TARGET})
set(SIMULATOR_OUTPUT_NAME ${TARGET_NAME}.${SIMULATOR_TARGET})
set(REPORT_OUTPUT_NAME ${TARGET_NAME}.${REPORT_TARGET})
set(FPGA_OUTPUT_NAME ${TARGET_NAME}.${FPGA_TARGET})

# FPGA board selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "intel_a10gx_pac:pac_a10")
    message(STATUS "FPGA_DEVICE was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_DEVICE} (Intel(R) PAC with Intel Arria(R) 10 GX FPGA). \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_DEVICE}")
endif()

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()

set(COMMON_COMPILE_FLAGS -fsycl -fintelfpga -Wall ${WIN_FLAG})
set(COMMON_LINK_FLAGS -fsycl -fintelfpga)

# A SYCL ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS -DFPGA_EMULATOR)
set(EMULATOR_LINK_FLAGS )
set(SIMULATOR_COMPILE_FLAGS -Xssimulation -DFPGA_SIMULATOR)
set(SIMULATOR_LINK_FLAGS -Xssimulation -Xsghdl -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS})
set(REPORT_COMPILE_FLAGS )
set(REPORT_LINK_FLAGS -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} -fsycl-link=early)
set(HARDWARE_COMPILE_FLAGS )
set(HARDWARE_LINK_FLAGS -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_OUTPUT_NAME})
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA simulator compilation and backend compilation

add_subdirectory(../../../include ${CMAKE_CURRENT_BINARY_DIR}/includes) 

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    icpx -fsycl -fintelfpga -DFPGA_EMULATOR loop_unroll.cpp -o loop_unroll.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -fintelfpga -DFPGA_EMULATOR -o loop_unroll.cpp.o -c loop_unroll.cpp
#    [link]    icpx -fsycl -fintelfpga loop_unroll.cpp.o -o loop_unroll.fpga_emu
add_executable(${EMULATOR_TARGET} ${SOURCE_FILE})
target_compile_options(${EMULATOR_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${EMULATOR_TARGET} PRIVATE ${EMULATOR_COMPILE_FLAGS})
target_link_libraries(${EMULATOR_TARGET} ${COMMON_LINK_FLAGS})
target_link_libraries(${EMULATOR_TARGET} ${EMULATOR_LINK_FLAGS})
target_link_libraries(${EMULATOR_TARGET} fpga_samples_shared_hdr)
set_target_properties(${EMULATOR_TARGET} PROPERTIES OUTPUT_NAME ${EMULATOR_OUTPUT_NAME})

###############################################################################
### FPGA Simulator
###############################################################################
# To compile in a single command:
#    icpx -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> -DFPGA_SIMULATOR loop_unroll.cpp -o loop_unroll.fpga_sim
# CMake executes:
#    [compile] icpx -fintelfpga -Xssimulation -DFPGA_SIMULATOR -o loop_unroll.cpp.o -c loop_unroll.cpp
#    [link]    icpx -fintelfpga -Xssimulation -Xsghdl -Xstarget=<FPGA_DEVICE> loop_unroll.cpp.o -o loop_unroll.fpga_sim
add_executable(${SIMULATOR_TARGET} ${SOURCE_FILE})
target_compile_options(${SIMULATOR_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${SIMULATOR_TARGET} PRIVATE ${SIMULATOR_COMPILE_FLAGS})
target_link_libraries(${SIMULATOR_TARGET} ${COMMON_LINK_FLAGS})
target_link_libraries(${SIMULATOR_TARGET} ${SIMULATOR_LINK_FLAGS})
target_link_libraries(${SIMULATOR_TARGET} fpga_samples_shared_hdr)
set_target_properties(${SIMULATOR_TARGET} PROPERTIES OUTPUT_NAME ${SIMULATOR_OUTPUT_NAME})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early loop_unroll.cpp -o loop_unroll.report
# The compile output is not an executable, but an intermediate compilation result unique to SYCL.
add_executable(${REPORT_TARGET} ${SOURCE_FILE})
target_compile_options(${REPORT_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${REPORT_TARGET} PRIVATE ${HARDWARE_COMPILE_FLAGS})
target_link_libraries(${REPORT_TARGET} ${COMMON_LINK_FLAGS})
target_link_libraries(${REPORT_TARGET} ${HARDWARE_LINK_FLAGS})
target_link_libraries(${REPORT_TARGET} fpga_samples_shared_hdr)
set_target_properties(${REPORT_TARGET} PROPERTIES OUTPUT_NAME ${REPORT_OUTPUT_NAME})
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> loop_unroll.cpp -o loop_unroll.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o loop_unroll.cpp.o -c loop_unroll.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> loop_unroll.cpp.o -o loop_unroll.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
target_compile_options(${FPGA_TARGET} PRIVATE ${COMMON_COMPILE_FLAGS})
target_compile_options(${FPGA_TARGET} PRIVATE ${HARDWARE_COMPILE_FLAGS})
target_link_libraries(${FPGA_TARGET} ${COMMON_LINK_FLAGS})
target_link_libraries(${FPGA_TARGET} ${HARDWARE_LINK_FLAGS})
target_link_libraries(${FPGA_TARGET} fpga_samples_shared_hdr)
set_target_properties(${FPGA_TARGET} PROPERTIES OUTPUT_NAME ${FPGA_OUTPUT_NAME})
# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.
