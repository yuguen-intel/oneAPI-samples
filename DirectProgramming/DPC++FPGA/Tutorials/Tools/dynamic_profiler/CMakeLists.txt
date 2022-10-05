# Direct CMake to use icpx -fsycl rather than the default C++ compiler/linker
set(CMAKE_CXX_COMPILER icpx)
 
if(NOT UNIX)
    include (Platform/Windows-Clang)
endif()

    add_subdirectory (src)
else() # Windows
    # This code sample does not currently support windows
    message(FATAL_ERROR "Error: This code sample cannot be configured to run on Windows!\
                    \nPlease look over the README for information on using this feature on windows.")
endif()
