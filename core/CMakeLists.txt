cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(TrackPilotCore)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Find optional packages for HTTP server
find_package(Boost COMPONENTS system thread)
if(Boost_FOUND)
    find_library(HTTPLIB_LIBRARY NAMES httplib)
    if(HTTPLIB_LIBRARY)
        set(HTTP_SERVER_AVAILABLE ON)
        add_definitions(-DHTTP_SERVER_AVAILABLE)
    endif()
endif()

# Add subdirectories
add_subdirectory(cmake)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(CORE_SOURCES
    src/main.cpp
    src/scheduling_engine.cpp
    src/rule_checker.cpp
    src/ai_model.cpp
    src/api_server.cpp
    src/logger.cpp
    src/config.cpp
)

# Header files
set(CORE_HEADERS
    include/scheduling_engine.h
    include/rule_checker.h
    include/ai_model.h
    include/api_server.h
    include/logger.h
    include/config.h
    include/types.h
)

# Create executable
add_executable(trackpilot_core ${CORE_SOURCES} ${CORE_HEADERS})

# Link libraries
target_link_libraries(trackpilot_core ${TORCH_LIBRARIES})
target_link_libraries(trackpilot_core Threads::Threads)

if(HTTP_SERVER_AVAILABLE)
    target_link_libraries(trackpilot_core ${Boost_LIBRARIES})
endif()

# Compiler-specific options
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET trackpilot_core
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:trackpilot_core>)
endif (MSVC)

# Install targets
install(TARGETS trackpilot_core DESTINATION bin)

# Testing
enable_testing()
add_subdirectory(tests)

# Build configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(trackpilot_core PRIVATE DEBUG_BUILD)
    set_property(TARGET trackpilot_core PROPERTY
        CXX_FLAGS "${CXX_FLAGS} -g -O0 -Wall -Wextra")
else()
    set_property(TARGET trackpilot_core PROPERTY
        CXX_FLAGS "${CXX_FLAGS} -O3 -DNDEBUG")
endif()