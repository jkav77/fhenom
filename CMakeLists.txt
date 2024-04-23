cmake_minimum_required (VERSION 3.28 )

project(FHEnom CXX)
set(CMAKE_CXX_STANDARD 20)
option( BUILD_STATIC "Set to ON to include static versions of the library" OFF)

find_package( spdlog CONFIG REQUIRED )

find_package( OpenFHE CONFIG HINTS ${OPENFHE_INSTALL_DIR} REQUIRED )
if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
    message(STATUS "WITH_INTEL_HEXL:     ${OpenFHE_WITH_INTEL_HEXL}")
    message(STATUS "INTEL_HEXL_HINT_DIR: ${OpenFHE_INTEL_HEXL_HINT_DIR}")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set( CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS} )
set( CMAKE_BINARY_DIR "build" )
set( CMAKE_SOURCE_DIR "src" )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin )


include_directories( include )
include_directories( SYSTEM ${OPENMP_INCLUDES} )
include_directories( SYSTEM ${OpenFHE_INCLUDE} )
include_directories( SYSTEM ${OpenFHE_INCLUDE}/third-party/include )
include_directories( SYSTEM ${OpenFHE_INCLUDE}/core )
include_directories( SYSTEM ${OpenFHE_INCLUDE}/core/include )
include_directories( SYSTEM ${OpenFHE_INCLUDE}/pke )
include_directories( SYSTEM ${OpenFHE_INCLUDE}/binfhe )
include_directories( SYSTEM /opt/homebrew/include )

if (OpenFHE_WITH_INTEL_HEXL)
    find_package(HEXL 1.2.5 HINTS ${OpenFHE_INTEL_HEXL_HINT_DIR} REQUIRED)
    if (NOT TARGET HEXL::hexl)
        message(FATAL_ERROR, "Intel HEXL not found")
    endif()

    get_target_property(INTEL_HEXL_LOCATION HEXL::hexl LOCATION)
    message(STATUS "INTEL_HEXL_LOCATION: ${INTEL_HEXL_LOCATION}")

    get_target_property(INTEL_HEXL_INCLUDE_DIR HEXL::hexl INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "INTEL_HEXL_INCLUDE_DIR: ${INTEL_HEXL_INCLUDE_DIR}")

    include_directories(${INTEL_HEXL_INCLUDE_DIR})
endif()

### add directories for other OpenFHE modules as needed for your project

link_directories( ${OpenFHE_LIBDIR} )
link_directories( ${OPENMP_LIBRARIES} )
if(BUILD_STATIC)
    set( CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries( ${OpenFHE_STATIC_LIBRARIES} )
else()
    set( CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS} )
    link_libraries( ${OpenFHE_SHARED_LIBRARIES} )
endif()

set( fhenom_src src/ckks_tensor.cpp src/ckks_vector.cpp src/context.cpp src/tensor.cpp )
set( fhenom_headers include/fhenom/ckks_tensor.hpp include/fhenom/ckks_vector.hpp include/fhenom/context.hpp include/fhenom/tensor.hpp )

add_library( fhenom SHARED ${fhenom_src} )
add_library( fhenom::fhenom ALIAS fhenom )

set_target_properties( fhenom PROPERTIES PUBLIC_HEADER "${fhenom_headers}" )

target_include_directories( fhenom PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include/fhenom>
)
target_include_directories( fhenom PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_subdirectory( tests )
