cmake_minimum_required (VERSION 3.22 )

project(fhenom VERSION 0.1 LANGUAGES CXX DESCRIPTION "Tasty FHE neural network library.")
set(CMAKE_CXX_STANDARD 20)

find_package( spdlog 1.9.2 CONFIG REQUIRED )

find_package( OpenFHE 1.1.4 CONFIG REQUIRED )
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

set( fhenom_src src/ckks_tensor.cpp src/ckks_vector.cpp src/context.cpp src/tensor.cpp )
set( fhenom_headers include/fhenom/ckks_tensor.h include/fhenom/ckks_vector.h include/fhenom/context.h include/fhenom/tensor.h include/fhenom/common.h )

add_library( ${PROJECT_NAME} SHARED ${fhenom_src} )
add_library( ${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME} )

set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${fhenom_headers}" )

target_include_directories( ${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include/fhenom>
)
target_include_directories( ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_link_libraries( ${PROJECT_NAME} PRIVATE OPENFHEcore OPENFHEpke -fopenmp spdlog::spdlog )

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}
)

include( CMakePackageConfigHelpers )
write_basic_package_version_file(
    ${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install( EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/${PROJECT_NAME}
)

configure_file( ${PROJECT_NAME}Config.cmake 
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" 
COPYONLY )

install( 
    FILES ${PROJECT_NAME}Config.cmake ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION lib/${PROJECT_NAME}
)

include( CTest )
enable_testing()
add_subdirectory( tests EXCLUDE_FROM_ALL )
