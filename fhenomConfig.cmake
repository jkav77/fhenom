include( CMakeFindDependencyMacro )
find_dependency( OpenFHE 1.1.4 )
find_dependency( spdlog 1.9.2 )

include( "${CMAKE_CURRENT_LIST_DIR}/fhenomTargets.cmake" )
