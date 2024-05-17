# FHEnom
FHEnom is a phenomenal fully homomorphic encryption machine learning library based on OpenFHE. FHEnom is designed to simplify building secure machine learning and synthetic data generation models. 

> [!WARNING]
> This library is still in development. It is full of bugs and incomplete features. Use at your own risk.

# Getting Started

## Installing 

FHEnom is implemented as a CMake project and can be installed by either compiling and installing the library on your system, or using CMake's FetchContent package and including it in your project.

### Installing with find_package

#### Clone this repository:

```bash
git clone git@github.com:jkav77/FHEnom.git
```
OR
```bash
git clone https:://github.com/jkav77/FHEnom.git
```

#### Build and install FHEnom 

Run the following from the root of the repository:
    
```bash
cmake -B build .
cmake --build build -j
cmake --install build
```

Note: the install command will likely require the use of `sudo`.

#### Update your project's CMakeLists.txt

Add the following lines to your CMakeLists.txt:

```cmake
find_package(FHEnom REQUIRED)
```

### Installing with FetchContent

Add the following lines to your project's CMakeLists.txt:

```cmake
include(FetchContent)
FetchContent_Declare(
    fhenom
    GIT_REPOSITORY https://github.com/jkav77/fhenom.git
    GIT_TAG main # or dev if you want the latest, possibly broken, changes
)
FetchContent_MakeAvailable(fhenom)
```
