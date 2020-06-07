cmake_minimum_required(VERSION 3.10)

# Set project
project(Mala VERSION 1.0)

# Specify standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")

configure_file(MalaConfig.h.in MalaConfig.h)

#subdir

#target
add_executable(Mala main.cpp mala.cpp mala_more.cpp mala_within_gibbs.cpp )

target_include_directories(Mala PUBLIC
				${PROJECT_BINARY_DIR}
			  )
