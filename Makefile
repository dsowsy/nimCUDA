# Compiler
COMPILER = nvcc

# Compiler Flags
COMPILER_FLAGS = --std c++11

# Executable Output Name
EXEC_NAME = nim.exe

# Source File
SRC = nim.cu

# Phony targets for building, cleaning, and running
.PHONY: all build clean run

# Default target: build the executable
all: build

# Build target: compiles the source file
build:
	$(COMPILER) $(COMPILER_FLAGS) $(SRC) -o $(EXEC_NAME)

# Clean target: removes the executable
clean:
	rm -f $(EXEC_NAME)

# Run target: executes the program
run: build
	./$(EXEC_NAME)

