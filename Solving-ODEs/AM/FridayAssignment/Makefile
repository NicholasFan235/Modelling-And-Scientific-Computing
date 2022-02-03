all:						 TestOdeSolversRunner 
# Switch in the following line when you are ready to make a 2nd-order solver.
#all:						 TestOdeSolversRunner TestHigherOrderOdeSolverRunner


# List here all object files for classes which are needed for compiling the test
# SOLVER_OBJECTS = Exception.o AbstractOdeSolver.o
SOLVER_OBJECTS = Exception.o AbstractOdeSolver.o ForwardEulerOdeSolver.o

### The testing framework is a two-step process
# 1. Header to C++ main program via cxxtest generating script
TestOdeSolversRunner.cpp: 	TestOdeSolvers.hpp $(SOLVER_OBJECTS)
							cxxtestgen --have-eh --error-printer -o TestOdeSolversRunner.cpp TestOdeSolvers.hpp
# 2. C++ main program to executable - Then run the executable with -v "verbose trace"
TestOdeSolversRunner:		TestOdeSolversRunner.cpp
							g++ -g -o TestOdeSolversRunner TestOdeSolversRunner.cpp  $(SOLVER_OBJECTS)\
							&& ./TestOdeSolversRunner -v

### Here's the instructions for the extra test
TestHigherOrderOdeSolver.cpp: 	TestHigherOrderOdeSolver.hpp $(SOLVER_OBJECTS) HigherOrderOdeSolver.o
							cxxtestgen --have-eh --error-printer -o TestHigherOrderOdeSolver.cpp TestHigherOrderOdeSolver.hpp
TestHigherOrderOdeSolverRunner:		TestHigherOrderOdeSolver.cpp
							g++ -g -o TestHigherOrderOdeSolverRunner TestHigherOrderOdeSolver.cpp  HigherOrderOdeSolver.o $(SOLVER_OBJECTS)\
							&& ./TestHigherOrderOdeSolverRunner -v
						

### Instructions for building the classes						
Exception.o: 				Exception.cpp Exception.hpp
							g++ -g -c Exception.cpp
AbstractOdeSolver.o: 		AbstractOdeSolver.cpp AbstractOdeSolver.hpp
							g++ -g -c AbstractOdeSolver.cpp
ForwardEulerOdeSolver.o: 	ForwardEulerOdeSolver.cpp ForwardEulerOdeSolver.hpp
							g++ -g -c ForwardEulerOdeSolver.cpp
HigherOrderOdeSolver.o: 	HigherOrderOdeSolver.cpp HigherOrderOdeSolver.hpp
							g++ -g -c HigherOrderOdeSolver.cpp
clean:
				rm -f *.o
										