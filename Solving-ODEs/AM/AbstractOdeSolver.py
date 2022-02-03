#!/usr/bin/env python3

"""AbstractOdeSolve.py: Abstract class (Python class with no implementation) for Initial Value Problem 2D ODEs"""

__author__ = "Joe Pitt-Francis"
__copyright__ = "Copyright 2019"

from abc import ABC, abstractmethod
# This is an abstract class but there is only one abstract (un-implemented) method: solve.
# All other methods are common to the derived classes and are given implementations.
class AbstractOdeSolver(ABC):
    """
    Locate variables are
    * Time-step or dt or delta t
    timeStepSize: float
    * Initial time
    startTime: float
    (We don't explicitly store the final time)
    * Number of time steps (of size dt) needed to get us to the final time
    numberOfTimeSteps: int
    * Righthand side function
    rhsFunction
    * Time value for each time-point - calculated during solve
    timeTrace: list of floats
    * Solution pair value for each time-point - calculated during solve
    solutionTrace: list of tuples
    """

    # Constructor
    def __init__(self):
        self.initialValues: tuple = (0.0, 0.0)
        self.rhsFunction = None
        self.numberOfTimeSteps: int = -1
        self.solutionTrace=[]
        self.timeTrace=[]

    def CheckSolution(self):
        """Helper method which is used internally to verify that the solution is ready"""
        if len(self.solutionTrace) == 0:
            raise Exception("There no solution.  Please run the Solve() method")
        # More serious error:
        assert len(self.solutionTrace) == len(self.timeTrace)
        # Last line trips if solution and time vectors have different size.  The Solve() method should update both


    def SetInitialValues(self, x: float, y: float):
        """
        Initial conditions
        :param x: x_0
        :param y: y_0
        """
        self.initialValues = (x, y)


    def SetInitialTimeDeltaTimeAndFinalTime(self, starttime: float, delta: float, endtime: float):
        """
        Set the bounds on the times.
        The time interval (and the time-step delta) can be negative.
        Throws if we can't get to the end time in a whole number of time-steps
        :param starttime: start time
        :param delta: time-step which can be negative
        :param endtime: end time
        """
        approximate_number_of_steps = (endtime - starttime)/delta
        num_steps : int = round(approximate_number_of_steps)
        # Check that we are stepping in the correct direction
        if num_steps <= 0:
            raise ValueError("Time step has the wrong sign")

        # Check that constant time step divides the interval
        expected_end_time_difference = starttime + delta*num_steps - endtime
        if expected_end_time_difference*expected_end_time_difference > 1e-15 :
            raise ValueError("Time step does not divide the time interval")

        # Set member variables
        self.startTime = starttime
        self.timeStepSize = delta
        self.numberOfTimeSteps = num_steps

    def SetInitialTimeNumberOfStepsAndFinalTime(self, starttime: float, steps: int, endtime: float):
        """
        Set the bounds on the times.
        The time interval (and the time-step delta) can be negative.
        Throws if there's no time-interval or steps is not positive
        :param starttime: start time
        :param steps: number of time steps to take (must be strictly positive)
        :param endtime: end time
        """
        # Check that we are stepping in the correct direction
        if steps <= 0:
            raise ValueError("Number of time steps should be positive")

        time_step = (endtime-starttime)/steps
        if time_step == 0:
            raise ValueError("Time interval is empty")
        # Set member variables
        self.startTime = starttime
        self.timeStepSize = time_step
        self.numberOfTimeSteps = steps

    def SetRhsFunction(self, functionName):
        """
        Set the righthand side (dx/dt) function
        :param functionName: the function to plug in
        """
        self.rhsFunction = functionName


    # Post-processing method : get out
    def GetTimeTrace(self)->list:
        """
        Post-processing method : get out cached time trace
        :return: a list of all times including startTime and endTime
        """
        # Sanity checkn
        self.CheckSolution()
        # Copy the values into a new vector
        return self.timeTrace.copy()

    def GetXTrace(self)->list:
        """
        Post-processing method : get out all calculated x-values
        :return: a list of all calculated x-values from the solution trace
        """
        # Sanity check
        self.CheckSolution()
        # Slice out the values
        xs,ys = zip(*self.solutionTrace)
        return list(xs)

    def GetYTrace(self)->list:
        """
        Post-processing method : get out all calculated y-values
        :return: a list of all calculated y-values from the solution trace
        """
        # Sanity check
        self.CheckSolution()
        # Slice out values
        xs,ys = zip(*self.solutionTrace)
        return list(ys)

    def DumpToFile(self, fileName: str):
        """
        Post-processing method : dump to named file or path
        This is column data:
        time     x-value     y-value
        :param fileName: the name of the file on disk
        """
        # Sanity check
        self.CheckSolution()
        f = open(fileName, "w")
        #if (write_output.is_open() == false)
        #    throw Exception("OdePost", "Can't open output file");
        #write_output.precision(10);
        for i in range(0, len(self.solutionTrace)):
            f.write("{:.10}\t{:.10}\t{}\n".format(
                self.timeTrace[i], self.solutionTrace[i][0], self.solutionTrace[i][1]))
        f.close()


    @abstractmethod
    def Solve(self):
        """
        Here is the method which makes this class "abstract".  It's not implemented here: only in the
        derived class/classes so we can't actually make an instance of this base class.
        """
        pass
