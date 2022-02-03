import sys
import math
from ForwardEulerOdeSolver import ForwardEulerOdeSolver

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import unittest
from typing import List

"""
  Functions used for generating an ODE system of the form
   x' = f(x,t) where x and f are vectors
   Specifically this a two-dimensional function.
   x' = f(x,y,t) y'=g(x,y,t)
"""


def rhs_decoupled_exponential_quadratic(v:tuple, t:float) -> (float, float):
    """
     x' = -5x  gives a solution x = A exp(-5t)
     y' = 2t   gives a solution y = t^2 + c
    """
    x = v[0]
    xprime = -5.0*x
    yprime = 2.0*t
    return (xprime, yprime)

def rhs_circle(v:tuple, t:float) -> (float, float):
    """
      x' = -y
      y' = +x
      You can solve this one as: dy/dx = (dy/dt)/(dx/dt) = -x/y.  Separate and integrate to give x^2 + y^2 = 2*c
    """
    (x,y) = v
    xprime = -y
    yprime =  x
    return (xprime, yprime)


class TestOdeSolvers(unittest.TestCase):
    def test_abstract_class_methods(self):
        """ This tests all the methods of the abstract class (on a concrete derived class) """

        # Can't make one of:
        # solver = AbstractOdeSolver()
        solver = ForwardEulerOdeSolver()

        # Set initial values
        solver.SetInitialValues(7.0, 42.0);
        self.assertAlmostEqual(solver.initialValues[0], 7.0, delta=2e-16)
        self.assertAlmostEqual(solver.initialValues[1], 42.0, delta=2e-16);

        # Time set up : method 1
        solver.SetInitialTimeDeltaTimeAndFinalTime(0.0, 0.1, 1.0)
        self.assertEqual(solver.numberOfTimeSteps, 10)
        self.assertEqual(solver.startTime, 0.0)
        self.assertEqual(solver.timeStepSize, 0.1)
        # Away from zero
        solver.SetInitialTimeDeltaTimeAndFinalTime(1e15, 0.1, 1e15 + 1.0)
        self.assertEqual(solver.numberOfTimeSteps, 10)
        # Negative steps
        solver.SetInitialTimeDeltaTimeAndFinalTime(1.0, -0.1, 0.0)
        self.assertEqual(solver.numberOfTimeSteps, 10)
        self.assertEqual(solver.timeStepSize, -0.1)

        # Time step doesn't divide interval
        with self.assertRaises(ValueError):
            solver.SetInitialTimeDeltaTimeAndFinalTime(0.0, 0.10000001, 1.0)

        # Time step is negative (but interval is positive)
        with self.assertRaises(ValueError):
            solver.SetInitialTimeDeltaTimeAndFinalTime(0.0, -0.1, 1.0)

        # Time set up : method 2
        solver.SetInitialTimeNumberOfStepsAndFinalTime(0.0, 10, 1.0)
        self.assertEqual(solver.numberOfTimeSteps, 10)
        self.assertEqual(solver.startTime, 0.0)
        self.assertEqual(solver.timeStepSize, 0.1)
        # Away from zero
        solver.SetInitialTimeNumberOfStepsAndFinalTime(1e15, 10, 1e15 + 1.0)
        self.assertEqual(solver.timeStepSize, 0.1)
        # Negative steps
        solver.SetInitialTimeNumberOfStepsAndFinalTime(1.0, 10, 0.0)
        self.assertEqual(solver.numberOfTimeSteps, 10)
        self.assertEqual(solver.timeStepSize, -0.1)
        # No steps
        with self.assertRaises(ValueError):
            solver.SetInitialTimeNumberOfStepsAndFinalTime(0.0, -1, 1.0)
        # Interval is empty
        with self.assertRaises(ValueError):
            solver.SetInitialTimeNumberOfStepsAndFinalTime(0.0, 100, 0.0)

        # Test that a righthand side function can be set and used
        solver.SetRhsFunction( rhs_decoupled_exponential_quadratic ) # See top of file for definition
        v = (1.0, 100.0)
        dvdt = solver.rhsFunction(v, 7.0)
        self.assertAlmostEqual(dvdt[0], -5.0, delta = 1e-15) # -5*x
        self.assertAlmostEqual(dvdt[1], 14.0, delta = 1e-15) # 2*t
        v = (-2.0, 100.0)
        dvdt = solver.rhsFunction(v, 8.0)
        self.assertAlmostEqual(dvdt[0], 10.0, delta = 1e-15) # -5 * -2
        self.assertAlmostEqual(dvdt[1], 16.0, delta = 1e-15) # 2*t

        # Test the post-processing functions
        with self.assertRaises(Exception):
            solver.GetTimeTrace()
        with self.assertRaises(Exception):
            solver.GetXTrace()
        with self.assertRaises(Exception):
            solver.GetYTrace()
        with self.assertRaises(Exception):
            solver.DumpToFile("./tempfile.txt")
        # Fake some data
        for i in range(0, 10):
            solver.timeTrace.append( 0.1*i )
            solver.solutionTrace.append( (42.0*i,4.0) )

        # Test the data are stored properly
        times = solver.GetTimeTrace()
        self.assertEqual(len(times), 10)
        self.assertAlmostEqual(times[9], 0.9, 1e-15)
        xs = solver.GetXTrace()
        self.assertAlmostEqual(xs[9], 42*9, 1e-15)
        ys = solver.GetYTrace()
        self.assertAlmostEqual(ys[5], 4.0, 1e-15)
        # Can't write to empty file name
        with self.assertRaises(FileNotFoundError):
            solver.DumpToFile("")

        # Test of file contents is not included.  Here we check that we can write without tripping an exception
        solver.DumpToFile("./tempfile.txt")

    def test_decoupled_forward_euler(self):
        """ This is the first test which uses a Solve() method on a derived class"""
        solver = ForwardEulerOdeSolver()
        solver.SetInitialValues(10.0, 0.0) # x=10*e^{-5t} y=t^2+0
        solver.SetRhsFunction( rhs_decoupled_exponential_quadratic ) # See top of file for definition
        # A solver with no time steps should throw
        with self.assertRaises(Exception):
            solver.Solve()
        solver.SetInitialTimeDeltaTimeAndFinalTime(0.0, 0.01, 1.0)
        solver.Solve()


        times = solver.GetTimeTrace()
        x = solver.GetXTrace()
        y = solver.GetYTrace()
        # Don't forget that the time should run from 0.0 to 1.0 inclusive
        self.assertEqual(len(times), 101)
        self.assertAlmostEqual(times[100], 1.0, delta=2e-16)
        self.assertAlmostEqual(times[-1],  1.0, delta=2e-16)
        self.assertAlmostEqual(times[50], 0.5, delta=2e-16)
        self.assertAlmostEqual(times[0],   0.0, delta=2e-16)

        # Quick check of the exponential part: x=10e^{-5t}
        self.assertEqual(len(x), 101)
        self.assertAlmostEqual(x[0],   10.0,               delta=1e-15)
        self.assertAlmostEqual(x[50],  10.0*math.exp(-5.0*0.5), delta=1e-1)
        self.assertAlmostEqual(x[100], 10.0*math.exp(-5.0),     delta=1e-1)
        # Quick check of the quadratic part: y=t^2
        self.assertEqual(len(y), 101)
        self.assertAlmostEqual(y[0],   0.0,     delta=1e-15)
        self.assertAlmostEqual(y[50],  0.5*0.5, delta=1e-2)
        self.assertAlmostEqual(y[100], 1.0,     delta=1e-1)

        # Solve back in time
        solver.SetInitialTimeNumberOfStepsAndFinalTime(0.0, 99, -1.0)
        solver.Solve()
        # Check the sizes of the output are correct (now that we've run the Solve twice).
        self.assertEqual(len(solver.GetTimeTrace()), 99+1)
        self.assertAlmostEqual(solver.GetTimeTrace()[-1], -1.0,         delta=1e-15)
        self.assertAlmostEqual(solver.GetXTrace()[-1],     10*math.exp(5.0), delta=500)
        self.assertAlmostEqual(solver.GetYTrace()[-1],     1.0,         delta=1.5e-2)

        # Convergence tests
        step_size = 1.0
        for test_index in range(0, 21):
            self.assertAlmostEqual(step_size, pow(2, -1.0*test_index), delta=1e-15) # step_size is going to go like (1/2)^i
            solver.SetInitialTimeDeltaTimeAndFinalTime(0.0, step_size, 1.0)
            solver.Solve()
            # abs is absolute value function
            final_x_error = abs( solver.GetXTrace()[-1] - 10.0*math.exp(-5.0) )
            final_y_error = abs( solver.GetYTrace()[-1] - 1.0            )
            final_t =  solver.GetTimeTrace()[-1]
            # This error is exactly linear in step_size
            self.assertAlmostEqual(final_y_error, step_size, delta=1e-8)
            # Check time is not accumulating error
            self.assertAlmostEqual(final_t, 1.0, delta=5e-16) # This should be exact because step_size is a power of 2

            if step_size >= 0.4:  # Unstable for stepsize >  (2 / |5|)
                # Euler unstable: expect a large error
                self.assertLess(10.0, final_x_error)
            else:
                # step_size < 0.4) is ok for stability
                self.assertAlmostEqual(final_x_error, 0.0, delta=step_size) # Claim that answer is within step_size of analytic solution
    
            # Show that the convergence really is linear
            if test_index > 4:
                assert(final_x_error > step_size/2.0)
            step_size = step_size / 2.0

        # "Final" step size gives this:
        self.assertEqual(len(solver.GetYTrace()), 1048577) # 2^21+1 > 1 million

    def test_simple_circle(self):
        """ Try with a coupled "circle" system """
        solver = ForwardEulerOdeSolver()
        num_steps = 10000
        solver.SetInitialValues(1.0, 0.0)  # For a unit circle
        solver.SetRhsFunction( rhs_circle ) # See top of file for definition
        solver.SetInitialTimeNumberOfStepsAndFinalTime(0.0, num_steps, 2*math.pi) # 2*pi is one circuit
        solver.Solve()
        solver.DumpToFile("circle.txt")
    
        times = solver.GetTimeTrace()
        x = solver.GetXTrace()
        y = solver.GetYTrace()
    


        # Check that the time-stepping is accurate enough
        # The actual machine precision of the calculation here is about
        #     2 * pi * 2.2204e-16 = 1.395e-15
        self.assertAlmostEqual(solver.GetTimeTrace()[-1], 2.0*math.pi, delta=2e-15)
    
        # Check the solution
        for i in range(0, num_steps+1):
            # Unit radius test
            self.assertAlmostEqual( math.sqrt(x[i]*x[i] + y[i]*y[i]), 1.0, delta=2e-3)
    
            # Actual solution test
            self.assertAlmostEqual( x[i], math.cos(times[i]), delta=2e-3)
            self.assertAlmostEqual( y[i], math.sin(times[i]), delta=2e-3)


if __name__ == '__main__':
    unittest.main()
