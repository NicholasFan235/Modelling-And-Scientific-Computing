import sys
import math
from AbstractOdeSolver import AbstractOdeSolver
from ForwardEulerOdeSolver import ForwardEulerOdeSolver
from OdeIntOdeSolver import OdeIntOdeSolver

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

class TestOdeIntOrderOdeSolver(unittest.TestCase):
    """This test suite is about testing a higher-order ODE solver (Runge-Kutta, Adams-Bashforth etc.)"""
    def solve_and_calculate_l2_error_for_circle(self, solver: AbstractOdeSolver):
        """
        Private helper method.
        This takes in a solver (with time-steps set), plugs in the circle function,
        solves it, then post-processes the L_2 error of the solution
        :param solver: Which solver (e.g. HigherOrderOdeSolver)
        :return: The L2-norm error
        """
        solver.SetRhsFunction(rhs_circle)
        solver.SetInitialValues(1.0, 0.0)  # For a unit circle

        solver.Solve()

        times = solver.GetTimeTrace()
        x = solver.GetXTrace()
        y = solver.GetYTrace()
        sum_square_error = 0.0
        for i in range(0, len(times)):
            sum_square_error += (x[i]-math.cos(times[i])) *  (x[i]-math.cos(times[i]))
            sum_square_error += (y[i]-math.sin(times[i])) *  (y[i]-math.sin(times[i]))
        return math.sqrt( sum_square_error/len(times) )

    def test_simple_circle_convergence(self):
        """ Convergence with a coupled "circle" system """
        # Do the same thing with both solvers simultaneously
        euler_solver = ForwardEulerOdeSolver()
        hi_solver = OdeIntOdeSolver()

        print('#step_size \t\t euler_error \t\t odeint_error')
        # Doubling loop
        num_steps = 1
        while num_steps<5e5:
            euler_solver.SetInitialTimeNumberOfStepsAndFinalTime(0.0, num_steps, 2.0*math.pi) # 2*pi is one circuit
            hi_solver.SetInitialTimeNumberOfStepsAndFinalTime(0.0, num_steps, 2.0*math.pi) # 2*pi is one circuit

            euler_error = self.solve_and_calculate_l2_error_for_circle(euler_solver) # Helper method defined above
            hi_error = self.solve_and_calculate_l2_error_for_circle(hi_solver)

            self.assertAlmostEqual(euler_solver.GetTimeTrace()[-1], 2.0*math.pi, delta=2e-15)
            self.assertAlmostEqual(hi_solver.GetTimeTrace()[-1],    2.0*math.pi, delta=2e-15)

            step_size = (2.0*math.pi)/num_steps
            print(step_size, '\t', euler_error, '\t', hi_error)
            num_steps *= 2

if __name__ == '__main__':
    unittest.main()
