import sys
import math
from AbstractOdeSolver import AbstractOdeSolver
from ForwardEulerOdeSolver import ForwardEulerOdeSolver
from HigherOrderOdeSolver import HigherOrderOdeSolver

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

class TestHigherOrderOdeSolver(unittest.TestCase):
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
        hi_solver = HigherOrderOdeSolver()

        print('#step_size \t\t euler_error \t\t hi_error')
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

            # When the step is not too coarse then we can actually bound the errors
            # (There are some constant factors here: Euler settles to error~1.8138*dt; RK2 to error~0.605*dt)
            if step_size < 0.05:
                # Euler has linear convergence
                self.assertAlmostEqual(euler_error, 0.0, delta=2.0*step_size)
                # Runge-Kutta RK2 has quadratic convergence.  Other methods will do better.
                self.assertAlmostEqual(hi_error,    0.0, delta=step_size*step_size)
            # Cache a couple of interesting points to get an idea of the order of convergence
            if num_steps == 8:
                step_size1 = step_size
                euler_error1 = euler_error
                hi_error1 = hi_error
            if num_steps == 2048:
                step_size2 = step_size
                euler_error2 = euler_error
                hi_error2 = hi_error
            num_steps *= 2

        # Report on the "slope of the convergence graph"
        euler_order = round(math.log(euler_error1/euler_error2)/math.log(step_size1/step_size2))
        hi_order = round(math.log(hi_error1/hi_error2)/math.log(step_size1/step_size2))
        print('Order of converge of Forward Euler solver is ', euler_order)
        print('Order of converge of higher-order solver is ', hi_order)
        self.assertEqual(euler_order, 1)
        self.assertGreaterEqual(hi_order, 2)

if __name__ == '__main__':
    unittest.main()
