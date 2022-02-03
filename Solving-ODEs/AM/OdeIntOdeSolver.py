from AbstractOdeSolver import AbstractOdeSolver
from scipy.integrate import odeint

class OdeIntOdeSolver(AbstractOdeSolver):
    def Solve(self):
        self.timeTrace = [i * self.timeStepSize + self.startTime\
            for i in range(self.numberOfTimeSteps + 1)]
        y = odeint(self.rhsFunction, self.initialValues, self.timeTrace)
        self.solutionTrace = list(map(tuple, y))
