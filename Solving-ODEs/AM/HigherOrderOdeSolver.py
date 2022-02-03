from AbstractOdeSolver import AbstractOdeSolver

class HigherOrderOdeSolver(AbstractOdeSolver):
    def Solve(self):
        # Setup Trace Arrays
        self.timeTrace = [i * self.timeStepSize + self.startTime for i in range(self.numberOfTimeSteps + 1)]
        self.solutionTrace = [None for i in range(len(self.timeTrace))]
        self.solutionTrace[0] = self.initialValues

        for i in range(1, len(self.timeTrace)):
            dx0, dy0 = self.rhsFunction(self.solutionTrace[i-1], self.timeTrace[i-1])
            dx, dy = self.rhsFunction(
                (self.solutionTrace[i-1][0] + 0.5 * self.timeStepSize * dx0,
                self.solutionTrace[i-1][1] + 0.5 * self.timeStepSize * dy0),
                self.timeTrace[i-1] + 0.5 * self.timeStepSize)
            self.solutionTrace[i] = \
                (self.solutionTrace[i-1][0] + dx * self.timeStepSize,
                self.solutionTrace[i-1][1] + dy * self.timeStepSize)
