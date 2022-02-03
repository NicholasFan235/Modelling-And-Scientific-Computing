#include "ForwardEulerOdeSolver.hpp"

#include <iostream>
#include <vector>

void ForwardEulerOdeSolver::Solve()
{
    if (mNumberOfTimeSteps < 0) throw Exception("Time Steps Error", "Timesteps Cannot be <0");

    mTimeTrace = std::vector<double>(mNumberOfTimeSteps + 1);
    mSolutionTrace = std::vector<Pair>(mNumberOfTimeSteps + 1);
    for (int i = 0; i < mTimeTrace.size(); i++)
        mTimeTrace[i] = mStartTime + i * mTimeStepSize;

    std::cout << mInitialValues.x << ", " << mInitialValues.y << std::endl;
    std::cout << mSolutionTrace.size() << std::endl;
    std::cout << mTimeTrace.size() << std::endl;
    
    mSolutionTrace[0] = mInitialValues;

    for (int i = 0; i < mNumberOfTimeSteps; i++)
    {
        Pair delta;
        mpRhsFunction(mSolutionTrace[i], mTimeTrace[i], delta);
        mSolutionTrace[i + 1] =
            Pair{mSolutionTrace[i].x + delta.x * mTimeStepSize,
                 mSolutionTrace[i].y + delta.y * mTimeStepSize};
    }
}