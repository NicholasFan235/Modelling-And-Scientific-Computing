/*
 * AbstractOdeSolver.cpp
 *
 *  Created on: 19 Jan 2016
 *      Author: jmpf
 */
#include <fstream>
#include <cassert>
#include <iostream>
#include "AbstractOdeSolver.hpp"


AbstractOdeSolver::AbstractOdeSolver()
{
    mInitialValues.x = 0.0;
    mInitialValues.y = 0.0;
    mpRhsFunction = NULL;
    mNumberOfTimeSteps = -1;
    mTimeTrace = std::vector<double>();
    mSolutionTrace = std::vector<Pair>();
}

void AbstractOdeSolver::CheckSolution() const
{
    if (mSolutionTrace.empty())
    {
        throw Exception("OdePost", "There no solution.  Please run the Solve() method");
    }
    // More serious error:
    std::cout << "mSolutionTrace Size: " << mSolutionTrace.size() << ", mTimeTrace Size: " << mTimeTrace.size() << std::endl;
    assert (mSolutionTrace.size() == mTimeTrace.size());
    // Last line trips if solution and time vectors have different size.  The Solve() method should update both
}


void AbstractOdeSolver::SetInitialValues(double x, double y)
{
    mInitialValues.x = x;
    mInitialValues.y = y;
}

void AbstractOdeSolver::SetInitialTimeDeltaTimeAndFinalTime(double startTime, double delta, double endTime)
{
    double approximate_number_of_steps = (endTime - startTime)/delta;
    int num_steps = int ( round(approximate_number_of_steps) );

    // Check that we are stepping in the correct direction
    if (num_steps <= 0)
    {
        throw Exception("OdeSetup", "Time step has the wrong sign");
    }

    // Check that constant time step divides the interval
    double expected_end_time_difference = startTime + delta*num_steps - endTime;
    if (expected_end_time_difference*expected_end_time_difference > 1e-15)
    {
        throw Exception("OdeSetup", "Time step does not divide the time interval");
    }

    // Set member variables
    mStartTime = startTime;
    mTimeStepSize = delta;
    mNumberOfTimeSteps = num_steps;


}
void AbstractOdeSolver::SetInitialTimeNumberOfStepsAndFinalTime(double startTime, int steps, double endTime)
{
    // Check that we are stepping in the correct direction
    if (steps <= 0)
    {
        throw Exception("OdeSetup", "Number of time steps should be positive");
    }
    double time_step = (endTime-startTime)/steps;
    if (time_step == 0)
    {
        throw Exception("OdeSetup", "Time interval is empty");
    }
    // Set member variables
    mStartTime = startTime;
    mTimeStepSize = time_step;
    mNumberOfTimeSteps = steps;



}

void AbstractOdeSolver::SetRhsFunction(void (*pFunctionName)(const Pair&, double, Pair&))
{
    mpRhsFunction = pFunctionName;
}


std::vector<double> AbstractOdeSolver::GetTimeTrace() const
{
    // Sanity check
    CheckSolution();
    // Copy the values into a new vector
    return mTimeTrace;
}

std::vector<double> AbstractOdeSolver::GetXTrace()
{
    // Sanity check
    CheckSolution();
    // Copy out values
    std::vector<double> temp;
    for (int i=0; i<mSolutionTrace.size(); i++)
    {
        temp.push_back( mSolutionTrace[i].x );
    }
    return temp;
}

std::vector<double> AbstractOdeSolver::GetYTrace()
{
    // Sanity check
    CheckSolution();
    // Copy out values
    std::vector<double> temp;
    for (int i=0; i<mSolutionTrace.size(); i++)
    {
        temp.push_back( mSolutionTrace[i].y );
    }
    return temp;
}

void AbstractOdeSolver::DumpToFile(const std::string& fileName)
{
    // Sanity check
    CheckSolution();
    std::ofstream write_output(fileName.c_str());
    if (write_output.is_open() == false)
    {
        throw Exception("OdePost", "Can't open output file");
    }
    write_output.precision(10);
    for (int i=0; i<mSolutionTrace.size(); i++)
    {
        write_output << mTimeTrace[i] << "\t";
        write_output << mSolutionTrace[i].x << "\t";
        write_output << mSolutionTrace[i].y << "\n";
    }

    write_output.close();

}

