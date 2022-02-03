/*
 * AbstractOdeSolver.hpp
 *
 *  Created on: 19 Jan 2016
 *      Author: jmpf
 */


#include <string>
#include <cmath>
#include <vector> // STL container for arbitrary length traces
#include "Exception.hpp"

#ifndef ABSTRACTODESOLVER_HPP_
#define ABSTRACTODESOLVER_HPP_

/**
 * Pair is a "struct".  This is just like a class but with just public data items.
 *
 * This is because it's arguably easier to read
 *          dvdt.x = f(v.x, v.y, t)
 * than
 *          dxdt[0] = f(x[0], x[1], t)
 */
struct
Pair
{
    double x; ///< named x variable for convenience
    double y; ///< y
};

/**
 * AbstractOdeSolver does all the administration work of setting up a 2-D initial value problem ODE.
 *  * Sets up initial conditions
 *  * Sets handle to a "right-hand side" function
 *  * Sets time stepper range
 *  * Provides storage and access for output
 *  * DOES NOT solve the ODE
 */
class AbstractOdeSolver
{
private:
    void CheckSolution() const;

protected:
    /** Initial conditions for system at start time */
     Pair mInitialValues;

    /** Time-step or dt or delta t */
    double mTimeStepSize;
    /** Initial time */
    double mStartTime;
    // We don't explicitly store the final time: double mFinalTime
    /** Number of time steps (of size dt) needed to get us to the final time */
    int mNumberOfTimeSteps;

    /** Righthand side function pointer */
    void (* mpRhsFunction)(const Pair&, double, Pair&);

    /** Time value for each time-point - calculated during solve */
    std::vector<double> mTimeTrace;
    /** Solution pair value for each time-point - calculated during solve */
    std::vector<Pair> mSolutionTrace;

public:
    /** Default constructor - makes sure that things are initialised to unset values */
    AbstractOdeSolver();

    /** Allow testing class to see some internals **/
    friend class TestOdeSolvers;

    /** Initial conditions */
    void SetInitialValues(double x, double y);

    /** Set the bounds on the times.
     *  The time interval (and the time-step delta) can be negative.
     *  Throws if we can't get to the end time in a whole number of time-steps
     */
    void SetInitialTimeDeltaTimeAndFinalTime(double startTime, double delta, double endTime);

    /** Set the bounds on the times.
     *  The time interval (and the time-step delta) can be negative.
     *  Throws if there's no time-interval or steps is not positive
     */
    void SetInitialTimeNumberOfStepsAndFinalTime(double startTime, int steps, double endTime);

    /**
     * Set the righthand side (dx/dt) function
     */
    void SetRhsFunction(void (*pFunctionName)(const Pair&, double, Pair&) );


    /**
     * Post-processing method : get out cached time trace
     */
    std::vector<double> GetTimeTrace() const;

    /**
     * Post-processing method : get out all calculated x-values
     */
    std::vector<double> GetXTrace();

    /**
     * Post-processing method : get out all calculated y-values
     */
    std::vector<double> GetYTrace();

    /**
     * Post-processing method : dump to named file or path
     *
     * This is column data:
     * time     x-value     y-value
     */
    void DumpToFile(const std::string& fileName);
    /**
     * Here is the method which makes this class "abstract".  It's not implemented here: only in the
     * derived class/classes so we can't actually make an instance of this base class.
     */
    virtual void Solve() = 0;
};

#endif /* ABSTRACTODESOLVER_HPP_ */
