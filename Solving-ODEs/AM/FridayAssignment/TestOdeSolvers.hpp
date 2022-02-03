#include <cxxtest/TestSuite.h>

#include "AbstractOdeSolver.hpp"
#include "ForwardEulerOdeSolver.hpp"

/**
 *
 *  Functions used for generating an ODE system of the form
 *  x' = f(x,t) where x and f are vectors
 *  Specifically this a two-dimensional function.
 *  x' = f(x,y,t) y'=g(x,y,t)
 */

/*
 * x' = -5x  gives a solution x = A exp(-5t)
 * y' = 2t   gives a solution y = t^2 + c
 */
void RhsDecoupledExponentialQuadratic(const Pair& v, double t, Pair& dvdt)
{
    dvdt.x = -5.0*v.x;
    dvdt.y = 2.0*t;
}

/*
 * x' = -y
 * y' = +x
 * You can solve this one as: dy/dx = (dy/dt)/(dx/dt) = -x/y.  Separate and integrate to give x^2 + y^2 = 2*c
 */
void RhsCircle(const Pair& v, double t, Pair& dvdt)
{
    dvdt.x = -v.y;
    dvdt.y =  v.x;
}

/**
 * For a guide to writing and running CxxTest: http://cxxtest.com/guide.html
 *
 * The Makefile in this folder is set up to compile and run this test suite.
 * There are some gentle examples of simple tests given at the top of this class.
 * Later tests in the suite are unit tests for the ODE solver(s).
 */
class TestOdeSolvers : public CxxTest::TestSuite
{
public:
    /** Every test in the test suite is a public method with a name beginning "Test..."*/
    void TestSomeExamples() //throws Exception
    {
        TS_ASSERT( 2 > -1 );        // Simple logical statement
        TS_ASSERT_EQUALS(2+2, 4);   // Equality test on whole numbers

        // 22/7 - pi ~= 1.2645e-3
        TS_ASSERT_DELTA(M_PI, 22.0/7.0, 2e-3);           // Near equality "delta" test on floating point
        //TS_ASSERT_DELTA(M_PI, 22.0/7.0, 1e-3);           // Should fail giving "found (3.1415 != 3.1428)"
    }

    /** If you want to exclude a failing test from the suite (but still have the code compiled) then just
     * change the beginning of its name to not begin "Test...".
     */
    void DoNotTestFailingExample()
    {
        // This test should never pass - but we are not running it
        TS_ASSERT_EQUALS(0, 1);
    }
    /* Note that if tests won't compile then you could also hide those tests completely*/
//    void TestWillNotCompile()
//    {
//        TS_ASSER...
//    }

    /****      Actual unit tests begin here  ******/
    /// This tests all the methods of the abstract class (on a concrete derived class)
    void TestAbstractClassMethods()
    {
        // Can't make one of: AbstractOdeSolver solver;
        ForwardEulerOdeSolver solver;

        /* Set initial values as array */
        /* Set initial values */
        solver.SetInitialValues(7.0, 42.0);
        TS_ASSERT_DELTA(solver.mInitialValues.x, 7.0, 2e-16);
        TS_ASSERT_DELTA(solver.mInitialValues.y, 42.0, 2e-16);

        /* Time set up : method 1*/
        solver.SetInitialTimeDeltaTimeAndFinalTime(0.0, 0.1, 1.0);
        TS_ASSERT_EQUALS(solver.mNumberOfTimeSteps, 10);
        TS_ASSERT_EQUALS(solver.mStartTime, 0.0);
        TS_ASSERT_EQUALS(solver.mTimeStepSize, 0.1);
        /* Away from zero */
        solver.SetInitialTimeDeltaTimeAndFinalTime(1e15, 0.1, 1e15 + 1.0);
        TS_ASSERT_EQUALS(solver.mNumberOfTimeSteps, 10);
        /* Negative steps */
        solver.SetInitialTimeDeltaTimeAndFinalTime(1.0, -0.1, 0.0);
        TS_ASSERT_EQUALS(solver.mNumberOfTimeSteps, 10);
        TS_ASSERT_EQUALS(solver.mTimeStepSize, -0.1);
        /* Time step doesn't divide interval*/
        TS_ASSERT_THROWS_ANYTHING( solver.SetInitialTimeDeltaTimeAndFinalTime(0.0, 0.10000001, 1.0) );

        /* Here's the same thing again but this time catching the exception */
        try
        {
            solver.SetInitialTimeDeltaTimeAndFinalTime(0.0, 0.10000001, 1.0);
        }
        catch (Exception& e)
        {
            std::cerr << "The following is just an illustration.  Ignore following 2 line error:\n";
            e.DebugPrint();
        }
        /* Time step is negative (but interval is positive) */
        TS_ASSERT_THROWS_ANYTHING( solver.SetInitialTimeDeltaTimeAndFinalTime(0.0, -0.1, 1.0) );

        /* Time set up : method 2*/
        solver.SetInitialTimeNumberOfStepsAndFinalTime(0.0, 10, 1.0);
        TS_ASSERT_EQUALS(solver.mNumberOfTimeSteps, 10);
        TS_ASSERT_EQUALS(solver.mStartTime, 0.0);
        TS_ASSERT_EQUALS(solver.mTimeStepSize, 0.1);
        /* Away from zero */
        solver.SetInitialTimeNumberOfStepsAndFinalTime(1e15, 10, 1e15 + 1.0);
        TS_ASSERT_EQUALS(solver.mTimeStepSize, 0.1);
        /* Negative steps */
        solver.SetInitialTimeNumberOfStepsAndFinalTime(1.0, 10, 0.0);
        TS_ASSERT_EQUALS(solver.mNumberOfTimeSteps, 10);
        TS_ASSERT_EQUALS(solver.mTimeStepSize, -0.1);
        /* No steps */
        TS_ASSERT_THROWS_ANYTHING( solver.SetInitialTimeNumberOfStepsAndFinalTime(0.0, -1, 1.0) );
        /* Interval is empty */
        TS_ASSERT_THROWS_ANYTHING( solver.SetInitialTimeNumberOfStepsAndFinalTime(0.0, 100, 0.0) );

        /* Test that a righthand side function can be set and used */
        solver.SetRhsFunction( &RhsDecoupledExponentialQuadratic ); // See top of file for definition
        Pair v = {1.0, 100.0};
        Pair dvdt;
        solver.mpRhsFunction(v, 7.0, dvdt);
        TS_ASSERT_DELTA(dvdt.x, -5.0 /* -5*x */, 1e-15);
        TS_ASSERT_DELTA(dvdt.y, 14.0 /* 2*t */  , 1e-15);
        v.x = -2.0;
        solver.mpRhsFunction(v, 8.0, dvdt);
        TS_ASSERT_DELTA(dvdt.x, 10.0 /*-5 * -2*/, 1e-15);
        TS_ASSERT_DELTA(dvdt.y, 16.0 /* 2*t */  , 1e-15);

        /* Test the post-processing functions */
        TS_ASSERT_THROWS_ANYTHING( solver.GetTimeTrace() );
        TS_ASSERT_THROWS_ANYTHING( solver.GetXTrace() );
        TS_ASSERT_THROWS_ANYTHING( solver.GetYTrace() );
        TS_ASSERT_THROWS_ANYTHING( solver.DumpToFile("./tempfile.txt") );

        /* Fake some data */
        for (int i=0; i<10; i++)
        {
            solver.mTimeTrace.push_back( 0.1*i );
            Pair temp;
            temp.x = 42.0*i;
            temp.y = 4;
            solver.mSolutionTrace.push_back( temp );
        }

        /* Test the data are stored properly */
        std::vector<double> times = solver.GetTimeTrace();
        TS_ASSERT_EQUALS(times.size(), 10);
        TS_ASSERT_DELTA(times[9], 0.9, 1e-15);
        std::vector<double> x = solver.GetXTrace();
        TS_ASSERT_DELTA(x[9], 42*9, 1e-15);
        std::vector<double> y = solver.GetYTrace();
        TS_ASSERT_DELTA(y[5], 4.0, 1e-15);
        //Can't write to empty file name
        TS_ASSERT_THROWS_ANYTHING( solver.DumpToFile("") );

        // Test of file contents is not included.  Here we check that we can write without tripping an exception
        TS_ASSERT_THROWS_NOTHING( solver.DumpToFile("./tempfile.txt") );
    }

    /** This is the first test which uses a Solve() method on a derived class */
    void TestDecoupledForwardEuler()
    {
        ForwardEulerOdeSolver solver;
        solver.SetInitialValues(10.0, 0.0);  // x=10*e^{-5t} y=t^2+0
        solver.SetRhsFunction( &RhsDecoupledExponentialQuadratic ); // See top of file for definition
        // A solver with no time steps should throw
        TS_ASSERT_THROWS_ANYTHING( solver.Solve() );
        solver.SetInitialTimeDeltaTimeAndFinalTime(0.0, 0.01, 1.0);

        solver.Solve();
        std::vector<double> times = solver.GetTimeTrace();
        std::vector<double> x = solver.GetXTrace();
        std::vector<double> y = solver.GetYTrace();
        // Don't forget that the time should run from 0.0 to 1.0 inclusive
        TS_ASSERT_EQUALS(times.size(), 101);
        TS_ASSERT_DELTA(times[0],     0.0, 1e-15);
        TS_ASSERT_DELTA(times[50],     0.5, 1e-15);
        TS_ASSERT_DELTA(times[100],    1.0, 1e-15);
        TS_ASSERT_DELTA(times.back(), 1.0, 1e-15);

        // Quick check of the exponential part: x=10e^{-5t}
        TS_ASSERT_EQUALS(x.size(), 101);
        TS_ASSERT_DELTA(x[0],   10.0, 1e-15);
        TS_ASSERT_DELTA(x[50],  10.0*exp(-5.0*0.5), 1e-1);
        TS_ASSERT_DELTA(x[100], 10.0*exp(-5.0),     1e-1);
        // Quick check of the quadratic part: y=t^2
        TS_ASSERT_EQUALS(y.size(), 101);
        TS_ASSERT_DELTA(y[0],   0.0, 1e-15);
        TS_ASSERT_DELTA(y[50],  0.5*0.5, 1e-2);
        TS_ASSERT_DELTA(y[100], 1.0,     1e-1);

        // Solve back in time
        solver.SetInitialTimeNumberOfStepsAndFinalTime(0.0, 99, -1.0);
        solver.Solve();
        // Check the sizes of the output are correct (now that we've run the Solve twice).
        TS_ASSERT_EQUALS(solver.GetTimeTrace().size(), 99+1);
        TS_ASSERT_DELTA(solver.GetTimeTrace().back(), -1.0, 1e-15);
        TS_ASSERT_DELTA(solver.GetXTrace().back(),     10*exp(5.0), 500);
        TS_ASSERT_DELTA(solver.GetYTrace().back(),     1.0,         1.5e-2);

        // Convergence tests
        double step_size = 1.0;
        for (int test_index=0; test_index<21; test_index++)
        {
            TS_ASSERT_DELTA(step_size, pow(2, -1.0*test_index), 1e-15); // step_size is going to go like (1/2)^i
            solver.SetInitialTimeDeltaTimeAndFinalTime(0.0, step_size, 1.0);
            solver.Solve();
            // fabs is the C++ absolute value function
            double final_x_error = fabs( solver.GetXTrace().back() - 10.0*exp(-5.0) );
            double final_y_error = fabs( solver.GetYTrace().back() - 1.0            );
            double final_t =  solver.GetTimeTrace().back();
            // This error is exactly linear in step_size
            TS_ASSERT_DELTA(final_y_error, step_size, 1e-8);
            // Check time is not accumulating error
            TS_ASSERT_DELTA(final_t, 1.0, 5e-16); // This should be exact because step_size is a power of 2


            if (step_size >= 0.4) // Unstable for stepsize >  (2 / |5|)
            {
                // Euler unstable: expect a large error
                TS_ASSERT_LESS_THAN(10.0, final_x_error);
            }
            else
            {
                // step_size < 0.4) is ok for stability
                TS_ASSERT_DELTA(final_x_error, 0.0, step_size); // Claim that answer is within step_size of analytic solution
            }

            // Show that the convergence really is linear
            if (test_index > 4)
            {
                TS_ASSERT(final_x_error > step_size/2.0);
            }
            step_size = step_size / 2.0;
        }
        // "Final" step size gives this:
        TS_ASSERT_EQUALS(solver.GetYTrace().size(), 1048577); // 2^21+1 > 1 million
    }

    /** Try with a coupled "circle" system  */
    void TestSimpleCircle()
    {
        ForwardEulerOdeSolver solver;
        const int num_steps = 10000;
        solver.SetInitialValues(1.0, 0.0);  // For a unit circle
        solver.SetRhsFunction( &RhsCircle ); // See top of file for definition
        solver.SetInitialTimeNumberOfStepsAndFinalTime(0.0, num_steps, 2*M_PI /* 2*pi is one circuit */);
        solver.Solve();
        solver.DumpToFile("circle.txt");

        std::vector<double> times = solver.GetTimeTrace();
        std::vector<double> x = solver.GetXTrace();
        std::vector<double> y = solver.GetYTrace();

        /*
         *
         * Check that the time-stepping is accurate enough
         * The actual machine precision of the calculation here is about
         *     2 * pi * 2.2204e-16 = 1.395e-15
         */
        TS_ASSERT_DELTA(solver.GetTimeTrace().back(), 2.0*M_PI, 2e-15);

        /*
         * Check the solution
         *
         */
        for (int i=0; i<=num_steps; i++)
        {
            // Unit radius test
            TS_ASSERT_DELTA( sqrt(x[i]*x[i] + y[i]*y[i]), 1.0, 2e-3);

            // Actual solution test
            TS_ASSERT_DELTA( x[i], cos(times[i]), 2e-3);
            TS_ASSERT_DELTA( y[i], sin(times[i]), 2e-3);
        }
    }


};
