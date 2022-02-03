#ifndef FORWARD_EULER_ODE_SOLVER_HPP_
#define FORWARD_EULER_ODE_SOLVER_HPP_

#include "AbstractOdeSolver.hpp"

class ForwardEulerOdeSolver : public AbstractOdeSolver
{
public:
    void Solve() override;
};

#endif // FORWARD_EULER_ODE_SOLVER_HPP_