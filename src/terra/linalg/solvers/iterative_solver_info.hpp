
#pragma once

namespace terra::linalg::solvers {

class IterativeSolverParameters
{
  public:
    IterativeSolverParameters(
        int    max_iterations,
        double relative_residual_tolerance,
        double absolute_residual_tolerance )
    : max_iterations_( max_iterations )
    , relative_residual_tolerance_( relative_residual_tolerance )
    , absolute_residual_tolerance_( absolute_residual_tolerance )
    {}

    int    max_iterations() const { return max_iterations_; }
    double relative_residual_tolerance() const { return relative_residual_tolerance_; }
    double absolute_residual_tolerance() const { return absolute_residual_tolerance_; }

  private:
    int    max_iterations_;
    double relative_residual_tolerance_;
    double absolute_residual_tolerance_;
};

class IterativeSolverStatistics
{};

} // namespace terra::linalg::solvers