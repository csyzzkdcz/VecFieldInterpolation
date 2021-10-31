#pragma once

#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>


namespace OptSolver
{
	void gradientFlowSolver(std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc, Eigen::VectorXd& x0, int numIter = 1000, double gradTol = 1e-14, double theta = 1, bool displayInfo = false);
}


