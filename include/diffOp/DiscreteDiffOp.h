#pragma once
#include <Eigen/Dense>

namespace DiscreteDiffOp
{
	using namespace Eigen;
	double discreteCurl(const MatrixXd& V, const MatrixXi& F, const VectorXd& area, const Eigen::MatrixXd& vecField, const int faceId, Eigen::VectorXd* deriv = NULL, Eigen::MatrixXd* hess = NULL);
	// compute the discrete curl of a given vector field "vecField" on a triangle mesh, using stokes theorem, the dreiv and hess are w.r.t. the vector field (in R2)

	double discreteDiv(const MatrixXd& V, const MatrixXi& F, const VectorXd& area, const Eigen::MatrixXd& vecField, const int faceId, Eigen::VectorXd* deriv = NULL, Eigen::MatrixXd* hess = NULL);
	// compute the discrete divergence of a given vector field "vecField" on a triangle mesh, using stokes theorem, the dreiv and hess are w.r.t. the vector field (in R2)
}