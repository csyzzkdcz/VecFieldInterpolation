#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

void testVecFieldSmoothingEnergyPerface(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& area, const Eigen::MatrixXd& vecField, const Eigen::VectorXd& scalarField, const int faceId, int flag = 0);

void testVecFieldSmoothingEnergy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& vecField, const Eigen::VectorXd& scalarField, int flag = 0);

double vecFieldSmoothingEnergyPerface(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& area, const Eigen::MatrixXd& vecField, const Eigen::VectorXd& scalarField, const int faceId, Eigen::VectorXd* deriv, Eigen::MatrixXd* hess, int flag = 1, bool isProj = false);

double vecFieldSmoothingEnergy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& vecField, const Eigen::VectorXd& scalarField, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, int flag = 1, bool isProj = false);

double scalarFieldSmoothingEnergy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& scalarField, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess);