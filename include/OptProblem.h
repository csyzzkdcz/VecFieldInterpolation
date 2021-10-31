#pragma once
#include "EnergyModel/energyModel.h"

class OptModel
{
public:
	OptModel(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::map<int, double> clampDOFs = {}) : _meshV(V), _meshF(F), _clampDOFs(clampDOFs)
	{
		setProjM();
	}
	double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj = false);
	void testEnergy(const Eigen::VectorXd &x);
	void convertState2Variable(const Eigen::MatrixXd& vecField, const Eigen::VectorXd& scaField, Eigen::VectorXd& x);
	void convertVariable2State(const Eigen::VectorXd& x, Eigen::MatrixXd& vecField, Eigen::VectorXd& scaField);

private:
	void setProjM();

	

	
private:
	Eigen::MatrixXd _meshV;
	Eigen::MatrixXi _meshF;
	std::map<int, double> _clampDOFs;
	Eigen::SparseMatrix<double> _projM;
public:
	Eigen::VectorXd _mass;
};