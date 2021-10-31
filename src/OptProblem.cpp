#include "../include/OptProblem.h"
#include <iostream>
#include <igl/doublearea.h>

void OptModel::setProjM()
{
	int row = 0;
	int nverts = _meshV.rows();
	int DOFs = 3 * nverts;

	Eigen::VectorXd faceAreas;
	igl::doublearea(_meshV, _meshF, faceAreas);

	std::vector<double> usefulMass;

	std::vector<Eigen::Triplet<double>> T;
	for (int i = 0; i < DOFs; i++)
	{
		if (_clampDOFs.find(i) != _clampDOFs.end())
			continue;
		T.push_back({ row, i, 1.0 });
		double mass = faceAreas(i / 3);
		usefulMass.push_back(mass);
		row++;
	}
	_projM.resize(row, DOFs);
	_projM.setFromTriplets(T.begin(), T.end());

	_mass.setZero(row);
	for(int i = 0; i < row; i++)
	    _mass(i) = usefulMass[i];
	std::cout << "project matrix setup done!" << std::endl;
}

void OptModel::convertState2Variable(const Eigen::MatrixXd& vecField, const Eigen::VectorXd& scaField, Eigen::VectorXd& x)
{
	int nverts = _meshV.rows();
	int DOFs = 3 * nverts;
	Eigen::VectorXd fullX(DOFs);

	for (int i = 0; i < nverts; i++)
	{
		fullX(i) = scaField(i);
		fullX(nverts + 2 * i) = vecField(i, 0);
		fullX(nverts + 2 * i + 1) = vecField(i, 1);
	}
	x = _projM * fullX;
}

void OptModel::convertVariable2State(const Eigen::VectorXd& x, Eigen::MatrixXd& vecField, Eigen::VectorXd& scaField)
{
	int nverts = _meshV.rows();
	scaField.setZero(nverts);
	vecField.setZero(nverts, 2);
	Eigen::VectorXd fullX = _projM.transpose() * x;
	
	for (auto& it : _clampDOFs)
	{
		fullX(it.first) = it.second;
	}
	
	for (int i = 0; i < nverts; i++)
	{
		scaField(i) = fullX(i);
		vecField(i, 0) = fullX(nverts + 2 * i);
		vecField(i, 1) = fullX(nverts + 2 * i + 1);
	}
}

double OptModel::computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
	int nverts = _meshV.rows();
	int DOFs = 3 * nverts;

	Eigen::MatrixXd vecField;
	Eigen::VectorXd scaField;

	convertVariable2State(x, vecField, scaField);
	
	Eigen::VectorXd grad1, grad2;
	Eigen::SparseMatrix<double> hess1, hess2;
	double energy = vecFieldSmoothingEnergy(_meshV, _meshF, vecField, scaField, deriv ? &grad1 : NULL, hess ? &hess1 : NULL, 1, isProj);
	double scaEnergy = scalarFieldSmoothingEnergy(_meshV, _meshF, scaField, deriv ? &grad2 : NULL, hess ? &hess2 : NULL);

	if (deriv)
	{
		grad1.segment(0, nverts) += grad2;
		(*deriv) = _projM * grad1;
	}

	if (hess)
	{
		std::vector<Eigen::Triplet<double>> T;
		for (int k = 0; k < hess2.outerSize(); ++k) {
			for (Eigen::SparseMatrix<double>::InnerIterator it(hess2, k); it; ++it)
				T.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
		}

		hess2.resize(DOFs, DOFs);
		hess2.setFromTriplets(T.begin(), T.end());

		hess1 = hess1 + hess2;
		(*hess) = _projM * hess1 * _projM.transpose();
	}

	return energy + scaEnergy;
}


void OptModel::testEnergy(const Eigen::VectorXd& x)
{
	Eigen::VectorXd deriv;
	Eigen::SparseMatrix<double> hess;

	double energy = computeEnergy(x, &deriv, &hess);

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	Eigen::VectorXd xPert = x;

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		xPert = x + eps * dir;

		Eigen::VectorXd deriv1;
		double energy1 = computeEnergy(xPert, &deriv1, NULL);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "energy-derivative: " << std::endl;
		std::cout << "finite difference: " << (energy1 - energy) / eps << ", directional derivative: " << deriv.dot(dir) << ", error: " << std::abs(deriv.dot(dir) - (energy1 - energy) / eps) << std::endl;

		std::cout << "derivative-hessian: " << std::endl;
		std::cout << "finite difference: " << ((deriv1 - deriv) / eps).norm() << ", directional derivative: " << (hess * dir).norm() << ". error: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}