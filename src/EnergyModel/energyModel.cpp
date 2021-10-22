#pragma once
#include "../../include/EnergyModel/energyModel.h"
#include "../../include/diffOp/DiscreteDiffOp.h"
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <vector>
#include <iostream>

Eigen::MatrixXd lowRankApprox(Eigen::MatrixXd A)
{
	Eigen::MatrixXd posHess = A;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
	es.compute(posHess);
	Eigen::VectorXd evals = es.eigenvalues();

	for (int i = 0; i < evals.size(); i++)
	{
		if (evals(i) < 0)
			evals(i) = 0;
	}
	Eigen::MatrixXd D = evals.asDiagonal();
	Eigen::MatrixXd V = es.eigenvectors();
	posHess = V * D * V.inverse();

	return posHess;
}

double vecFieldSmoothingEnergyPerface(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& area, const Eigen::MatrixXd& vecField, const Eigen::VectorXd& scalarField, const int faceId, Eigen::VectorXd* deriv, Eigen::MatrixXd* hess, int flag, bool isProj)
{
	Eigen::VectorXd faceCurlDeriv, faceDivDeriv;
	Eigen::MatrixXd faceCurlHess, faceDivHess;
	double curl = DiscreteDiffOp::discreteCurl(V, F, area, vecField, faceId, (deriv || hess) ? &faceCurlDeriv : NULL, hess ? &faceCurlHess : NULL);

	double div = DiscreteDiffOp::discreteDiv(V, F, area, vecField, faceId, (deriv || hess) ? &faceDivDeriv : NULL, hess ? &faceDivHess : NULL);

	int dim = F.cols();
	if (flag == 0)
	{
		double energy = 0.5 * (curl * curl + div * div) * area(faceId);

		if (deriv)
		{
			deriv->setZero(dim * 3);

			for (int i = 0; i < dim; i++)
			{
				deriv->segment<2>(2 * i + dim) += (curl * faceCurlDeriv.segment<2>(2 * i) + div * faceDivDeriv.segment<2>(2 * i)) * area(faceId);
			}
		}

		if (hess)
		{
			hess->setZero(dim * 3, dim * 3);
			hess->block(dim, dim, 2 * dim, 2 * dim) = (faceCurlDeriv * faceCurlDeriv.transpose() + faceDivDeriv * faceDivDeriv.transpose() + curl * faceCurlHess + div * faceDivHess) * area;

			if (isProj)
				hess->block(dim, dim, 2 * dim, 2 * dim) = lowRankApprox(hess->block(dim, dim, 2 * dim, 2 * dim));
		}

		return energy;
	}
	else
	{
		double aveRS = 0;
		Eigen::VectorXd aveRSDeriv = Eigen::VectorXd::Zero(dim);
		Eigen::MatrixXd aveRSHess = Eigen::MatrixXd::Zero(dim, dim);
		
		for (int i = 0; i < dim; i++)
		{
			aveRS += scalarField(F(faceId, i)) * scalarField(F(faceId, i)) / dim;
			aveRSDeriv(i) = 2 * scalarField(F(faceId, i)) / dim;
			aveRSHess(i, i) = 2.0 / dim;
		}

		double energy = 0.5 * (aveRS * curl * curl + div * div) * area(faceId);

		if (deriv)
		{
			deriv->setZero(dim * 3);

			for (int i = 0; i < dim; i++)
			{
				(*deriv)(i) += 0.5 * (aveRSDeriv(i) * curl * curl) * area(faceId);

				deriv->segment<2>(2 * i + dim) += (aveRS * curl * faceCurlDeriv.segment<2>(2 * i) + div * faceDivDeriv.segment<2>(2 * i)) * area(faceId);
			}
		}

		if (hess)
		{
			hess->setZero(dim * 3, dim * 3);
			
			hess->block(0, 0, dim, dim) = 0.5 * aveRSHess * curl * curl * area(faceId);

			hess->block(0, dim, dim, 2 * dim) = aveRSDeriv * curl * faceCurlDeriv.transpose() * area(faceId);
			hess->block(dim, 0, 2 * dim, dim) = faceCurlDeriv * curl * aveRSDeriv.transpose() * area(faceId);

		
			hess->block(dim, dim, 2 * dim, 2 * dim) = (aveRS * faceCurlDeriv * faceCurlDeriv.transpose() + faceDivDeriv * faceDivDeriv.transpose() + aveRS * curl * faceCurlHess + div * faceDivHess) * area(faceId);

			if (isProj)
			{
				(*hess) = lowRankApprox(*hess);
			}
				
			
		}

		return energy;
	}


}

double vecFieldSmoothingEnergy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& vecField, const Eigen::VectorXd& scalarField, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, int flag, bool isProj)
{
	int nfaces = F.rows();
	int nverts = V.rows();
	int DOFs = 3 * nverts;

	Eigen::VectorXd faceArea;
	igl::doublearea(V, F, faceArea);
	faceArea *= 0.5;
	double energy = 0;

	if (deriv)
	{
		deriv->setZero(DOFs);
	}
	if (hess)
	{
		hess->resize(DOFs, DOFs);
	}

	std::vector<Eigen::Triplet<double>> T;

	for (int i = 0; i < nfaces; i++)
	{
		Eigen::VectorXd perfaceDeriv;
		Eigen::MatrixXd perFaceHess;

		energy += vecFieldSmoothingEnergyPerface(V, F, faceArea, vecField, scalarField, i, (deriv || hess) ? &perfaceDeriv : NULL, hess ? &perFaceHess : NULL, flag, isProj);

		if (deriv)
		{
			for (int j = 0; j < F.cols(); j++)
			{
				int vid = F(i, j);
				(*deriv)(vid) += perfaceDeriv(j);
				(*deriv)(nverts + 2 * vid) += perfaceDeriv(F.cols() + 2 * j);
				(*deriv)(nverts + 2 * vid + 1) += perfaceDeriv(F.cols() + 2 * j + 1);
			}
		}

		if (hess)
		{
			for(int j = 0; j < F.cols(); j++)
				for (int k = 0; k < F.cols(); k++)
				{
					int vid1 = F(i, j);
					int vid2 = F(i, k);

					T.push_back(Eigen::Triplet<double>(vid1, vid2, perFaceHess(j, k)));

					T.push_back(Eigen::Triplet<double>(2 * vid1 + nverts, vid2, perFaceHess(2 * j + F.cols(), k)));
					T.push_back(Eigen::Triplet<double>(2 * vid1 + nverts + 1, vid2, perFaceHess(2 * j + 1 + F.cols(), k)));

					T.push_back(Eigen::Triplet<double>(vid1, 2 * vid2 + nverts, perFaceHess(j, 2 * k + F.cols())));
					T.push_back(Eigen::Triplet<double>(vid1, 2 * vid2 + nverts + 1, perFaceHess(j, 2 * k + 1 + F.cols())));

					for(int m = 0; m < 2; m++)
						for (int n = 0; n < 2; n++)
						{
							T.push_back(Eigen::Triplet<double>(2 * vid1 + nverts + m, 2 * vid2 + nverts + n, perFaceHess(2 * j + m + F.cols(), 2 * k + n + F.cols())));
						}
					
				}
		}
	}
	if (hess)
		hess->setFromTriplets(T.begin(), T.end());
	return energy;
}

double scalarFieldSmoothingEnergy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& scalarField, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess)
{
	Eigen::SparseMatrix<double> L;
	igl::cotmatrix(V, F, L);

	double energy = -0.5 * scalarField.dot(L * scalarField);
	if (deriv)
	{
		*(deriv) = -L * scalarField;
	}
	if (hess)
	{
		*(hess) = -L;
	}
	return energy;
}

void testVecFieldSmoothingEnergyPerface(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::VectorXd& area, const Eigen::MatrixXd& vecField, const Eigen::VectorXd& scalarField, const int faceId, int flag)
{
	int dim = F.cols();
	Eigen::VectorXd deriv;
	Eigen::MatrixXd hess;

	double energy = vecFieldSmoothingEnergyPerface(V, F, area, vecField, scalarField, faceId, &deriv, &hess, flag);

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	auto newVec = vecField;
	auto newSca = scalarField;
	std::cout << "test face id: " << faceId << ", energy: " << energy << std::endl;
	std::cout << "deriv: " << deriv.transpose() << std::endl;
	std::cout << "hess: \n" << hess << std::endl;

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);
		
		for (int j = 0; j < dim; j++)
		{
			int vid = F(faceId, j);
			newSca(vid) = scalarField(vid) + dir(j) * eps;
			newVec(vid, 0) = vecField(vid, 0) + dir(dim + 2 * j) * eps;
			newVec(vid, 1) = vecField(vid, 1) + dir(dim + 2 * j + 1) * eps;
		}

		Eigen::VectorXd deriv1;
		double energy1 = vecFieldSmoothingEnergyPerface(V, F, area, newVec, newSca, faceId, &deriv1, NULL, flag);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "energy-derivative: " << std::endl;
		std::cout << "finite difference: " << (energy1 - energy) / eps << ", directional derivative: " << deriv.dot(dir) << ", error: " << std::abs(deriv.dot(dir) - (energy1 - energy) / eps) << std::endl;

		std::cout << "derivative-hessian: " << std::endl;
		std::cout << "finite difference: " << ((deriv1 - deriv) / eps).norm() << ", directional derivative: " << (hess * dir).norm() << ". error: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}


void testVecFieldSmoothingEnergy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& vecField, const Eigen::VectorXd& scalarField, int flag)
{
	int dim = F.cols();
	Eigen::VectorXd deriv;
	Eigen::SparseMatrix<double> hess;

	double energy = vecFieldSmoothingEnergy(V, F, vecField, scalarField, &deriv, &hess, flag);

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	auto newVec = vecField;
	auto newSca = scalarField;

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		for (int k = 0; k < F.rows(); k++)
		{
			for (int j = 0; j < dim; j++)
			{
				int vid = F(k, j);
				newSca(vid) = scalarField(vid) + dir(vid) * eps;
				newVec(vid, 0) = vecField(vid, 0) + dir(V.rows() + 2 * vid) * eps;
				newVec(vid, 1) = vecField(vid, 1) + dir(V.rows() + 2 * vid + 1) * eps;
			}
		}
		

		Eigen::VectorXd deriv1;
		double energy1 = vecFieldSmoothingEnergy(V, F, newVec, newSca, &deriv1, NULL, flag);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "energy-derivative: " << std::endl;
		std::cout << "finite difference: " << (energy1 - energy) / eps << ", directional derivative: " << deriv.dot(dir) << ", error: " << std::abs(deriv.dot(dir) - (energy1 - energy) / eps) << std::endl;

		std::cout << "derivative-hessian: " << std::endl;
		std::cout << "finite difference: " << ((deriv1 - deriv) / eps).norm() << ", directional derivative: " << (hess * dir).norm() << ". error: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}