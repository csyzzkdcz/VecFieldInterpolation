#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "LineSearch.h"
#include "NewtonDescent.h"

namespace OptSolver
{
    /*
	* We try to solve the problem
	* min_x E(x), starting from x = x0,
	* to make sure we find the solution which is close to x0, we instead solve
	* min 1/2 ||x - x0||^2
	*   s.t. \nabla E(x) = 0
	*
	* We use the method mentioned in the paper "FEPR: Fast Energy Projection for Real-Time Simulation of Deformable Objects"
     * soling the sequential QP is:
     * min 1/2 ||x - x_k||^2 = 1/2 (x - x_k)^T M (x - x_k)
     *   s.t.  \nabla E(x_k) + HE(x_k) (x - x_k) = 0
     * This is equivalent to
     * \lambda_{k+1} = H^T M^{-1} H * \nabla E(x_k)
     * x_{k+1} = x_k - M^{-1} H \lambda_{k+1}
	*/
    void FEPR(std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc, std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> findMaxStep, const Eigen::VectorXd& prevX, const Eigen::VectorXd &M,  Eigen::VectorXd& updatedX, bool displayInfo = false){

        std::vector<Eigen::Triplet<double>> massTrip;
        Eigen::SparseMatrix<double> massMat(M.size(), M.size());

        for (int i = 0; i < M.size(); i++)
            massTrip.push_back({ i, i , M(i) });
        massMat.setFromTriplets(massTrip.begin(), massTrip.end());

        massTrip.clear();
        Eigen::SparseMatrix<double> massMatInv(M.size(), M.size());

        for (int i = 0; i < M.size(); i++)
            massTrip.push_back({ i, i , 1.0 / M(i) });
        massMatInv.setFromTriplets(massTrip.begin(), massTrip.end());


        // do the projection
        Eigen::VectorXd xk = prevX;
        double reg = 1e-8;
        for (int iter = 0; iter < 1000; iter++)
        {
            Eigen::VectorXd lambda;
            Eigen::SparseMatrix<double> H, HTMInvH;
            Eigen::VectorXd F;
            double E = objFunc(xk, &F, &H, false);
            if(F.norm() < 1e-7)
            {
                if (displayInfo)
                    std::cout << "reach the equilibrium state!" << std::endl;
                break;
            }
            std::cout << "\ninner iteration: " << iter << std::endl;
            std::cout << "grad_norm: " << F.norm() << ", H.norm: " << H.norm() << std::endl;
//            HTMInvH = H.transpose() * massMatInv * H;
            HTMInvH = H.transpose() * massMatInv * H;

            Eigen::SparseMatrix<double> I = HTMInvH, tmpH;
            I.setIdentity();
            tmpH = HTMInvH;

            Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(tmpH);

            while (solver.info() != Eigen::Success)
            {
                if (displayInfo)
                {
                    std::cout << "Matrix is not positive definite, current reg = " << reg << std::endl;
                }

                tmpH = HTMInvH + reg * I;
                solver.compute(tmpH);
                reg = std::max(2 * reg, 1e-16);
            }

            lambda = solver.solve(F);
//            xk = xk - massMatInv * H * lambda;
            xk = xk - H * lambda;
            reg = std::max(0.5 * reg, 1e-16);

        }
        updatedX = xk;
    }
}