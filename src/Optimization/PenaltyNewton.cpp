
#include "../../include/Optimization/LineSearch.h"
#include "../../include/Optimization/PenaltyNewton.h"
#include "../../include/Optimization/NewtonDescent.h"

void OptSolver::penaltyNewtonSolver(std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc, std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> findMaxStep, Eigen::VectorXd& x0, int numIter, double gradTol, double xTol, double fTol, double penaltyRatio, bool disPlayInfo)
{
	const int DIM = x0.rows();
	Eigen::VectorXd grad = Eigen::VectorXd::Zero(DIM);
	Eigen::SparseMatrix<double> hessian;

	Eigen::VectorXd neggrad, delta_x;
	double maxStepSize = 1.0;
	double reg = 1e-8;

	bool isProj = false;

	auto funValWithPenalty = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj)
	{
	    Eigen::VectorXd deriv;
	    Eigen::SparseMatrix<double> H;

		double E = objFunc(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);
		
		E += 0.5 * penaltyRatio * (x - x0).dot(x - x0);
		
		if(grad)
		{
			(*grad) = deriv + penaltyRatio * (x - x0);
		}
		if(hess)
		{
			Eigen::SparseMatrix<double> idMat = H;
			idMat.setIdentity();
			(*hess) = H + penaltyRatio * idMat;
		}
		return E;
	};
    

    Eigen::VectorXd initX = x0;

	int i = 0;
	for (; i < numIter; i++)
	{
		if(disPlayInfo)
			std::cout << "\niter: " << i << std::endl;
		double f = funValWithPenalty(x0, &grad, &hessian, isProj);
       
        Eigen::SparseMatrix<double> I(DIM, DIM);
		I.setIdentity();

		Eigen::SparseMatrix<double> H = hessian;
		Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(hessian);

		while (solver.info() != Eigen::Success)
		{
			if (disPlayInfo)
			{
				if (isProj)
					std::cout << "some small perturb is needed to remove round-off error, current reg = " << reg << std::endl;
				else
					std::cout << "Matrix is not positive definite, current reg = " << reg << std::endl;
			}
				
			hessian = H + reg * I;
			solver.compute(hessian);
			reg = std::max(2 * reg, 1e-16);
		}

		neggrad = -grad;
		delta_x = solver.solve(neggrad);

		maxStepSize = findMaxStep(x0, delta_x);

		double rate = LineSearch::backtrackingArmijo(x0, grad, delta_x, funValWithPenalty, maxStepSize);

		if (!isProj)
		{
			reg *= 0.5;
			reg = std::max(reg, 1e-16);
		}
		

		x0 = x0 + rate * delta_x;

		double fnew = funValWithPenalty(x0, &grad, NULL, isProj);
		if (disPlayInfo)
		{
			std::cout << "line search rate : " << rate << ", actual hessian : " << !isProj << ", reg = " << reg << std::endl;
			std::cout << "f_old: " << f << ", f_new: " << fnew << ", grad norm: " << grad.norm() << ", delta x: " << rate * delta_x.norm() << ", delta_f: " << f - fnew << std::endl;
		}
		
		if (f - fnew < 1e-5 || rate * delta_x.norm() < 1e-5 || grad.norm() < 1e-4)
			isProj = false;


		if (rate < 1e-8)
		{
			std::cout << "terminate with small line search rate (<1e-8): L2-norm = " << grad.norm() << std::endl;
			return;
		}

		if (grad.norm() < gradTol)
		{
			std::cout << "terminate with gradient L2-norm = " << grad.norm() << std::endl;
			return;
		}
			
		if (rate * delta_x.norm() < xTol)
		{
			std::cout << "terminate with small variable change, gradient L2-norm = " << grad.norm() << std::endl;
			return;
		}
			
		if (f - fnew < fTol)
		{ 
			std::cout << "terminate with small energy change, gradient L2-norm = " << grad.norm() << std::endl;
			return;
		}
	}
	if (i >= numIter)
		std::cout << "terminate with reaching the maximum iteration, with gradient L2-norm = " << grad.norm() << std::endl;
		
}

