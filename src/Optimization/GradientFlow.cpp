#include "../../include/Optimization/GradientFlow.h"
#include "../../include/Optimization/LineSearch.h"
#include "../../include/Optimization/NewtonDescent.h"

void OptSolver::gradientFlowSolver(std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> objFunc, Eigen::VectorXd& x0, int numIter, double gradTol, double theta, bool disPlayInfo)
{
	const int DIM = x0.rows();
	Eigen::VectorXd grad = Eigen::VectorXd::Zero(DIM);
	Eigen::SparseMatrix<double> hessian;

	Eigen::VectorXd neggrad, delta_x;
	double maxStepSize = 1.0;
	double reg = 1e-8;

	int i = 0;
	for (; i < numIter; i++)
	{
		if(disPlayInfo)
			std::cout << "\niter: " << i << std::endl;
        double f = objFunc(x0, &grad, &hessian, false);

        Eigen::SparseMatrix<double> H = hessian;
        Eigen::SparseMatrix<double> I(DIM, DIM);
        I.setIdentity();
        hessian = theta * H;

        Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(hessian);

        while (solver.info() != Eigen::Success)
        {
            if (disPlayInfo)
            {
                std::cout << "Matrix is not positive definite, current reg = " << reg << std::endl;
            }
            hessian = theta * H + reg * I;
            solver.compute(hessian);
            reg = std::max(2 * reg, 1e-16);
        }
        
        while(true)
        {
            
            neggrad = -grad;
            delta_x = solver.solve(neggrad);

            if(solver.info() != Eigen::Success)
            {
                std::cout << "Eigen solver failed, with residual: " << (hessian * delta_x - neggrad).norm() << std::endl;
                exit(1);
            }

            Eigen::VectorXd tmpX = x0 + delta_x;
            Eigen::VectorXd tmpgrad;

            double fnew = objFunc(tmpX, &tmpgrad, NULL, false);
            
            if(fnew < f)
            {
                x0 = tmpX;
                grad = tmpgrad;
                if (disPlayInfo)
                {
                    std::cout << "f_old: " << f << ", f_new: " << fnew << ", grad norm: " << grad.norm() << ", delta x: " << delta_x.norm() << ", delta_f: " << f - fnew << std::endl;
                }
                
                if (grad.norm() < gradTol)
                {
                    std::cout << "terminate with gradient L2-norm = " << grad.norm() << std::endl;
                    return;
                }
                break;
            }
            else
            {
                // double rate = LineSearch::backtrackingArmijo(x0, grad, delta_x, objFunc, 1.0);
                // if (disPlayInfo)
                // {
                //     std::cout << "not decrease energy, try with line search rate: " << rate << std::endl;
                // }
                // x0 = x0 + rate * delta_x;
                // fnew = objFunc(x0, &grad, NULL, false);
                // if (disPlayInfo)
                // {
                //     std::cout << "f_old: " << f << ", f_new: " << fnew << ", grad norm: " << grad.norm() << ", delta x: " << delta_x.norm() << ", delta_f: " << f - fnew << std::endl;
                // }
                // break;
                if (disPlayInfo)
                {
                    std::cout << "not decrease energy, current reg = " << reg << std::endl;
                }
                reg *= 2;
                hessian = theta * H + reg * I;
                solver.compute(hessian);
            }

        }
		
	}
	if (i >= numIter)
		std::cout << "terminate with reaching the maximum iteration, with gradient L2-norm = " << grad.norm() << std::endl;
		
}