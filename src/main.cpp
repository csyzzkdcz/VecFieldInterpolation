#include "polyscope/polyscope.h"

#include <igl/PI.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/boundary_loop.h>
#include <igl/exact_geodesic.h>
#include <igl/gaussian_curvature.h>
#include <igl/invert_diag.h>
#include <igl/lscm.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>
//#include <igl/triangle/triangulate.h>

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

#include <iostream>
#include <unordered_set>
#include <utility>

#include "../include/EnergyModel/energyModel.h"
#include "../include/OptProblem.h"
#include "../include/Optimization/NewtonDescent.h"
#include <map>

// The mesh, Eigen representation
Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;

Eigen::MatrixXd vecField(0, 0);
Eigen::VectorXd scalarField(0);

int bndTypes = 0;


void initFields()
{
	vecField.setRandom(meshV.rows(), 2);
	scalarField.setConstant(meshV.rows(), 1);

	for (int i = 0; i < meshV.rows(); i++)
	{
		double x = meshV(i, 0);
		double y = meshV(i, 1);

		if (x * x + y * y != 0)
		{
			vecField.row(i) << -y / (x * x + y * y), x / (x * x + y * y);
		}
		else
			vecField.row(i).setZero();
	}
}

void updateFieldsInView()
{
	polyscope::getSurfaceMesh("input mesh")
		->addVertexScalarQuantity("vertex scalar field", scalarField,
			polyscope::DataType::SYMMETRIC);

	Eigen::MatrixXd vec_vertices(meshV.rows(), 3);
	vec_vertices.block(0, 0, meshV.rows(), 2) = 0.02 * vecField;
	vec_vertices.col(2).setZero();
	polyscope::getSurfaceMesh("input mesh")
		->addVertexVectorQuantity("vertex vector field", vec_vertices, polyscope::VectorType::AMBIENT);
}

void getClampedDOFs(std::map<int, double>& clampedDOFs)
{
	clampedDOFs.clear();
	if (bndTypes == 0)
	{
		int nverts = meshV.rows();
		std::vector<int> bnd;
		igl::boundary_loop(meshF, bnd);

		for (int i = 0; i < bnd.size(); i++)
		{
			int vid = bnd[i];
			clampedDOFs[vid] = scalarField(vid);
			clampedDOFs[nverts + 2 * vid] = vecField(vid, 0);
			clampedDOFs[nverts + 2 * vid + 1] = vecField(vid, 1);
		}
	}
	else
	{
		int nverts = meshV.rows();
		std::vector<int> bnd;
		igl::boundary_loop(meshF, bnd);
		double minx = meshV.col(0).minCoeff();
		double maxx = meshV.col(0).maxCoeff();
		double miny = meshV.col(1).minCoeff();
		double maxy = meshV.col(1).maxCoeff();

		for (int i = 0; i < bnd.size(); i++)
		{
			int vid = bnd[i];
			double x = meshV(vid, 0);
			double y = meshV(vid, 1);

			if ((std::abs(x - minx) < 1e-7 && std::abs(y - miny) < 1e-7) ||
				(std::abs(x - maxx) < 1e-7 && std::abs(y - miny) < 1e-7) ||
				(std::abs(x - minx) < 1e-7 && std::abs(y - maxy) < 1e-7) ||
				(std::abs(x - maxx) < 1e-7 && std::abs(y - maxy) < 1e-7))
			{
				clampedDOFs[vid] = scalarField(vid);
				clampedDOFs[nverts + 2 * vid] = vecField(vid, 0);
				clampedDOFs[nverts + 2 * vid + 1] = vecField(vid, 1);
			}
			else
			{
				if (std::abs(x - minx) < 1e-7 || std::abs(x - maxx) < 1e-7)
				{
					clampedDOFs[nverts + 2 * vid] = vecField(vid, 0);
				}
				else
					clampedDOFs[nverts + 2 * vid + 1] = vecField(vid, 1);
			}
			
		}
	}
}

void callback() {
  ImGui::PushItemWidth(100);

  // scalar fields
  ImGui::Combo("bnd types", (int*)&bndTypes, "Direchlet\0Const Projection\0\0");
  if (ImGui::Button("update vector field"))
  {
	  std::map<int, double> clampedDOFs;
	  getClampedDOFs(clampedDOFs);
	  OptModel model(meshV, meshF, clampedDOFs);

	  auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj)
	  {
		  Eigen::VectorXd deriv;
		  Eigen::SparseMatrix<double> H;
		  double E = model.computeEnergy(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);

		  if (grad)
		  {
			  (*grad) = deriv;
		  }

		  if (hess)
		  {
			  (*hess) = H;
		  }

		  return E;
	  };
	  auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir)
	  {
		  return 1.0;
	  };
	  Eigen::VectorXd x;
	  model.convertState2Variable(vecField, scalarField, x);

	  OptSolver::newtonSolver(funVal, maxStep, x, 1000, 1e-7, 0, 0, true);
	  model.convertVariable2State(x, vecField, scalarField);

	  updateFieldsInView();
	  
  }
  if (ImGui::Button("reset vector field"))
  {
	  initFields();
	  updateFieldsInView();
  }

  ImGui::PopItemWidth();
}

int main(int argc, char** argv) {
	if (argc != 2)
	{
		std::cout << "please specific the mesh name!" << std::endl;
		return 1;
	}

	// Options
	polyscope::options::autocenterStructures = true;
	polyscope::view::windowWidth = 1024;
	polyscope::view::windowHeight = 1024;

	// Initialize polyscope
	polyscope::init();

	std::string filename = argv[1];
	std::cout << "loading: " << filename << std::endl;

	// Read the mesh
	igl::readOBJ(filename, meshV, meshF);

	
	// Register the mesh with Polyscope
	polyscope::registerSurfaceMesh("input mesh", meshV, meshF);
	initFields();

	// Add the callback
	polyscope::state::userCallback = callback;

	// Show the gui
	polyscope::show();

	return 0;
}
