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
#include <igl/file_dialog_open.h>
#include <igl/file_dialog_save.h>
#include <igl/boundary_loop.h>
//#include <igl/triangle/triangulate.h>
#include <filesystem>
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <utility>

#include "../include/EnergyModel/energyModel.h"
#include "../include/OptProblem.h"
#include "../include/Optimization/NewtonDescent.h"
#include <map>

enum DynamicType
{
	DT_Rotate = 0,
	DT_ENLARGE = 1,
	DT_COMPOSITE = 2,
	DT_HALF_ROTATE = 3,
	DT_HALF_ENLARGE = 4,
	DT_HALF_COMPOSITE = 5
};

// The mesh, Eigen representation
Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;

Eigen::MatrixXd vecField(0, 0);

Eigen::VectorXd scalarField(0);
Eigen::VectorXd scalarFieldNormalized(0);

std::vector<Eigen::MatrixXd> vecFieldLists;
std::vector<Eigen::VectorXd> scalarFieldLists;
std::vector<Eigen::VectorXd> scalarFieldNormalizedLists;

int bndTypes = 0;
DynamicType dynamicTypes = DynamicType::DT_Rotate;
int totalFrames = 100;
int curFrame = 0;
float ratio = 0.02;
bool isNormalize = false;
bool isWeightedVec = false;
bool isSaveScreenShot = false;

double scaMin = 0;
double scaMax = 1;

double curMin = 0;
double curMax = 1;

void initFields(Eigen::MatrixXd& mVecField, Eigen::VectorXd& mScalarField)
{
	mVecField.setRandom(meshV.rows(), 2);
	mScalarField.setConstant(meshV.rows(), 1);

	for (int i = 0; i < meshV.rows(); i++)
	{
		double x = meshV(i, 0);
		double y = meshV(i, 1);

		if (x * x + y * y != 0)
		{
			mVecField.row(i) << -y / (x * x + y * y), x / (x * x + y * y);
		}
		else
			mVecField.row(i).setZero();
	}
}

void updateFieldsInView()
{
	std::cout << "update view" << std::endl;
	if (isNormalize)
	{
		polyscope::getSurfaceMesh("input mesh")
			->addVertexScalarQuantity("vertex scalar field", scalarFieldNormalized,
				polyscope::DataType::SYMMETRIC);
		polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex scalar field")->setEnabled(true);
	}
	else
	{
		polyscope::getSurfaceMesh("input mesh")
			->addVertexScalarQuantity("vertex scalar field", scalarField,
				polyscope::DataType::SYMMETRIC);
		polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex scalar field")->setEnabled(true);
	}
		
	

	Eigen::MatrixXd vec_vertices(meshV.rows(), 3);
	vec_vertices.block(0, 0, meshV.rows(), 2) = ratio * vecField;
	if (isWeightedVec)
	{
		for (int i = 0; i < meshV.rows(); i++)
			vec_vertices.row(i) *= scalarField(i);
	}
	vec_vertices.col(2).setZero();
	polyscope::getSurfaceMesh("input mesh")
		->addVertexVectorQuantity("vertex vector field", vec_vertices, polyscope::VectorType::AMBIENT);
	polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex vector field")->setEnabled(true);
}

void getClampedDOFs(std::map<int, double>& clampedDOFs, Eigen::MatrixXd refVecField, Eigen::VectorXd refScaField)
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
			clampedDOFs[vid] = refScaField(vid);
			clampedDOFs[nverts + 2 * vid] = refVecField(vid, 0);
			clampedDOFs[nverts + 2 * vid + 1] = refVecField(vid, 1);
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
				clampedDOFs[vid] = refScaField(vid);
				clampedDOFs[nverts + 2 * vid] = refVecField(vid, 0);
				clampedDOFs[nverts + 2 * vid + 1] = refVecField(vid, 1);
			}
			else
			{
				if (std::abs(x - minx) < 1e-7 || std::abs(x - maxx) < 1e-7)
				{
					clampedDOFs[nverts + 2 * vid] = refVecField(vid, 0);
				}
				else
					clampedDOFs[nverts + 2 * vid + 1] = refVecField(vid, 1);
			}
			
		}
	}
}

void optimizeVecField(std::map<int, double>& clampedDOFs, Eigen::MatrixXd& initVecField, Eigen::VectorXd& initScaField)
{
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
	model.convertState2Variable(initVecField, initScaField, x);

	OptSolver::newtonSolver(funVal, maxStep, x, 1000, 1e-7, 0, 0, true);
	model.convertVariable2State(x, initVecField, initScaField);
}

void manipulateVecField(const Eigen::MatrixXd& input, Eigen::MatrixXd& output, DynamicType type, double theta = 0, double r = 1)
{
	Eigen::Matrix2d A;
	if (theta == 0)
		A << 1, 0, 0, 1;
	else
		A << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);
	output = input;
	for (int i = 0; i < output.rows(); i++)
	{
		if (type == DynamicType::DT_Rotate)
		{
			output.row(i) = (A * input.row(i).transpose()).transpose();
		}
		else if (type == DynamicType::DT_ENLARGE)
			output.row(i) = r * input.row(i);
		else if (type == DynamicType::DT_COMPOSITE)
		{
			output.row(i) = r * (A * input.row(i).transpose()).transpose();
		}
		else if (type == DynamicType::DT_HALF_ROTATE)
		{
			if(meshV(i, 0) > 0)
				output.row(i) = (A * input.row(i).transpose()).transpose();
		}
		else if (type == DynamicType::DT_HALF_ENLARGE)
		{
			if (meshV(i, 0) > 0)
				output.row(i) = r * input.row(i);
		}
		else
		{
			if (meshV(i, 0) > 0)
				output.row(i) = (A * input.row(i).transpose()).transpose();
			else
				output.row(i) = r * input.row(i);
		}
	}
		
}

bool loadVectorField(std::string vecPath, Eigen::MatrixXd& vecF)
{
	int nverts = meshV.rows();
	std::ifstream vecfile(vecPath);

	if (!vecfile)
	{
		Eigen::VectorXd tmpSca;
		std::cout << "failed to vector fields from files." << vecPath << std::endl;
		initFields(vecF, tmpSca);
		return false;
	}
	else
	{
		//std::cout << "load vector fields from files." << vecPath << std::endl;
		vecF.resize(nverts, 2);
		for (int i = 0; i < nverts; i++)
		{
			std::string line;
			std::getline(vecfile, line);
			std::stringstream ss(line);
			std::string x, y;
			ss >> x;
			ss >> y;
			vecF.row(i) << std::stod(x), std::stod(y);
		}
		return true;
	}
}

bool loadScalarField(std::string scaPath, Eigen::VectorXd& scaF)
{
	int nverts = meshV.rows();
	std::ifstream scafile(scaPath);

	if (!scafile)
	{
		Eigen::MatrixXd tmpVec;
		std::cout << "failed to scalar fields from files." << scaPath << std::endl;
		initFields(tmpVec, scaF);
		return false;
	}
	else
	{
		//std::cout << "load scalar fields from files." << scaPath << std::endl;
		scaF.resize(nverts);
		for (int i = 0; i < nverts; i++)
		{
			std::string line;
			std::getline(scafile, line);
			std::stringstream ss(line);
			std::string x;
			ss >> x;
			scaF(i) = std::stod(x);
		}
		return true;
	}
}

void setBndDynamicTypesFromData(std::string dataPath)
{
	std::replace(dataPath.begin(), dataPath.end(), '\\', '/'); // handle the backslash issue for windows
	std::cout << "after replacement, path: " << dataPath << std::endl;

	int index = dataPath.rfind("/");
	dataPath = dataPath.substr(0, index);

	index = dataPath.rfind("/");
	
	std::string dymodeltype = dataPath.substr(index + 1, dataPath.size() - 1);
	std::cout << "dynamic type: " << dymodeltype << std::endl;

	dataPath = dataPath.substr(0, index);
	index = dataPath.rfind("/");
	std::string bndTypeString = dataPath.substr(index + 1, dataPath.size() - 1);
	std::cout << "bndTypeString: " << bndTypeString << std::endl;

	if (bndTypeString == "Direchlet")
		bndTypes = 0;
	else
		bndTypes = 1;

	if (dymodeltype == "rotate")
		dynamicTypes = DynamicType::DT_Rotate;
	else if (dymodeltype == "enlarge")
		dynamicTypes = DynamicType::DT_ENLARGE;
	else if (dymodeltype == "composite")
		dynamicTypes = DynamicType::DT_COMPOSITE;
	else if (dymodeltype == "half_rotate")
		dynamicTypes = DynamicType::DT_HALF_ROTATE;
	else if (dymodeltype == "half_enlarge")
		dynamicTypes = DynamicType::DT_HALF_ENLARGE;
	else
		dynamicTypes = DynamicType::DT_HALF_COMPOSITE;
}


void normalizeCurrentFrame(const Eigen::VectorXd& input, Eigen::VectorXd& normalizedScalarField, double* min = NULL, double* max = NULL)
{
	double curMin = input.minCoeff();
	double curMax = input.maxCoeff();

	if (min)
		curMin = *min;
	if (max)
		curMax = *max;

	if (curMin > curMax)
	{
		std::cout << "some errors in the input min and max values." << std::endl;

		curMin = input.minCoeff();
		curMax = input.maxCoeff();
	}

	normalizedScalarField = input;
	for (int i = 0; i < input.size(); i++)
	{
		normalizedScalarField(i) = (input(i) - curMin) / (curMax - curMin);
	}

}


void callback() {
  ImGui::PushItemWidth(100);
  if (ImGui::Button("Import Data List"))
  {
	  scalarFieldLists.clear();
	  scalarFieldNormalizedLists.clear();
	  vecFieldLists.clear();

	  std::string vecPath = igl::file_dialog_open();
	  
	  int index = vecPath.rfind("_");
	  std::string vecPrefix = vecPath.substr(0, index);
	  std::cout << "vector fields prefix is: " << vecPrefix << std::endl;

	  setBndDynamicTypesFromData(vecPath);

	  int i = 0;
	  while (true)
	  {
		  Eigen::MatrixXd tmpVec;
		  std::string fileName = vecPrefix + "_" + std::to_string(i) + ".txt";
		  bool isLoaded = loadVectorField(fileName, tmpVec);
		  if (isLoaded)
		  {
			  vecFieldLists.push_back(tmpVec);
			  i++;
		  }
		  else
			  break;
	  }

	  std::string scaPath = igl::file_dialog_open();
	  int index1 = scaPath.rfind("_");
	  std::string scaPrefix = scaPath.substr(0, index1);
	  std::cout << "scalar fields prefix is: " << scaPrefix << std::endl;

	  int j = 0;
	  double min = std::numeric_limits<double>::infinity();
	  double max = -min;

	  while (true)
	  {
		  Eigen::VectorXd tmpVec;
		  std::string fileName = scaPrefix + "_" + std::to_string(j) + ".txt";
		  bool isLoaded = loadScalarField(fileName, tmpVec);
		  if (isLoaded)
		  {
			  scalarFieldLists.push_back(tmpVec);
			  min = std::min(tmpVec.minCoeff(), min);
			  max = std::max(tmpVec.maxCoeff(), max);
			  j++;
		  }
		  else
			  break;
	  }

	  if (i != j)
	  {
		  std::cout << "error in the mismatch !" << std::endl;
	  }
	  else
	  {
		  std::cout << "total frame: " << i - 1 << std::endl;
		  totalFrames = i - 1;

		  for (int n = 0; n <= totalFrames; n++)
		  {
			  Eigen::VectorXd normalizedVec;
			  normalizeCurrentFrame(scalarFieldLists[n], normalizedVec, &min, &max);
			  scalarFieldNormalizedLists.push_back(normalizedVec);
		  }

		  vecField = vecFieldLists[0];
		  scalarField = scalarFieldLists[0];
		  scalarFieldNormalized = scalarFieldNormalizedLists[0];
		  scaMin = min;
		  scaMax = max;

		  curMin = scalarField.minCoeff();
		  curMax = scalarField.maxCoeff();

		  updateFieldsInView();
	  }
  }


  if (ImGui::Button("Import Data"))
  {
	  std::string vecPath = igl::file_dialog_open();
	  loadVectorField(vecPath, vecField);

	  setBndDynamicTypesFromData(vecPath);


	  std::string scaPath = igl::file_dialog_open();
	  loadScalarField(scaPath, scalarField);
	  normalizeCurrentFrame(scalarField, scalarFieldNormalized);
	  scaMin = scalarField.minCoeff();
	  scaMax = scalarField.maxCoeff();

	  curMin = scalarField.minCoeff();
	  curMax = scalarField.maxCoeff();

	  updateFieldsInView();
  }
  ImGui::SameLine();
  if (ImGui::Button("Export Data"))
  {
	  std::string vecPath = igl::file_dialog_save();
	  std::ofstream vecFile(vecPath);
	  vecFile << vecField << std::endl;

	  std::string scaPath = igl::file_dialog_save();
	  std::ofstream scaFile(scaPath);
	  scaFile << scalarField << std::endl;
  }
  // scalar fields
  ImGui::Combo("bnd types", (int*)&bndTypes, "Direchlet\0Const Projection\0\0");
  ImGui::Combo("dynamic types", (int*)&dynamicTypes, "rotate\0enlarge\0composite\0half rotate\0half enlarge\0half composite\0\0");

  if (ImGui::Button("update vector field"))
  {
	  std::map<int, double> clampedDOFs;
	  getClampedDOFs(clampedDOFs, vecField, scalarField);
	  optimizeVecField(clampedDOFs, vecField, scalarField);
	  normalizeCurrentFrame(scalarField, scalarFieldNormalized);

	  scaMin = scalarField.minCoeff();
	  scaMax = scalarField.maxCoeff();

	  curMin = scalarField.minCoeff();
	  curMax = scalarField.maxCoeff();

	  updateFieldsInView();
	  
  }
  if (ImGui::Button("reset vector field"))
  {
	  initFields(vecField, scalarField);
	  normalizeCurrentFrame(scalarField, scalarFieldNormalized);

	  scaMin = scalarField.minCoeff();
	  scaMax = scalarField.maxCoeff();

	  curMin = scalarField.minCoeff();
	  curMax = scalarField.maxCoeff();

	  updateFieldsInView();
  }
  if (ImGui::Button("update the dynamic vector field"))
  {
	  int numSteps = totalFrames;
	  std::string filePath = "C:/Users/csyzz/Projects/VecFieldInterpolation/build/results/" + std::to_string(numSteps) + "Steps/";

	  if (bndTypes == 0)
		  filePath += "Direchlet/";
	  else
		  filePath += "ConstProj/";

	  std::string DTmodel = "";

	  if (dynamicTypes == DynamicType::DT_Rotate)
		  DTmodel = "rotate/";
	  else if (dynamicTypes == DynamicType::DT_ENLARGE)
		  DTmodel = "enlarge/";
	  else if (dynamicTypes == DynamicType::DT_COMPOSITE)
		  DTmodel = "composite/";
	  else if (dynamicTypes == DynamicType::DT_HALF_ROTATE)
		  DTmodel = "half_rotate/";
	  else if (dynamicTypes == DynamicType::DT_HALF_ENLARGE)
		  DTmodel = "half_enlarge/";
	  else
		  DTmodel = "half_composite/";

	  std::string folder = filePath + DTmodel;
	  if (!std::filesystem::exists(folder))
	  {
		  std::cout << "create directory: " << folder << std::endl;
		  if (!std::filesystem::create_directories(folder))
		  {
			  std::cout << "create folder failed." << folder << std::endl;
		  }
	  }
	  Eigen::MatrixXd initVecField, curVecField;
	  Eigen::VectorXd initScalarField, curScalarField;

	  initFields(initVecField, initScalarField);
	  curVecField = initVecField;
	  curScalarField = initScalarField;

	  for (int i = 0; i <= numSteps; i++)
	  {
		  std::cout << "\ncurrent outer iter: " << i << std::endl;
		  
		  double r = 1 + 9.0 / numSteps * i;
		  double theta = 1.0 / numSteps * 2 * M_PI * i;
		  std::map<int, double> clampedDOFs;
		  Eigen::MatrixXd rotatedVecField;
		  manipulateVecField(initVecField, rotatedVecField, dynamicTypes, theta, r);
		  getClampedDOFs(clampedDOFs, rotatedVecField, initScalarField);

		  optimizeVecField(clampedDOFs, curVecField, curScalarField);

		  std::string vecPath = folder + "vecField_" + std::to_string(i) + ".csv";
		  std::ofstream vecFile(vecPath);
		  for (int i = 0; i < curVecField.rows(); i++)
		  {
			  vecFile << curVecField(i, 0) << ",\t" << curVecField(i, 1) << ",\t" << 0 << std::endl;
		  }

		  vecPath = folder + "vecField_" + std::to_string(i) + ".txt";
		  vecFile = std::ofstream(vecPath);
		  vecFile << curVecField << std::endl;


		  std::string magPath = folder + "magField_" + std::to_string(i) + ".csv";
		  std::ofstream magFile(magPath);
		  for (int i = 0; i < curScalarField.rows(); i++)
			  magFile << curScalarField[i] << ",\t" << 3.14159 << std::endl;

		  magPath = folder + "magField_" + std::to_string(i) + ".txt";
		  magFile = std::ofstream(magPath);
		  magFile << curScalarField << std::endl;

	  }
	  vecField = curVecField;
	  scalarField = curScalarField;
	  normalizeCurrentFrame(scalarField, scalarFieldNormalized);

	  scaMin = scalarField.minCoeff();
	  scaMax = scalarField.maxCoeff();

	  curMin = scalarField.minCoeff();
	  curMax = scalarField.maxCoeff();

	  updateFieldsInView();
  }
  ImGui::SameLine();
  if (ImGui::InputInt("total Frames ", &totalFrames))
  {
	  if (totalFrames <= 0)
		  totalFrames = 10;
  }

  if (ImGui::DragInt("current frame", &curFrame, 0.1, 0, totalFrames))
  {
	  std::cout << "current frame: " << curFrame << std::endl;
	  if (curFrame < vecFieldLists.size())
	  {
		  vecField = vecFieldLists[curFrame];
		  scalarField = scalarFieldLists[curFrame];
		  scalarFieldNormalized = scalarFieldNormalizedLists[curFrame];

		  curMin = scalarField.minCoeff();
		  curMax = scalarField.maxCoeff();

		  updateFieldsInView();
		  if (isSaveScreenShot)
		  {
			  int numSteps = totalFrames;
			  std::string filePath = "C:/Users/csyzz/Projects/VecFieldInterpolation/build/results/" + std::to_string(numSteps) + "Steps/";

			  if (bndTypes == 0)
				  filePath += "Direchlet/";
			  else
				  filePath += "ConstProj/";

			  std::string DTmodel = "";

			  if (dynamicTypes == DynamicType::DT_Rotate)
				  DTmodel = "rotate/";
			  else if (dynamicTypes == DynamicType::DT_ENLARGE)
				  DTmodel = "enlarge/";
			  else if (dynamicTypes == DynamicType::DT_COMPOSITE)
				  DTmodel = "composite/";
			  else if (dynamicTypes == DynamicType::DT_HALF_ROTATE)
				  DTmodel = "half_rotate/";
			  else if (dynamicTypes == DynamicType::DT_HALF_ENLARGE)
				  DTmodel = "half_enlarge/";
			  else
				  DTmodel = "half_composite/";

			  std::string folder = filePath + DTmodel + "imags/";
			  if (!std::filesystem::exists(folder))
			  {
				  std::cout << "create directory: " << folder << std::endl;
				  if (!std::filesystem::create_directories(folder))
				  {
					  std::cout << "create folder failed." << folder << std::endl;
				  }
			  }
			  polyscope::screenshot(folder + "results_" + std::to_string(curFrame) + ".png");
		  }
		 
	  }
  }
  if (ImGui::DragFloat("vector ratio", &ratio, 0.001, 0, 1))
  {
	  updateFieldsInView();
  }
  if (ImGui::Checkbox("weighted by Scalar", &isWeightedVec))
  {
	  updateFieldsInView();
  }

  if (ImGui::Checkbox("normalize", &isNormalize))
  {
	  updateFieldsInView();
  }
  ImGui::Checkbox("screen shot", &isSaveScreenShot);

  ImGui::InputDouble("global min", &scaMin);
  ImGui::SameLine();
  ImGui::InputDouble("global max", &scaMax);

  ImGui::InputDouble("current min", &curMin);
  ImGui::SameLine();
  ImGui::InputDouble("current max", &curMax);

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
	initFields(vecField, scalarField);

	// Add the callback
	polyscope::state::userCallback = callback;

	// Show the gui
	polyscope::show();

	return 0;
}
