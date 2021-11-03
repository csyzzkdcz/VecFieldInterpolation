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
#include "../include/Optimization/ProjectionScheme.h"
#include "../include/Optimization/GradientFlow.h"
#include "../include/Optimization/PenaltyNewton.h"
#include "../include/json.hpp"

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

enum MotionType
{
	MT_LINEAR = 0,
	MT_ENTIRE_LINEAR = 1,
	MT_ROTATION = 2,
	MT_SINEWAVE = 3,
	MT_COMPLICATE = 4,
	MT_SPIRAL = 5
};

enum OptMethod
{
    OPT_NT = 0,
    OPT_FEPR = 1,
    OPT_GF = 2,
    OPT_PENALTY = 3
};

// The mesh, Eigen representation
Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;

Eigen::MatrixXd vecField(0, 0);

Eigen::VectorXd scalarField(0);
Eigen::VectorXd scalarFieldNormalized(0);
std::string modelName;

std::vector<Eigen::MatrixXd> vecFieldLists;
std::vector<Eigen::VectorXd> scalarFieldLists;
std::vector<Eigen::VectorXd> scalarFieldNormalizedLists;

int bndTypes = 0;
DynamicType dynamicTypes = DynamicType::DT_Rotate;
MotionType motionType = MotionType::MT_LINEAR;
int totalFrames = 100;
int curFrame = 0;
float ratio = 0.02;
double penaltyRatio = 1e-3;
double GFTheta = 1;
bool isNormalize = false;
bool isWeightedVec = false;
bool isSaveScreenShot = false;
bool isShowVec = true;
bool isShowSca = true;
bool isSingularityMotion = false;
OptMethod optMet = OptMethod::OPT_NT;

double scaMin = 0;
double scaMax = 1;

double curMin = 0;
double curMax = 1;

std::string savingFolder;
std::string exeFolder;


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

bool loadProblem(const std::string& path)
{
	vecFieldLists.clear();
	scalarFieldLists.clear();
	scalarFieldNormalizedLists.clear();

    using json = nlohmann::json;
    std::ifstream inputJson(path);
    if(!inputJson)
    {
        std::cerr << "missing json file in " << path << std::endl;
        return false;
    }
    json jval;
    inputJson >> jval;

    std::string curFolder = jval["current_folder"];

    std::string meshName = jval["mesh_name"];
    meshName = curFolder + meshName;

    if(!igl::readOBJ(meshName, meshV, meshF))
    {
        std::cout << "failed to load mesh " << meshName << std::endl;
        return false;
    }

    std::string vecListPrefix = jval["vec_prefix"];

    vecListPrefix = curFolder + vecListPrefix;

    int i = 0;
    while (true)
    {
        Eigen::MatrixXd tmpVec;
        std::string fileName = vecListPrefix + "_" + std::to_string(i) + ".txt";
        bool isLoaded = loadVectorField(fileName, tmpVec);
        if (isLoaded)
        {
            vecFieldLists.push_back(tmpVec);
            i++;
        }
        else
            break;
    }

    std::string scaListPrefix = jval["sca_prefix"];
    scaListPrefix = curFolder + scaListPrefix;

    int j = 0;
    double min = std::numeric_limits<double>::infinity();
    double max = -min;

    while (true)
    {
        Eigen::VectorXd tmpVec;
        std::string fileName = scaListPrefix + "_" + std::to_string(j) + ".txt";
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
        std::cout << "error in the mismatch!" << std::endl;
        return false;
    }
    else if (i == 0)
    {
        std::cout << "no frames are loaded!" << std::endl;
        return false;
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
    }
    // load model type:
    isSingularityMotion = jval["is_singularity_motion"];

    std::string opM = jval["solver_type"];
    if(opM == "Newton")
        optMet = OPT_NT;
    else if (opM == "Gradient Flow")
        optMet = OPT_GF;
    else if (opM == "FEPR")
        optMet = OPT_FEPR;
    else if (opM == "Penalty")
        optMet = OPT_PENALTY;
    else
    {
        std::cerr << "wrong solver type: " << opM << std::endl;
        return false;
    }

    penaltyRatio = jval["penalty_ratio"];
	GFTheta = jval["gradient_flow_theta"];

    std::string dyType = jval["dynamic_type"];
    if(dyType == "rotation")
        dynamicTypes = DT_Rotate;
    else if (dyType == "enlarge")
        dynamicTypes = DT_ENLARGE;
    else if (dyType == "composite")
        dynamicTypes = DT_COMPOSITE;
    else if (dyType == "half_rotation")
        dynamicTypes = DT_HALF_ROTATE;
    else if (dyType == "half_enlarge")
        dynamicTypes = DT_HALF_ENLARGE;
    else if (dyType == "half_composite")
        dynamicTypes = DT_HALF_COMPOSITE;
    else
    {
        std::cerr << "wrong dynamic type: " << dyType << std::endl;
        return false;
    }

	std::string mType = jval["motion_type"];
	if (mType == "linear")
		motionType = MT_LINEAR;
	else if (mType == "entire_linear")
		motionType = MT_ENTIRE_LINEAR;
	else if (mType == "rotation")
		motionType = MT_ROTATION;
	else if (mType == "sin_wave")
		motionType = MT_SINEWAVE;
	else if (mType == "composite")
		motionType = MT_COMPLICATE;
	else if (mType == "spiral")
		motionType = MT_SPIRAL;
	else
	{
		std::cerr << "wrong motion type: " << mType << std::endl;
		return false;
	}

	std::string bType = jval["bnd_type"];
	if (bType == "Direchlet")
		bndTypes = 0;
	else if (bType == "const_projection")
		bndTypes = 1;
	else
	{
		std::cerr << "wrong bnd tyoe: " << bType << std::endl;
		return false;
	}

	return true;
}

bool saveProblem(const std::string& folder)
{
	using json = nlohmann::json;
	json jval;
	jval["current_folder"] = folder;
	jval["mesh_name"] = modelName + ".obj";
	igl::writeOBJ(folder + modelName + ".obj", meshV, meshF);
	
	jval["vec_prefix"] = "/sims/vecField";
	jval["sca_prefix"] = "/sims/scaField";

	jval["is_singularity_motion"] = isSingularityMotion;
	std::string sType = "";
	
	if (optMet == OPT_NT)
		sType = "Newton";
	else if (optMet == OPT_GF)
		sType = "Gradient Flow";
	else if (optMet == OPT_FEPR)
		sType = "FEPR";
	else if (optMet == OPT_PENALTY)
		sType = "Penalty";
	jval["solver_type"] = sType;

	jval["penalty_ratio"] = penaltyRatio;
	jval["gradient_flow_theta"] = GFTheta;

	std::string dyType = "";
	if (dynamicTypes == DT_Rotate)
		dyType = "rotation";
	else if (dynamicTypes == DT_ENLARGE)
		dyType = "enlarge";
	else if (dynamicTypes == DT_COMPOSITE)
		dyType = "composite";
	else if (dynamicTypes == DT_HALF_ROTATE)
		dyType = "half_rotation";
	else if (dynamicTypes == DT_HALF_ENLARGE)
		dyType = "half_enlarge";
	else if (dynamicTypes == DT_HALF_COMPOSITE)
		dyType = "half_composite";
	
	jval["dynamic_type"] = dyType;

	std::string mType = ""; 
	if (motionType == MT_LINEAR)
		mType = "linear";
	else if (motionType == MT_ENTIRE_LINEAR)
		mType = "entire_linear";
	else if (motionType == MT_ROTATION)
		mType = "rotation";
	else if (motionType == MT_SINEWAVE)
		mType = "sin_wave";
	else if (motionType == MT_COMPLICATE)
		mType = "composite";
	else if (motionType == MT_SPIRAL)
		mType = "spiral";

	jval["motion_type"] = mType;


	std::string bType = "";
	if (bndTypes == 0)
		bType = "Direchlet";
	else if (bndTypes == 1)
		bType = "const_projection";
	jval["bnd_type"] = bType;
	
	std::ofstream o(folder + "/data.json");
	o << std::setw(4) << jval << std::endl;
	return true;

}


void generateFieldsWithSingularity(Eigen::MatrixXd& mVecField, Eigen::VectorXd& mScalarField, double x0, double y0)
{
	int nverts = meshV.rows();
	mVecField.setRandom(nverts, 2);
	mScalarField.setConstant(nverts, 1);

	for(int i = 0; i < nverts; i++)
	{
		double x = meshV(i, 0);
		double y = meshV(i, 1);

		x -= x0;
		y -= y0;

		if (x * x + y * y != 0)
		{
			mVecField.row(i) << -y / (x * x + y * y), x / (x * x + y * y);
		}
		else
			mVecField.row(i).setZero();
	}
}

void generateSingularity(double &x0, double &y0, double t, MotionType motion)
{
    double r = 0.8;
    if(motion == MotionType::MT_LINEAR)
    {
        x0 = -r + 2 * r * t;
        y0 = 0;
    }
    else if (motion == MotionType::MT_ENTIRE_LINEAR)
    {
        x0 = -1.0 + 2 * t;
        y0 = 0;
    }
    else if (motion == MotionType::MT_ROTATION)
    {
        double theta = t * 2 * M_PI;
        x0 = r * std::cos(theta);
        y0 = r * std::sin(theta);
    }
    else if (motion == MotionType::MT_SINEWAVE)
    {
        x0 = -r + 2 * r * t;
        y0 = r * std::sin(M_PI / r * x0);
    }
    else if (motion == MotionType::MT_COMPLICATE)
    {
        double theta = t * 4 * M_PI;
        x0 = r * std::cos(theta);

        double p = -r + 2 * r * t;
        y0 = r * std::sin(4 * M_PI / r * p);
    }
    else if (motion == MotionType::MT_SPIRAL)
    {
        double curR = (1 - t) * r;
        double theta = t * 6 * M_PI;
        x0 = curR * std::cos(theta);
        y0 = curR * std::sin(theta);
    }
    else
    {
        std::cout << "undefined motion type!" << std::endl;
        exit(1);
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
		polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex scalar field")->setEnabled(isShowSca);
	}
	else
	{
		polyscope::getSurfaceMesh("input mesh")
			->addVertexScalarQuantity("vertex scalar field", scalarField,
				polyscope::DataType::SYMMETRIC);
		polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex scalar field")->setEnabled(isShowSca);
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
	polyscope::getSurfaceMesh("input mesh")->getQuantity("vertex vector field")->setEnabled(isShowVec);
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

void optimizeVecField(std::map<int, double>& clampedDOFs, Eigen::MatrixXd& initVecField, Eigen::VectorXd& initScaField, OptMethod methodType)
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
	if (methodType == OptMethod::OPT_GF)
	{
	    OptSolver::gradientFlowSolver(funVal, x, 1000, 1e-7, GFTheta, true);
	}
	else if(methodType == OptMethod::OPT_PENALTY)
	{
	    OptSolver::penaltyNewtonSolver(funVal, maxStep, x, 1000, 1e-7, 0, 0, penaltyRatio, true);
	}
	else if (methodType == OptMethod::OPT_FEPR)
	{
		Eigen::VectorXd newX;
	    OptSolver::FEPR(funVal, maxStep, x, model._mass, newX, true);
	    x = newX;
	}
	else
	{
		OptSolver::newtonSolver(funVal, maxStep, x, 1000, 1e-7, 0, 0, true);   
	}
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



void updateSavingFolder(const std::string& parentFolder)
{
	std::string filePath = parentFolder + "/results/" + modelName + "/" + std::to_string(totalFrames) + "Steps/";

	if (bndTypes == 0)
		filePath += "Direchlet/";
	else
		filePath += "ConstProj/";

	std::string DTmodel = "";

	if (!isSingularityMotion)
	{
		if (dynamicTypes == DynamicType::DT_Rotate)
			DTmodel = "boundary_motion/rotate/";
		else if (dynamicTypes == DynamicType::DT_ENLARGE)
			DTmodel = "boundary_motion/enlarge/";
		else if (dynamicTypes == DynamicType::DT_COMPOSITE)
			DTmodel = "boundary_motion/composite/";
		else if (dynamicTypes == DynamicType::DT_HALF_ROTATE)
			DTmodel = "boundary_motion/half_rotate/";
		else if (dynamicTypes == DynamicType::DT_HALF_ENLARGE)
			DTmodel = "boundary_motion/half_enlarge/";
		else
			DTmodel = "boundary_motion/half_composite/";
	}
	else
	{
		if (motionType == MotionType::MT_LINEAR)
			DTmodel = "singularity_motion/linear/";
		else if (motionType == MotionType::MT_ENTIRE_LINEAR)
			DTmodel = "singularity_motion/linear_entire/";
		else if (motionType == MotionType::MT_ROTATION)
			DTmodel = "singularity_motion/rotation/";
		else if (motionType == MotionType::MT_SINEWAVE)
			DTmodel = "singularity_motion/sin/";
		else if (motionType == MotionType::MT_COMPLICATE)
			DTmodel = "singularity_motion/complicate/";
		else if (motionType == MotionType::MT_SPIRAL)
			DTmodel = "singularity_motion/spiral/";
		else
		{
			std::cout << "undefined motion type!" << std::endl;
			exit(1);
		}
	}


	std::string optMT = "";
	if (optMet == OptMethod::OPT_NT)
		optMT = "Newton/";
	else if (optMet == OptMethod::OPT_FEPR)
		optMT = "FEPR/";
	else if (optMet == OptMethod::OPT_GF)
		optMT = "GF_" + std::to_string(GFTheta) + "/";
	else
		optMT = "Penalty_" + std::to_string(penaltyRatio) + "/";
	savingFolder = filePath + DTmodel + optMT;
	std::cout << "saving folder: " << savingFolder << std::endl;
}

void callback() {
  ImGui::PushItemWidth(100);
  if (ImGui::Button("Import Data List"))
  {
	  std::string jsonPath = igl::file_dialog_open();
	  if (!loadProblem(jsonPath))
	  {
		  std::cout << "load problem error! " << std::endl;
		  exit(1);
	  }
	  updateFieldsInView();
	  updateSavingFolder(exeFolder);
  }


  if (ImGui::Button("Import Data"))
  {
	  std::string jsonPath = igl::file_dialog_open();
	  if (!loadProblem(jsonPath))
	  {
		  std::cout << "load problem error! " << std::endl;
		  exit(1);
	  }
	  updateFieldsInView();
	  updateSavingFolder(exeFolder);
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
  if (ImGui::Combo("bnd types", (int*)&bndTypes, "Direchlet\0Const Projection\0\0"))
	  updateSavingFolder(exeFolder);
  if(ImGui::Combo("dynamic types", (int*)&dynamicTypes, "rotate\0enlarge\0composite\0half rotate\0half enlarge\0half composite\0\0"))
	  updateSavingFolder(exeFolder);
  if(ImGui::Checkbox("Use singularity motion", &isSingularityMotion))
	  updateSavingFolder(exeFolder);
  if(ImGui::Combo("Motion types", (int *)&motionType, "linear\0entire linear\0rotate\0sin-wave\0complicate\0spiral\0\0"))
	  updateSavingFolder(exeFolder);
  if(ImGui::Combo("Opt types", (int*)&optMet, "Newton\0FEPR\0GF\0Penalty\0\0"))
	  updateSavingFolder(exeFolder);
  if(ImGui::InputDouble("GF theta", &GFTheta))
	  updateSavingFolder(exeFolder);
  ImGui::SameLine();
  if(ImGui::InputDouble("Penalty ratio", &penaltyRatio))
	  updateSavingFolder(exeFolder);

  if (ImGui::Button("update vector field"))
  {
	  saveProblem(savingFolder);
	  std::map<int, double> clampedDOFs;
	  getClampedDOFs(clampedDOFs, vecField, scalarField);
	  optimizeVecField(clampedDOFs, vecField, scalarField, OptMethod::OPT_NT);
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
	  std::string folder = savingFolder + "/sims/";
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

	  saveProblem(savingFolder);

	  for (int i = 0; i <= numSteps; i++)
	  {
		  std::cout << "\ncurrent outer iter: " << i << std::endl;
		  
		  double r = 1 + 9.0 / numSteps * i;
		  double theta = 1.0 / numSteps * 2 * M_PI * i;
		  std::map<int, double> clampedDOFs;
		  Eigen::MatrixXd rotatedVecField;
		  if(!isSingularityMotion)
		    manipulateVecField(initVecField, rotatedVecField, dynamicTypes, theta, r);
		  else
		  {
		      double t = i * 1.0 / numSteps;
		      double x0, y0;
              generateSingularity(x0, y0, t, motionType);
              std::cout << "x0: " << x0 << ", y0: " << y0 << std::endl;
              generateFieldsWithSingularity(rotatedVecField, initScalarField, x0, y0);
		  }
		  getClampedDOFs(clampedDOFs, rotatedVecField, initScalarField);

		  if(i == 0)
		      optimizeVecField(clampedDOFs, curVecField, curScalarField, OptMethod::OPT_NT);
		  else
		      optimizeVecField(clampedDOFs, curVecField, curScalarField, optMet);
//		  curVecField = rotatedVecField;
//		  curScalarField = initScalarField;
		  
		  std::string vecPath = folder + "/vecField_" + std::to_string(i) + ".csv";
		  std::ofstream vecFile(vecPath);
		  for (int i = 0; i < curVecField.rows(); i++)
		  {
			  vecFile << curVecField(i, 0) << ",\t" << curVecField(i, 1) << ",\t" << 0 << std::endl;
		  }

		  vecPath = folder + "/vecField_" + std::to_string(i) + ".txt";
		  vecFile = std::ofstream(vecPath);
		  vecFile << curVecField << std::endl;
		  vecFieldLists.push_back(curVecField);


		  std::string magPath = folder + "/scaField_" + std::to_string(i) + ".csv";
		  std::ofstream magFile(magPath);
		  for (int i = 0; i < curScalarField.rows(); i++)
			  magFile << curScalarField[i] << ",\t" << 3.14159 << std::endl;

		  magPath = folder + "/scaField_" + std::to_string(i) + ".txt";
		  magFile = std::ofstream(magPath);
		  magFile << curScalarField << std::endl;
		  scalarFieldLists.push_back(curScalarField);

		  normalizeCurrentFrame(scalarField, scalarFieldNormalized);
		  scalarFieldNormalizedLists.push_back(scalarFieldNormalized);

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
	  updateSavingFolder(exeFolder);
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
			  std::string folder = savingFolder + "imags/";
			  std::cout << "saving folder: " << folder << std::endl;
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
	  updateSavingFolder(exeFolder);
  }
  if(ImGui::Checkbox("Enable Vector Field", &isShowVec))
  {
      updateFieldsInView();
  }
  ImGui::SameLine();
  if (ImGui::Checkbox("weighted by Scalar", &isWeightedVec))
  {
	  updateFieldsInView();
  }
  if(ImGui::Checkbox("Enable Scalar Field", &isShowSca))
      updateFieldsInView();
  ImGui::SameLine();
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

int main(int argc, char** argv)
{
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
	int id0 = filename.rfind(".obj");
	int id1 = filename.rfind("/");
	modelName = filename.substr(id1 + 1, id0 - id1 - 1);
	std::cout << modelName << std::endl;

	// Read the mesh
	igl::readOBJ(filename, meshV, meshF);
	
	// Register the mesh with Polyscope
	polyscope::registerSurfaceMesh("input mesh", meshV, meshF);
	initFields(vecField, scalarField);

	exeFolder = std::filesystem::current_path().u8string();
	updateSavingFolder(exeFolder);

	// Add the callback
	polyscope::state::userCallback = callback;

	// Show the gui
	polyscope::show();

	return 0;
}
