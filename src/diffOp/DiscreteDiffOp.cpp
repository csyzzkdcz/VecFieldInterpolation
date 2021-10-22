#include "../../include/diffOp/DiscreteDiffOp.h"
#include <igl/face_areas.h>


double DiscreteDiffOp::discreteCurl(const MatrixXd& V, const MatrixXi& F, const VectorXd& area, const Eigen::MatrixXd& vecField, const int faceId, Eigen::VectorXd* deriv, Eigen::MatrixXd* hess)
{
	int dim = F.cols(); // 3 for the triangle mesh, 4 for the quad mesh
	double curl = 0;

	int nDOFs = dim * 2;

	if (deriv)
	{
		deriv->setZero(nDOFs);
	}
	if (hess)
	{
		hess->setZero(nDOFs, nDOFs);
	}

	for (int i = 0; i < dim; i++)
	{
		int vid0 = F(faceId, i);
		int vid1 = F(faceId, (i + 1) % dim);

		Eigen::VectorXd e = (V.row(vid1) - V.row(vid0)).segment<2>(0);

		curl += (vecField.row(vid0) + vecField.row(vid1)).dot(e) / 2 / area(faceId);

		if (deriv)
		{
			deriv->segment<2>(2 * i) += e / 2 / area(faceId);
			deriv->segment<2>(2 * ((i + 1) % dim)) += e / 2 / area(faceId);
		}
	}
	return curl;
}


double DiscreteDiffOp::discreteDiv(const MatrixXd& V, const MatrixXi& F, const VectorXd& area, const Eigen::MatrixXd& vecField, const int faceId, Eigen::VectorXd* deriv, Eigen::MatrixXd* hess)
{
	int dim = F.cols(); // 3 for the triangle mesh, 4 for the quad mesh
	double div = 0;

	int nDOFs = dim * 2;

	if (deriv)
	{
		deriv->setZero(nDOFs);
	}
	if (hess)
	{
		hess->setZero(nDOFs, nDOFs);
	}

	for (int i = 0; i < dim; i++)
	{
		int vid0 = F(faceId, i);
		int vid1 = F(faceId, (i + 1) % dim);

		Eigen::Vector2d e = (V.row(vid1) - V.row(vid0)).segment<2>(0);
		Eigen::Vector2d eperp(e(1), -e(0));

		div += (vecField.row(vid0) + vecField.row(vid1)).dot(eperp) / 2 / area(faceId);

		if (deriv)
		{
			deriv->segment<2>(2 * i) += eperp / 2 / area(faceId);
			deriv->segment<2>(2 * ((i + 1) % dim)) += eperp / 2 / area(faceId);
		}
	}
	return div;
}