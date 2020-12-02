#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <ostream>
#include <igl/readOFF.h>
#include <math.h> 
#include <igl/writeOFF.h>
#include <igl/doublearea.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/jet.h>

#include <igl/gaussian_curvature.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>


#include "HalfedgeBuilder.cpp"


using namespace Eigen; // to use the classes provided by Eigen library
using namespace std;

MatrixXd V;
MatrixXi F;


void get_center_circumcenter(Vector3d a, Vector3d b, Vector3d c, Vector3d& center) {
	MatrixXd res = a;
	res += (pow((c - a).norm(), 2) * ((b - a).cross(c - a)).cross(b - a) + pow((b - a).norm(), 2) * ((c - a).cross(b - a)).cross(c - a)) / (2 * pow(((b - a).cross(c - a)).norm(), 2));
	center = res;
}

float compute_area(Vector3d a, Vector3d b, Vector3d c) {
	float area;
	Vector3d center;
	get_center_circumcenter(a, b, c, center);
	Vector3d mid1 = (a + b) / 2.;
	Vector3d mid2 = (a + c) / 2.;
	area = (mid1 - a).norm() * (mid1 - center).norm() / 2. + (mid2 - a).norm() * (mid2 - center).norm() / 2.;
	return area;

}
int vertexDegree(HalfedgeDS he, int v) {
	int res = 0;
	int e = he.getEdge(v);
	int e2 = he.getNext(he.getOpposite(e));
	res++;
	while (e2 != e) {
		e2 = he.getNext(he.getOpposite(e2));
		res++;
	}
	return res;
}

void calculate_angle(HalfedgeDS he, MatrixXd& M, MatrixXd& S) {
	M = MatrixXd(he.sizeOfVertices(), he.sizeOfVertices());
	S = MatrixXd(he.sizeOfVertices(), he.sizeOfVertices());
	S.setZero();
	M.setZero();
	std::cout << "Compute M and S" << std::endl;
	for (int p = 0; p < he.sizeOfVertices(); p++) {
		int e = he.getEdge(p);
		int degree = vertexDegree(he, p);
		
		for (int k = 0; k < degree; k++) {
			int pj = he.getTarget(he.getOpposite(e));
			int pjP = he.getTarget(he.getNext(he.getOpposite(e)));
			int pjM = he.getTarget(he.getNext(e));

			double l = (V.row(p) - V.row(pj)).norm();
			double l1 = (V.row(p) - V.row(pjP)).norm();
			double l2 = (V.row(pjP) - V.row(pj)).norm();
			double alpha = acos((l1 * l1 + l2 * l2 - l * l) / (2 * l1 * l2));
			l1 = (V.row(p) - V.row(pjM)).norm();
			l2 = (V.row(pjM) - V.row(pj)).norm();
			double beta = acos((l1 * l1 + l2 * l2 - l * l) / (2 * l1 * l2));

			M(p, pj) = (1 / tan(alpha) + 1 / tan(beta)) / 2;
			S(p, p) += compute_area(V.row(p), V.row(pj), V.row(pjP));
			e = he.getPrev(he.getOpposite(e));			
		}
	}
	
}

float max_dist(MatrixXd GPS) {
	float dist = 0;
	for (int k = 0; k < GPS.rows(); k++) {
		if (dist < GPS.row(k).norm()) {
			dist = GPS.row(k).norm();
		}
	}
	return dist;
}

int get_hist(int k, int i, float dist_max, int m,MatrixXd GPS) {
	float dist_k = GPS.row(k).norm();
	float dist_i = GPS.row(i).norm();
	int circle_k = (int)(dist_k * m / dist_max);
	int circle_i = (int)(dist_i * m / dist_max);
	if (circle_k == m) {
		circle_k = m - 1;
	}
	if (circle_i == m) {
		circle_i = m - 1;
	}
	if (circle_i == circle_k) {
		return circle_i;
	}

	int max = circle_i;
	int min = circle_k;
	if (max < min) {
		max = circle_k;
		min = circle_i;
	}

	int indice = m - 2;
	int sum = m - 1;
	for (int i = 0; i < min; i++) {
		sum += indice;
		indice--;
	}
	
	return sum + max ;

}

void create_hist(MatrixXd GPS, MatrixXd& hist,int m) {
	MatrixXd res = MatrixXd::Zero(GPS.rows(),GPS.rows());
	for (int k = 0; k < GPS.rows(); k++) {
		for (int i = k; i < GPS.rows(); i++) {
			res(k, i) = GPS.row(k) * GPS.row(i).transpose();
		}
	}
	float max_value = res.maxCoeff();
	// distance will always be between 0 and 100 ?
	res = res * 100 / max_value;
	std::cout << res << std::endl;

	float dist_max = max_dist(GPS);
	MatrixXd count = MatrixXd::Zero(m * (m + 1) / 2,1); // number of points
	hist = MatrixXd::Zero(m*(m+1)/2,100);
	for (int k = 0; k < GPS.rows(); k++) {
		for (int i = k; i < GPS.rows(); i++) {
			float dist = (int)(res(k, i));
			if (dist == 100) {
				dist = 99;
			}
			int histogram = get_hist(k,i,dist_max,m,GPS);
			
			hist(histogram,dist) += 1;
			count(histogram,0)++;
		}
	}

	// get percentage
	for (int i = 0; i < m * (m + 1) / 2; i++) {
		if ((int)count(i, 0) != 0) {
			hist.row(i) = hist.row(i) / (int)count(i, 0) * 100 ;
		}
		
	}

}

void save_hist(MatrixXd hist,std::string text_file) {
	fstream myfile;
	myfile.open(text_file, fstream::out);
	for (int k = 0; k < hist.rows(); k++) {
		for (int i = 0; i < hist.cols(); i++) {
			myfile << hist(k,i) << "\t";
		}
		myfile << std::endl;
	}
	myfile.close();
}

void createOctagon(MatrixXd& Vertices, MatrixXi& Faces)
{
	Vertices = MatrixXd(6, 3);
	Faces = MatrixXi(8, 3);

	Vertices << 0.0, 0.0, 1.0,
		1.000000, 0.000000, 0.000000,
		0.000000, 1.00000, 0.000000,
		-1.000000, 0.000000, 0.000000,
		0.000000, -1.000000, 0.000000,
		0.000000, 0.000000, -1.000000;

	Faces << 0, 1, 2,
		0, 2, 3,
		0, 3, 4,
		0, 4, 1,
		5, 2, 1,
		5, 3, 2,
		5, 4, 3,
		5, 1, 4;
}


// ------------ main program ----------------
int main(int argc, char *argv[])
{	
	// igl::readOBJ("elephant-02.obj", V, F);
	createOctagon(V, F);
	HalfedgeBuilder* builder = new HalfedgeBuilder();
	HalfedgeDS he = builder->createMesh(V.rows(), F);

	/*
	MatrixXd M, S;
	calculate_angle(he, M, S);
	std::cout << M << std::endl;
	std::cout << S << std::endl;
	*/

	MatrixXd GPS(4, 2);
	GPS << 1, 2,
		2, 3,
		3, 4,
		4, 5;

	std::cout << GPS << std::endl;

	MatrixXd hist;
	create_hist(GPS, hist,2);
	std::cout << hist.cols() << std::endl;

	save_hist(hist,"octogon_hist.txt");

	

	igl::opengl::glfw::Viewer viewer; // create the 3d viewer
	viewer.data().set_mesh(V, F);
	viewer.core(0).align_camera_center(V, F);
	viewer.launch(); // run the viewer
}
