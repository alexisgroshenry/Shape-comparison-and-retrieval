#include <Eigen/Core>
#include <Eigen/SparseCore>


#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <ostream>
#include <math.h> 

#include <igl/readOFF.h>
#include <igl/writeOFF.h>
#include <igl/doublearea.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/jet.h>
#include <igl/gaussian_curvature.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>


#include "HalfedgeBuilder.cpp"
#include "Histogram.cpp"
#include "Spectra/SymEigsShiftSolver.h"
#include "Spectra/MatOp/SparseSymShiftSolve.h"
#include "Spectra/MatOp/SparseGenMatProd.h"


using namespace Eigen;
using namespace std;
using namespace Spectra;

void get_circumcenter(Eigen::Vector3d a, Vector3d b, Vector3d c, Vector3d& center) {
	MatrixXd res = a;
	res += (pow((c - a).norm(), 2) * ((b - a).cross(c - a)).cross(b - a) + pow((b - a).norm(), 2) * ((c - a).cross(b - a)).cross(c - a)) / (2 * pow(((b - a).cross(c - a)).norm(), 2));
	center = res;
}

double compute_area(Vector3d a, Vector3d b, Vector3d c) {
	double area;
	Vector3d center;
	get_circumcenter(a, b, c, center);
	Vector3d mid1 = (a + b) / 2.;
	Vector3d mid2 = (a + c) / 2.;
	area = (mid1 - a).norm() * (mid1 - center).norm() / 2. + (mid2 - a).norm() * (mid2 - center).norm() / 2.;
	return area;
}

MatrixXd V;
MatrixXi F;

int vertexDegree(HalfedgeDS he, int v) {
	int res = 0;
	int e = he.getEdge(v);
	int e2 = he.getPrev(he.getOpposite(e));
	res++;
	while (e2 != e) {
		e2 = he.getPrev(he.getOpposite(e2));
		res++;
	}
	return res;
}

void calculate_angle(HalfedgeDS he, SparseMatrix<double>& M, SparseMatrix<double>& S, SparseMatrix<double>& S_inv, VectorXd& mesh_areas) {
	typedef Eigen::Triplet<double> T;
	vector<T> tripletList_m , tripletList_s_inv, tripletList_s;
	tripletList_s_inv.reserve(he.sizeOfVertices());
	tripletList_s.reserve(he.sizeOfVertices());
	//estimation of entries for number of coef in M
	tripletList_m.reserve(20*he.sizeOfVertices());
	for (int p = 0; p < he.sizeOfVertices(); p++) {
		int e = he.getEdge(p);
		int degree = vertexDegree(he, p);
		double s = 0;
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

			double val = (1 / tan(alpha) + 1 / tan(beta)) / 2;
			tripletList_m.push_back(T(p, pj, val));
			double tmp_area= compute_area(V.row(p), V.row(pj), V.row(pjP));
			if (he.getFace(e)!=-1)
				mesh_areas[he.getFace(e)] += tmp_area;
			s += tmp_area;
			e = he.getPrev(he.getOpposite(e));
		}
		tripletList_s.push_back(T(p, p, s));
		tripletList_s_inv.push_back(T(p, p, 1 / s));
	}
	M.setFromTriplets(tripletList_m.begin(), tripletList_m.end());
	S.setFromTriplets(tripletList_s.begin(), tripletList_s.end());
	S_inv.setFromTriplets(tripletList_s_inv.begin(), tripletList_s_inv.end());
}

//compute the eigenvalues of all the mesh whose names are in files_list.txt
void generate_eigenvalues() {
	cout << "Generating eigenvalues list" << endl;
	vector<string> list_files;
	ifstream infile("files_list.txt");
	string file;
	while (infile >> file) { list_files.push_back(file); }
	infile.close();

	fstream myfile;
	myfile.open("eigenvalues_list.txt", fstream::out);
	for (const auto& file : list_files) {
		cout << file << endl;
		igl::readOFF(file, V, F);

		HalfedgeBuilder* builder = new HalfedgeBuilder();
		HalfedgeDS he = builder->createMesh(V.rows(), F);

		SparseMatrix<double> M(he.sizeOfVertices(), he.sizeOfVertices());
		SparseMatrix<double> S(he.sizeOfVertices(), he.sizeOfVertices());
		SparseMatrix<double> S_inv(he.sizeOfVertices(), he.sizeOfVertices());
		VectorXd mesh_areas = VectorXd::Zero(F.rows());
		calculate_angle(he, M, S, S_inv,mesh_areas);

		SparseMatrix<double> N = S_inv * M;
		SparseSymShiftSolve<double> op(N);
		int nev = 20;
		int ncv = 30;
		SymEigsShiftSolver< double, LARGEST_MAGN, SparseSymShiftSolve<double> > eigs(&op, nev, ncv, 0.0);

		// Initialize and compute
		eigs.init();
		int nconv = eigs.compute();
		// Retrieve results
		VectorXd evalues;
		if (eigs.info() == SUCCESSFUL)
			evalues = eigs.eigenvalues();
		for (int i = 0; i < nev; i++)
			myfile << evalues(i) << "\t";
		myfile << endl;
	}
	myfile.close();
	cout << "finished" << endl;
}

//compute the histograms of all the mesh whose names are in files_list.txt
void generate_histograms() {
	cout << "Generating histogram list" << endl;
	vector<string> list_files;
	ifstream infile("files_list.txt");
	string file;
	while (infile >> file) { list_files.push_back(file); }
	infile.close();
	int count = 10;
	for (const auto& file : list_files) {
		cout << file << endl;
		igl::readOBJ(file, V, F);

		HalfedgeBuilder* builder = new HalfedgeBuilder();
		HalfedgeDS he = builder->createMesh(V.rows(), F);


		SparseMatrix<double> M(he.sizeOfVertices(), he.sizeOfVertices());
		SparseMatrix<double> S(he.sizeOfVertices(), he.sizeOfVertices());
		SparseMatrix<double> S_inv(he.sizeOfVertices(), he.sizeOfVertices());
		VectorXd mesh_areas = VectorXd::Zero(F.rows());
		calculate_angle(he, M, S, S_inv, mesh_areas);
		for (int i = 1; i < mesh_areas.size(); i++) {
			mesh_areas[i] += mesh_areas[i - 1];
		}

		SparseMatrix<double> N = S_inv * M;
		SparseSymShiftSolve<double> op(N);
		int nev = 10;
		int ncv = 18;
		SymEigsShiftSolver< double, LARGEST_MAGN, SparseSymShiftSolve<double> > eigs(&op, nev, ncv, 0.0);

		// Initialize and compute
		eigs.init();
		int nconv = eigs.compute();
		// Retrieve results
		VectorXd evalues;
		MatrixXd evecs;
		if (eigs.info() == SUCCESSFUL)
		{
			evalues = eigs.eigenvalues();
			evecs = eigs.eigenvectors();
		}


		for (int j = 0; j < nev; j++) {
			//not needed? --> evecs.col(j) = S * evecs.col(j);
			double norm = evecs.col(j).transpose() * S * evecs.col(j);
			evecs.col(j) /= sqrt(norm);
		}

		// to keep track of the eigen value sign
		MatrixXd eigen_values_sign(nev, nev);
		eigen_values_sign.setConstant(0);
		for (int k = 0; k < nev; k++) {
			if (evalues(k) < 0) {
				eigen_values_sign(k, k) = -1.;
			}
			else {
				eigen_values_sign(k, k) = 1.;
			}
		}

		MatrixXd GPS(he.sizeOfVertices(), nev);
		for (int j = 0; j < nev; j++) {
			GPS.col(j) = 1 / sqrt(abs(evalues(j))) * evecs.col(j);
		}

		std::string file_name = to_string(count) + ".txt";
		create_hist(F, GPS, 8, mesh_areas, 1000, 100, file_name, eigen_values_sign);
		count++;
	}


}

// ------------ main program ----------------
int main(int argc, char* argv[]){
	
	//generate_eigenvalues();

	//generate_histograms();
	igl::readOBJ("models/lion-poses/lion-01.obj", V, F);
	igl::opengl::glfw::Viewer viewer; // create the 3d viewer

	viewer.data().set_mesh(V, F);

	viewer.core(0).align_camera_center(V, F);
	viewer.launch();
	return 0;
}