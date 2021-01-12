double max_dist(MatrixXd GPS, VectorXi sampled_points) {
	double dist = 0;
	for (int k = 0; k < sampled_points.size(); k++) {
		double n = GPS.row(sampled_points[k]).norm();
		if (dist < n) {
			dist = n;
		}
	}
	return dist;
}

int get_hist(int k, int i, double dist_max, int m, MatrixXd GPS) {
	double dist_k = GPS.row(k).norm();
	double dist_i = GPS.row(i).norm();
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
	return sum + max;
}

void save_hist(MatrixXd hist, double min_value, double max_value, std::string text_file) {
	fstream myfile;
	myfile.open(text_file, fstream::out);

	myfile << min_value << endl;
	myfile << max_value << endl;
	for (int k = 0; k < hist.rows(); k++) {
		for (int i = 0; i < hist.cols(); i++) {
			myfile << hist(k, i) << "\t";
		}
		myfile << endl;
	}
	myfile.close();
}

int findFace(double r, VectorXd mesh_areas) {
	if (mesh_areas.size() == 1)
		return 0;
	int median_idx = mesh_areas.size() / 2;
	double median = mesh_areas[median_idx];
	if (r < median)
		return findFace(r, mesh_areas.head(median_idx));
	else
		return median_idx + findFace(r, mesh_areas.tail(mesh_areas.size()-median_idx));
}

void create_hist(MatrixXi F, MatrixXd GPS, int m, VectorXd mesh_areas, int n, int b, std::string text_file, MatrixXd eigen_values_sign) {
	// cout << "		Sampling points" << endl;
	VectorXi sampled_points(n);
	double max_area = mesh_areas[mesh_areas.size() - 1];
	// cout << max_area << endl;
	for (int i = 0; i < n; i++) {
		double randNum = (double)rand() / RAND_MAX * max_area;
		//binary search to find the corresponding face
		int f = findFace(randNum, mesh_areas);
		//pick a random vertex in the face
		sampled_points[i] = F.row(f)[rand() % 3];
		//cout << sampled_points[i] << endl;
		/*same sampling method as in osada paper
		double r1 = rand();
		double r2 = rand();
		sampled_points.row(i) = (1 - sqrt(r1)) * V.row(F[f, 0]) + sqrt(r1) * (1 - r2) * V.row(F[f, 1]) + sqrt(r1) * r2 * V.row(F[f, 2]);*/
	}
	MatrixXd res(n, n);
	res.setZero();
	// cout << "		Computing pairwise distances" << endl;
	for (int k = 0; k < n; k++) {
		for (int i = k; i < n; i++) {
			res(k, i) = GPS.row(sampled_points[k]) * GPS.row(sampled_points[i]).transpose();
		}
	}
	// cout << "		Filling histogram" << endl;
	double max_value = res.maxCoeff();
	double min_value = res.minCoeff();
	// distance will always be between 0 and 100 ?
	MatrixXd min_mat(n, n);
	min_mat.setConstant(min_value);
	res = (res - min_mat) * b / (max_value - min_value);
	double dist_max = max_dist(GPS, sampled_points);
	int counter = 0;
	int nb = 0;
	MatrixXd count = MatrixXd::Zero(m * (m + 1) / 2, 1); // number of points in each histogram
	MatrixXd hist = MatrixXd::Zero(m * (m + 1) / 2, b);
	for (int k = 0; k < n; k++) {
		for (int i = k; i < n; i++) {
			int dist = (int)(res(k, i));
			if (dist < 100)
			{
				if (dist == b) {
					dist = b - 1;
				}
				int histogram = get_hist(sampled_points[k], sampled_points[i], dist_max, m, GPS);

				hist(histogram, dist) += 1;
				count(histogram, 0) += 1;
			}



		}
	}
	// get percentage
	for (int i = 0; i < m * (m + 1) / 2; i++) {
		if ((int)count(i, 0) != 0) {
			hist.row(i) = hist.row(i) / (int)count(i, 0) * 100;
		}
	}
	// cout << "     Saving histogram" << endl;
	save_hist(hist, min_value, max_value, text_file);
}