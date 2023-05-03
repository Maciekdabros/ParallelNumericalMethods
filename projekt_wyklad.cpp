#include <iostream>
#include <iomanip>
#include <random>
#include <omp.h>
#include <thread>
#include <vector>
#include <fstream>
#include <mutex>
using namespace std;

//Generation
void print_matrix(const vector<vector<double>>& matrix, int cols) {
	for (const auto& row : matrix) {
		for (int i = 0; i < cols + 1; i++) {
			cout << fixed << setprecision(4) << row[i] << " ";
		}
		cout << endl;
	}
}
void print_vector(const vector<vector<double>>& matrix) {
	int size = matrix.size();
	std::cout << "[";
	for (int i = 0; i < size; i++) {
		std::cout << std::fixed << std::setprecision(4) << matrix[i][0];
		if (i < size - 1) {
			std::cout << ", ";
		}
	}
	std::cout << "]" << std::endl;
}
vector<vector<double>> generate_matrix(int size) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(-1.0, 1.0);

	vector<vector<double>> matrix(size, vector<double>(size + 1));
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size + 1; j++) {
			matrix[i][j] = dis(gen);
		}
	}

	return matrix;
}
vector<vector<double>> generate_matrix2(int size) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(-1.0, 1.0);
	uniform_real_distribution<> diagonal_dis(2.0, 5.0);
	uniform_real_distribution<> b_dis(-1.0, 1.0);

	vector<vector<double>> matrix(size, vector<double>(size + 1)); // Add extra column for b
	for (int i = 0; i < size; i++) {
		double sum = 0;
		for (int j = 0; j < size; j++) {
			matrix[i][j] = dis(gen);
			sum += abs(matrix[i][j]);
		}
		// Make the diagonal element greater than the sum of other elements in the row
		matrix[i][i] = diagonal_dis(gen) * sum;

		// Generate the b value and add it to the last column
		matrix[i][size] = b_dis(gen);
	}

	return matrix;
}

//Monte Carlo
double f(double x) {
	return pow(x, 3) - 4 * pow(x, 2) + 1;
}
double random_number(double a, double b, mt19937& gen) {
	uniform_real_distribution<> dis(a, b);
	return dis(gen);
}
double monte_carlo_integration(double a, double b, int n, mt19937& gen) {
	double sum = 0.0;
	for (int i = 0; i < n; i++) {
		double x = random_number(a, b, gen);
		sum += f(x);
	}

	return sum * (b - a) / n;
}
double monte_carlo_integration_parallel(double a, double b, int n, mt19937& gen) {
	double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < n; i++) {
		double x = random_number(a, b, gen);
		sum += f(x);
	}
	return sum * (b - a) / n;
}
void monte_carlo_worker_thread(double a, double b, int n, double& sum, mt19937& gen) {
	for (int i = 0; i < n; i++) {
		double x = random_number(a, b, gen);
		sum += f(x);
	}
}
double monte_carlo_integration_parallel_thread(double a, double b, int n, mt19937& gen) {
	int num_threads = thread::hardware_concurrency();
	vector<thread> threads(num_threads);
	vector<double> partial_sums(num_threads, 0.0);

	int n_per_thread = n / num_threads;

	for (int i = 0; i < num_threads; i++) {
		threads[i] = thread(monte_carlo_worker_thread, a, b, n_per_thread, ref(partial_sums[i]), ref(gen));
	}

	for (auto& t : threads) {
		t.join();
	}

	double sum = 0.0;
	for (const auto& partial_sum : partial_sums) {
		sum += partial_sum;
	}

	return sum * (b - a) / n;
}

//Simpson
double simpson_sequential(double a, double b, int n)
{
	double h = (b - a) / n;
	double sum = f(a) + f(b), x;

	for (int i = 1; i < n; i++)
	{
		x = a + i * h;
		if (i % 2 == 0)
			sum += 2 * f(x);
		else
			sum += 4 * f(x);
	}
	sum *= h / 3;
	return sum;
}
double simpson_parallel(double a, double b, int n)
{
	double h = (b - a) / n;
	double sum = f(a) + f(b), x;

#pragma omp parallel for reduction(+:sum)
	for (int i = 1; i < n; i++)
	{
		x = a + i * h;
		if (i % 2 == 0)
			sum += 2 * f(x);
		else
			sum += 4 * f(x);
	}

	sum *= h / 3;
	return sum;
}
void simpson_worker_thread(double a, double h, int start, int end, int stride, double& sum) {
	double x;
	for (int i = start; i < end; i += stride) {
		x = a + i * h;
		if (i % 2 == 0)
			sum += 2 * f(x);
		else
			sum += 4 * f(x);
	}
}

double simpson_parallel_thread(double a, double b, int n) {
	double h = (b - a) / n;
	double sum = f(a) + f(b);

	int num_threads = thread::hardware_concurrency();
	vector<thread> threads(num_threads);
	vector<double> partial_sums(num_threads, 0);

	int n_per_thread = n / num_threads;

	for (int t = 0; t < num_threads; ++t) {
		int start = 1 + t * n_per_thread;
		int end = min(start + n_per_thread, n);
		threads[t] = thread(simpson_worker_thread, a, h, start, end, 1, ref(partial_sums[t]));
	}

	for (auto& t : threads) {
		t.join();
	}

	for (double partial_sum : partial_sums) {
		sum += partial_sum;
	}

	sum *= h / 3;
	return sum;
}

//Gauss elimination
vector<vector<double>> gauss_elimination_sequential(const vector<vector<double>>& augmented_matrix) {
	int size = augmented_matrix.size();
	vector<vector<double>> matrix = augmented_matrix;

	for (int i = 0; i < size; i++) {
		int pivot_row = i;
		for (int j = i + 1; j < size; j++) {
			if (abs(matrix[j][i]) > abs(matrix[pivot_row][i])) {
				pivot_row = j;
			}
		}

		if (matrix[pivot_row][i] == 0.0) {
			cerr << "Brak jednoznacznego rozwiazania" << endl;
			exit(1);
		}

		swap(matrix[i], matrix[pivot_row]);

		for (int j = i + 1; j < size; j++) {
			double factor = matrix[j][i] / matrix[i][i];
			for (int k = i; k <= size; k++) {
				matrix[j][k] -= factor * matrix[i][k];
			}
		}
	}

	vector<vector<double>> result(size, vector<double>(1));
	for (int i = size - 1; i >= 0; i--) {
		double sum = 0;
		for (int j = i + 1; j < size; j++) {
			sum += matrix[i][j] * result[j][0];
		}
		result[i][0] = (matrix[i][size] - sum) / matrix[i][i];
	}

	return result;
}
vector<vector<double>> gauss_elimination_parallel(const vector<vector<double>>& augmented_matrix) {
	int size = augmented_matrix.size();
	vector<vector<double>> matrix = augmented_matrix;
	for (int i = 0; i < size; i++) {
		int pivot_row = i;
		for (int j = i + 1; j < size; j++) {
			if (abs(matrix[j][i]) > abs(matrix[pivot_row][i])) {
				pivot_row = j;
			}
		}

		if (matrix[pivot_row][i] == 0.0) {
			cerr << "Brak jednoznacznego rozwiazania" << endl;
			exit(1);
		}

		swap(matrix[i], matrix[pivot_row]);

#pragma omp parallel for
		for (int j = i + 1; j < size; j++) {
			double factor = matrix[j][i] / matrix[i][i];
			for (int k = i; k <= size; k++) {
				matrix[j][k] -= factor * matrix[i][k];
			}
		}
	}

	vector<vector<double>> result(size, vector<double>(1));
	for (int i = size - 1; i >= 0; i--) {
		double sum = 0;

#pragma omp parallel for reduction(+:sum)
		for (int j = i + 1; j < size; j++) {
			sum += matrix[i][j] * result[j][0];
		}
		result[i][0] = (matrix[i][size] - sum) / matrix[i][i];
	}

	return result;
}
void gauss_elimination_worker_thread(vector<vector<double>>& matrix, int start_row, int end_row, int col) {
	for (int j = start_row; j < end_row; j++) {
		double factor = matrix[j][col] / matrix[col][col];
		for (int k = col; k <= matrix.size(); k++) {
			matrix[j][k] -= factor * matrix[col][k];
		}
	}
}
vector<vector<double>> gauss_elimination_parallel_thread(const vector<vector<double>>& augmented_matrix) {
	int size = augmented_matrix.size();
	vector<vector<double>> matrix = augmented_matrix;

	for (int i = 0; i < size; i++) {
		int pivot_row = i;
		for (int j = i + 1; j < size; j++) {
			if (abs(matrix[j][i]) > abs(matrix[pivot_row][i])) {
				pivot_row = j;
			}
		}

		if (matrix[pivot_row][i] == 0.0) {
			cerr << "Brak jednoznacznego rozwiazania" << endl;
			exit(1);
		}

		swap(matrix[i], matrix[pivot_row]);

		int num_threads = thread::hardware_concurrency();
		vector<thread> threads(num_threads);
		int n_per_thread = floor((size - i - 1) / static_cast<double>(num_threads));

		for (int t = 0; t < num_threads; ++t) {
			int start_row = i + 1 + t * n_per_thread;
			int end_row = (t == num_threads - 1) ? size : start_row + n_per_thread;
			threads[t] = thread(gauss_elimination_worker_thread, ref(matrix), start_row, end_row, i);
		}

		for (auto& t : threads) {
			t.join();
		}
	}

	vector<vector<double>> result(size, vector<double>(1));
	for (int i = size - 1; i >= 0; i--) {
		double sum = 0;
		for (int j = i + 1; j < size; j++) {
			sum += matrix[i][j] * result[j][0];
		}
		result[i][0] = (matrix[i][size] - sum) / matrix[i][i];
	}

	return result;
}

//Gauss-Jordan
vector<vector<double>> gauss_jordan_sequential(const vector<vector<double>>& augmented_matrix) {
	int size = augmented_matrix.size();
	vector<vector<double>> matrix = augmented_matrix;

	for (int i = 0; i < size; i++) {
		int maxRow = i;
		for (int k = i + 1; k < size; k++) {
			if (abs(matrix[k][i]) > abs(matrix[maxRow][i])) {
				maxRow = k;
			}
		}

		for (int k = i; k <= size; k++) {
			swap(matrix[maxRow][k], matrix[i][k]);
		}

		for (int k = 0; k < size; k++) {
			if (k != i) {
				double factor = matrix[k][i] / matrix[i][i];
				for (int j = i; j <= size; j++) {
					matrix[k][j] -= factor * matrix[i][j];
				}
			}
		}
	}

	vector<vector<double>> x(size, vector<double>(1));
	for (int i = 0; i < size; i++) {
		x[i][0] = matrix[i][size] / matrix[i][i];
	}

	return x;
}
vector<vector<double>> gauss_jordan_parallel(const vector<vector<double>>& augmented_matrix) {
	int size = augmented_matrix.size();
	vector<vector<double>> matrix = augmented_matrix;

	for (int i = 0; i < size; i++) {
		int maxRow = i;
		for (int k = i + 1; k < size; k++) {
			if (abs(matrix[k][i]) > abs(matrix[maxRow][i])) {
				maxRow = k;
			}
		}
#pragma omp parallel for
		for (int k = i; k <= size; k++) {
			swap(matrix[maxRow][k], matrix[i][k]);
		}

#pragma omp parallel for
		for (int k = 0; k < size; k++) {
			if (k != i) {
				double factor = matrix[k][i] / matrix[i][i];
				for (int j = i; j <= size; j++) {
					matrix[k][j] -= factor * matrix[i][j];
				}
			}
		}
	}

	vector<vector<double>> x(size, vector<double>(1));
	for (int i = 0; i < size; i++) {
		x[i][0] = matrix[i][size] / matrix[i][i];
	}

	return x;
}
void gauss_jordan_worker(vector<vector<double>>& matrix, int start_row, int end_row, int i) {
	for (int k = start_row; k < end_row; k++) {
		if (k != i) {
			double factor = matrix[k][i] / matrix[i][i];
			for (int j = i; j <= matrix.size(); j++) {
				matrix[k][j] -= factor * matrix[i][j];
			}
		}
	}
}
vector<vector<double>> gauss_jordan_parallel_thread(const vector<vector<double>>& augmented_matrix) {
	int size = augmented_matrix.size();
	vector<vector<double>> matrix = augmented_matrix;

	int num_threads = std::thread::hardware_concurrency();
	vector<std::thread> threads(num_threads);
	vector<int> start_rows(num_threads);
	vector<int> end_rows(num_threads);

	for (int i = 0; i < size; i++) {
		int maxRow = i;
		for (int k = i + 1; k < size; k++) {
			if (abs(matrix[k][i]) > abs(matrix[maxRow][i])) {
				maxRow = k;
			}
		}

		for (int k = i; k <= size; k++) {
			swap(matrix[maxRow][k], matrix[i][k]);
		}

		int n_per_thread = size / num_threads;

		for (int t = 0; t < num_threads; t++) {
			start_rows[t] = t * n_per_thread;
			end_rows[t] = (t == num_threads - 1) ? size : start_rows[t] + n_per_thread;
			threads[t] = std::thread(gauss_jordan_worker, std::ref(matrix), start_rows[t], end_rows[t], i);
		}

		for (auto& t : threads) {
			t.join();
		}
	}

	vector<vector<double>> x(size, vector<double>(1));
	for (int i = 0; i < size; i++) {
		x[i][0] = matrix[i][size] / matrix[i][i];
	}

	return x;
}

//Gauss-Seidel
vector<vector<double>> gauss_seidel_sequential(const vector<vector<double>>& augmented_matrix, int MAX_ITER) {
	int n = augmented_matrix.size();
	vector<double> x(n, 0);
	vector<vector<double>> result(n, vector<double>(1));
	double temp;

	for (int k = 0; k < MAX_ITER; k++) {
		for (int i = 0; i < n; i++) {
			temp = augmented_matrix[i][n];
			for (int j = 0; j < n; j++) {
				if (i != j)
					temp -= augmented_matrix[i][j] * x[j];
			}
			x[i] = temp / augmented_matrix[i][i];
		}
	}

	for (int i = 0; i < n; i++) {
		result[i][0] = x[i];
	}

	return result;
}
vector<vector<double>> gauss_seidel_parallel2(const vector<vector<double>>& augmented_matrix, int MAX_ITER) {
	int n = augmented_matrix.size();
	vector<double> x(n, 0);
	vector<vector<double>> result(n, vector<double>(1));
	double temp;

	for (int k = 0; k < MAX_ITER; k++) {
#pragma omp parallel for private(temp) schedule(static)
		for (int i = 0; i < n; i++) {
			temp = augmented_matrix[i][n];
			for (int j = 0; j < n; j++) {
				if (i != j)
					temp -= augmented_matrix[i][j] * x[j];
			}
			x[i] = temp / augmented_matrix[i][i];
		}
	}

	for (int i = 0; i < n; i++) {
		result[i][0] = x[i];
	}

	return result;
}
void gauss_seidel_parallel(vector<vector<double>>& A, vector<double>& b, vector<double>& x, int MAX_ITER, double EPSILON) {
	int n = A.size();

	for (int k = 0; k < MAX_ITER; k++) {
		double max_diff = 0.0;

#pragma omp parallel
		{
			double local_max_diff = 0.0;

#pragma omp for
			for (int i = 0; i < n; i++) {
				double temp = b[i];
				double old_xi = x[i];

				for (int j = 0; j < n; j++) {
					if (i != j) {
						temp -= A[i][j] * x[j];
					}
				}

				double new_xi = temp / A[i][i];
				x[i] = new_xi;

				double diff = std::abs(new_xi - old_xi);
				if (diff > local_max_diff) {
					local_max_diff = diff;
				}
			}

#pragma omp critical
			{
				if (local_max_diff > max_diff) {
					max_diff = local_max_diff;
				}
			}
		}

		if (max_diff < EPSILON) {
			break;
		}
	}
}
void gauss_seidel_relaxed_iteration(const vector<vector<double>>& augmented_matrix, vector<double>& x, int start, int end, double relaxation_factor, mutex& x_mutex) {
	double temp;
	for (int i = start; i < end; i++) {
		temp = augmented_matrix[i][x.size()];
		for (int j = 0; j < x.size(); j++) {
			if (i != j)
				temp -= augmented_matrix[i][j] * x[j];
		}

		double new_value = (1 - relaxation_factor) * x[i] + relaxation_factor * (temp / augmented_matrix[i][i]);
		{
			lock_guard<mutex> lock(x_mutex);
			x[i] = new_value;
		}
	}
}
vector<vector<double>> gauss_seidel_parallel_thread(const vector<vector<double>>& augmented_matrix, int MAX_ITER, double relaxation_factor) {
	int n = augmented_matrix.size();
	vector<double> x(n, 0);
	vector<vector<double>> result(n, vector<double>(1));

	mutex x_mutex;

	for (int k = 0; k < MAX_ITER; k++) {
		int num_threads = thread::hardware_concurrency();
		vector<thread> threads(num_threads);

		int n_per_thread = floor(n / static_cast<double>(num_threads));

		for (int t = 0; t < num_threads; ++t) {
			int start = t * n_per_thread;
			int end = (t == num_threads - 1) ? n : start + n_per_thread;
			threads[t] = thread(gauss_seidel_relaxed_iteration, ref(augmented_matrix), ref(x), start, end, relaxation_factor, ref(x_mutex));
		}

		for (auto& t : threads) {
			t.join();
		}
	}

	for (int i = 0; i < n; i++) {
		result[i][0] = x[i];
	}

	return result;
}

//Jacobi
vector<vector<double>> jacobi_sequential(const vector<vector<double>>& augmented_matrix, int MAX_ITER) {
	int n = augmented_matrix.size();
	vector<double> x(n, 0);
	vector<double> new_x(n);
	vector<vector<double>> result(n, vector<double>(1));
	double temp;

	for (int k = 0; k < MAX_ITER; k++) {
		for (int i = 0; i < n; i++) {
			temp = augmented_matrix[i][n];
			for (int j = 0; j < n; j++) {
				if (i != j)
					temp -= augmented_matrix[i][j] * x[j] + 1;
			}
			new_x[i] = temp / augmented_matrix[i][i];
		}
		x = new_x;
	}

	for (int i = 0; i < n; i++) {
		result[i][0] = x[i];
	}

	return result;
}
vector<vector<double>> jacobi_parallel(const vector<vector<double>>& augmented_matrix, int MAX_ITER) {
	int n = augmented_matrix.size();
	vector<double> x(n, 0);
	vector<double> new_x(n);
	vector<vector<double>> result(n, vector<double>(1));
	double temp;

	for (int k = 0; k < MAX_ITER; k++) {
#pragma omp parallel for private(temp)
		for (int i = 0; i < n; i++) {
			temp = augmented_matrix[i][n];
			for (int j = 0; j < n; j++) {
				if (i != j)
					temp -= augmented_matrix[i][j] * x[j];
			}
			new_x[i] = temp / augmented_matrix[i][i];
		}
		x = new_x;
	}

	for (int i = 0; i < n; i++) {
		result[i][0] = x[i];
	}

	return result;
}
void jacobi_iteration(const vector<vector<double>>& augmented_matrix, const vector<double>& x, vector<double>& new_x, int start, int end) {
	double temp;
	for (int i = start; i < end; i++) {
		temp = augmented_matrix[i][x.size()];
		for (int j = 0; j < x.size(); j++) {
			if (i != j)
				temp -= augmented_matrix[i][j] * x[j];
		}
		new_x[i] = temp / augmented_matrix[i][i];
	}
}
vector<vector<double>> jacobi_parallel_thread(const vector<vector<double>>& augmented_matrix, int MAX_ITER) {
	int size = augmented_matrix.size();
	vector<double> x(size, 0);
	vector<double> new_x(size);
	vector<vector<double>> result(size, vector<double>(1));

	for (int k = 0; k < MAX_ITER; k++) {
		vector<thread> threads;
		int num_threads = thread::hardware_concurrency();

		int n_per_thread = size / num_threads;
		for (int t = 0; t < num_threads; t++) {
			int start = t * n_per_thread;
			int end = (t == num_threads - 1) ? size : start + n_per_thread;
			threads.push_back(thread(jacobi_iteration, ref(augmented_matrix), ref(x), ref(new_x), start, end));
		}

		for (thread& t : threads) {
			t.join();
		}

		x = new_x;
	}

	for (int i = 0; i < size; i++) {
		result[i][0] = x[i];
	}

	return result;
}
//vector<vector<double>> jacobi_sequential(const vector<vector<double>>& augmented_matrix, int MAX_ITER) {
//	int n = augmented_matrix.size();
//	vector<double> x(n, 0);
//	vector<double> new_x(n);
//	vector<vector<double>> result(n, vector<double>(1));
//	double temp;
//
//	for (int k = 0; k < MAX_ITER; k++) {
//		cout << "Iteracja " << k + 1 << ":" << endl;
//		for (int i = 0; i < n; i++) {
//			temp = augmented_matrix[i][n];
//			for (int j = 0; j < n; j++) {
//				if (i != j)
//					temp -= augmented_matrix[i][j] * x[j];
//			}
//			new_x[i] = temp / augmented_matrix[i][i];
//		}
//
//		// Wypisanie wartości wektora x na bieżącym kroku
//		for (int i = 0; i < n; i++) {
//			cout << "x[" << i << "] = " << fixed << setprecision(6) << new_x[i] << endl;
//		}
//		cout << endl;
//
//		x = new_x;
//	}
//
//	for (int i = 0; i < n; i++) {
//		result[i][0] = x[i];
//	}
//
//	return result;
//}

//Tests
void calculate_averages(int SIZE, int TEST_RUNS, double totalSequentialTimeGauss,
	double totalParallelTimeGauss, double totalParallelTimeGaussThread,
	double totalSequentialTimeJordan, double totalParallelTimeJordan, double totalParallelTimeJordanThread,
	double totalSequentialTimeGaussSeidel, double totalParallelTimeGaussSeidel, double totalParallelTimeGaussSeidelThread,
	double totalSequentialTimeJacobi, double totalParallelTimeJacobi, double totalParallelTimeJacobiThread) {
	cout << "Results for SIZE: " << SIZE << endl;
	cout << "Average sequential Gauss elimination time: " << totalSequentialTimeGauss / TEST_RUNS << " seconds" << endl;
	cout << "Average parallel Gauss elimination time(OPENMP): " << totalParallelTimeGauss / TEST_RUNS << " seconds" << endl;
	cout << "Average parallel Gauss elimination time(THREAD): " << totalParallelTimeGaussThread / TEST_RUNS << " seconds" << endl;
	cout << "Average sequential Gauss Jordan elimination time: " << totalSequentialTimeJordan / TEST_RUNS << " seconds" << endl;
	cout << "Average parallel Gauss Jordan elimination time(OPENMP): " << totalParallelTimeJordan / TEST_RUNS << " seconds" << endl;
	cout << "Average parallel Gauss Jordan elimination time(THREAD): " << totalParallelTimeJordanThread / TEST_RUNS << " seconds" << endl;
	cout << "Average sequential Gauss Seidel time: " << totalSequentialTimeGaussSeidel / TEST_RUNS << " seconds" << endl;
	cout << "Average parallel Gauss Seidel time(OPENMP): " << totalParallelTimeGaussSeidel / TEST_RUNS << " seconds" << endl;
	cout << "Average parallel Gauss Seidel time(THREAD): " << totalParallelTimeGaussSeidelThread / TEST_RUNS << " seconds" << endl;
	cout << "Average sequential Jacobi time: " << totalSequentialTimeJacobi / TEST_RUNS << " seconds" << endl;
	cout << "Average parallel Jacobi time(OPENMP): " << totalParallelTimeJacobi / TEST_RUNS << " seconds" << endl;
	cout << "Average parallel Jacobi time(THREAD): " << totalParallelTimeJacobiThread / TEST_RUNS << " seconds" << endl;
}

void calculate_averages(int n_intervals, int TEST_RUNS, double totalSequentialTimeMonteCarlo, double totalParallelTimeMonteCarlo, double totalParallelThreadTimeMonteCarlo,
	double totalSequentialSimpson, double totalParallelSimpson, double totalParallelThreadSimpson) {
	cout << "Results for " << n_intervals << " intervals:" << endl;
	cout << "Average sequential Monte Carlo integration time: " << totalSequentialTimeMonteCarlo / TEST_RUNS << " seconds" << endl;
	cout << "Average parallel Monte Carlo integration time (OPENMP): " << totalParallelTimeMonteCarlo / TEST_RUNS << " seconds" << endl;
	cout << "Average parallel Monte Carlo integration time (THREAD): " << totalParallelThreadTimeMonteCarlo / TEST_RUNS << " seconds" << endl;
	cout << "Average sequential Simpson integration result: " << totalSequentialSimpson / TEST_RUNS << endl;
	cout << "Average parallel Simpson integration result (OPENMP): " << totalParallelSimpson / TEST_RUNS << endl;
	cout << "Average parallel Simpson integration result (THREAD): " << totalParallelThreadSimpson / TEST_RUNS << endl;
}

void run_test(const vector<vector<double>>& A, const vector<vector<double>>& B, int SIZE, double& seqGauss,
	double& parGauss, double& parGaussThread, double& seqJordan, double& parJordan, double& parJordanThread,
	double& seqGaussSeidel, double& parGaussSeidel, double& parGaussSeidelThread,
	double& seqJacobi, double& parJacobi, double& parJacobiThread) {
	vector<vector<double>> X;
	int MAX_ITER = 50;

	//print_matrix(A, SIZE);
	double start_seq_gauss = omp_get_wtime();
	X = gauss_elimination_sequential(A);
	double end_seq_gauss = omp_get_wtime();
	//print_vector(X);

	//print_matrix(A, SIZE);
	double start_par_gauss = omp_get_wtime();
	X = gauss_elimination_parallel(A);
	double end_par_gauss = omp_get_wtime();
	//print_vector(X);

	//print_matrix(A, SIZE);
	double start_par_gauss_thread = omp_get_wtime();
	X = gauss_elimination_parallel_thread(A);
	double end_par_gauss_thread = omp_get_wtime();
	//print_vector(X);

	//print_matrix(A, SIZE);
	double start_seq_jordan = omp_get_wtime();
	X = gauss_jordan_sequential(A);
	double end_seq_jordan = omp_get_wtime();
	//print_vector(X);

	//print_matrix(A, SIZE);
	double start_par_jordan = omp_get_wtime();
	X = gauss_jordan_parallel(A);
	double end_par_jordan = omp_get_wtime();
	//print_vector(X);

	//print_matrix(A, SIZE);
	double start_par_jordan_thread = omp_get_wtime();
	X = gauss_jordan_parallel_thread(A);
	double end_par_jordan_thread = omp_get_wtime();
	//print_vector(X);

	seqGauss = end_seq_gauss - start_seq_gauss;
	parGauss = end_par_gauss - start_par_gauss;
	parGaussThread = end_par_gauss_thread - start_par_gauss_thread;
	seqJordan = end_seq_jordan - start_seq_jordan;
	parJordan = end_par_jordan - start_par_jordan;
	parJordanThread = end_par_jordan_thread - start_par_jordan_thread;

	//print_matrix(B, SIZE);
	double start_seq_gauss_seidel = omp_get_wtime();
	X = gauss_seidel_sequential(B, MAX_ITER);
	double end_seq_gauss_seidel = omp_get_wtime();
	seqGaussSeidel = end_seq_gauss_seidel - start_seq_gauss_seidel;
	print_vector(X);

	//print_matrix(B, SIZE);
	double start_par_gauss_seidel = omp_get_wtime();
	X = gauss_seidel_parallel2(B, MAX_ITER);
	double end_par_gauss_seidel = omp_get_wtime();
	parGaussSeidel = end_par_gauss_seidel - start_par_gauss_seidel;
	print_vector(X);

	//print_matrix(B, SIZE);
	double start_par_gauss_seidel_thread = omp_get_wtime();
	X = gauss_seidel_parallel_thread(B, MAX_ITER, 1.5);
	double end_par_gauss_seidel_thread = omp_get_wtime();
	parGaussSeidelThread = end_par_gauss_seidel_thread - start_par_gauss_seidel_thread;
	print_vector(X);

	double start_seq_jacobi = omp_get_wtime();
	X = jacobi_sequential(B, MAX_ITER);
	double end_seq_jacobi = omp_get_wtime();
	//print_vector(X);

	double start_par_jacobi = omp_get_wtime();
	X = jacobi_parallel(B, MAX_ITER);
	double end_par_jacobi = omp_get_wtime();
	//print_vector(X);

	//print_matrix(B, SIZE);
	double start_par_jacobi_thread = omp_get_wtime();
	X = jacobi_parallel_thread(B, MAX_ITER);
	double end_par_jacobi_thread = omp_get_wtime();
	//print_vector(X);

	seqJacobi = end_seq_jacobi - start_seq_jacobi;
	parJacobi = end_par_jacobi - start_par_jacobi;
	parJacobiThread = end_par_jacobi_thread - start_par_jacobi_thread;
}

void run_test(int n_intervals, double& seqMonteCarlo, double& parMonteCarlo, double& parMonteCarloThread
	, double& seqSimpson, double& parSimpson, double& parSimpsonThread) {
	random_device rd;
	mt19937 gen(rd());
	double a = 0.0;
	double b = 2.0;
	double monteCarloResult, simpsonResult;

	double start_seq_MonteCarlo = omp_get_wtime();
	monteCarloResult = monte_carlo_integration(a, b, n_intervals, gen);
	double end_seq_MonteCarlo = omp_get_wtime();
	seqMonteCarlo = end_seq_MonteCarlo - start_seq_MonteCarlo;
	cout << monteCarloResult;
	cout << endl;

	double start_par_MonteCarlo = omp_get_wtime();
	monteCarloResult = monte_carlo_integration_parallel(a, b, n_intervals, gen);
	double end_par_MonteCarlo = omp_get_wtime();
	parMonteCarlo = end_par_MonteCarlo - start_par_MonteCarlo;
	cout << monteCarloResult;
	cout << endl;

	double start_par_MonteCarlo_thread = omp_get_wtime();
	monteCarloResult = monte_carlo_integration_parallel_thread(a, b, n_intervals, gen);
	double end_par_MonteCarlo_thread = omp_get_wtime();
	parMonteCarloThread = end_par_MonteCarlo_thread - start_par_MonteCarlo_thread;
	cout << monteCarloResult;
	cout << endl;

	double start_seq_Simpson = omp_get_wtime();
	simpsonResult = simpson_sequential(a, b, n_intervals);
	double end_seq_Simpson = omp_get_wtime();
	seqSimpson = end_seq_Simpson - start_seq_Simpson;
	cout << simpsonResult;
	cout << endl;

	double start_par_Simpson = omp_get_wtime();
	simpsonResult = simpson_parallel(a, b, n_intervals);
	double end_par_Simpson = omp_get_wtime();
	parSimpson = end_par_Simpson - start_par_Simpson;
	cout << simpsonResult;
	cout << endl;

	double start_par_Simpson_thread = omp_get_wtime();
	simpsonResult = simpson_parallel_thread(a, b, n_intervals);
	double end_par_Simpson_thread = omp_get_wtime();
	parSimpsonThread = end_par_Simpson_thread - start_par_Simpson_thread;
	cout << simpsonResult;
	cout << endl;
}

void export_to_csv(const string& filename, const vector<int>& sizes,
	const vector<double>& seqGauss, const vector<double>& parGauss, const vector<double>& parGaussThread,
	const vector<double>& seqJordan, const vector<double>& parJordan, const vector<double>& parJordanThread, const vector<double>& seqGaussSeidel,
	const vector<double>& parGaussSeidel, const vector<double>& parGaussSeidelThread, const vector<double>& seqJacobi, const vector<double>& parJacobi,
	const vector<double>& parJacobiThread) {
	ofstream file(filename);

	if (!file.is_open()) {
		cerr << "Cannot open the file: " << filename << endl;
		return;
	}

	file << "Size,SequentialGauss,ParallelGauss(OPENMP),ParallelGauss(THREAD),SequentialJordan,ParallelJordan,ParallelJordan(THREAD),SequentialGaussSeidel,ParallelGaussSeidel,ParallelGaussSeidel(THREAD),SequentialJacobi,ParallelJacobi,ParallelJacobi(THREAD)\n";

	// Write the data
	for (size_t i = 0; i < sizes.size(); i++) {
		file << sizes[i] << "," << seqGauss[i] << "," << parGauss[i] << "," << parGaussThread[i] << "," << seqJordan[i] << "," << parJordan[i] << "," << parJordanThread[i] << "," << seqGaussSeidel[i] << "," << parGaussSeidel[i] << "," << parGaussSeidelThread[i] << "," << seqJacobi[i] << "," << parJacobi[i] << "," << parJacobiThread[i] << "\n";
	}

	file.close();
}

void export_to_csv(const string& filename, const vector<int>& intervals,
	const vector<double>& seqMonteCarlo, const vector<double>& parMonteCarlo, const vector<double>& parMonteCarloThread,
	const vector<double>& seqSimpson, const vector<double>& parSimpson, const vector<double>& parSimpsonThread) {
	ofstream file(filename);

	if (!file.is_open()) {
		cerr << "Cannot open the file: " << filename << endl;
		return;
	}

	file << "Intervals,SequentialMonteCarlo,ParallelMonteCarlo(OPENMP),ParallelMonteCarlo(THREAD),SequentialSimpson,ParallelSimpson(OPENMP),ParallelSimpson(THREAD)\n";

	// Write the data
	for (size_t i = 0; i < intervals.size(); i++) {
		file << intervals[i] << "," << seqMonteCarlo[i] << "," << parMonteCarlo[i] << "," << parMonteCarloThread[i] << ","
			<< seqSimpson[i] << "," << parSimpson[i] << "," << parSimpsonThread[i] << "\n";
	}

	file.close();
}

int main() {
	int TEST_RUNS = 1;
	vector<int> sizes = { 2000 };
	vector<int> integration_intervals = { 100000000 };

	vector<double> seqGaussResults, parGaussResults,
		parGaussThreadResults, seqJordanResults,
		parJordanResults, parJordanThreadResults, seqGaussSeidelResults, parGaussSeidelResults,
		parGaussSeidelThreadResults, seqJacobiResults, parJacobiResults, parJacobiThreadResults,
		seqMonteCarloResults, parMonteCarloResults, parMonteCarloThreadResults,
		seqSimpsonResults, parSimpsonResults, parSimpsonThreadResults;

	for (int SIZE : sizes) {
		double totalSequentialTimeGauss = 0, totalParallelTimeGauss = 0, totalParallelTimeGaussThread = 0;
		double totalSequentialTimeJordan = 0, totalParallelTimeJordan = 0, totalParallelTimeJordanThread = 0;
		double totalSequentialTimeGaussSeidel = 0, totalParallelTimeGaussSeidel = 0, totalParallelTimeGaussSeidelThread = 0;
		double totalSequentialTimeJacobi = 0, totalParallelTimeJacobi = 0, totalParallelTimeJacobiThread = 0;

		for (int testRun = 0; testRun < TEST_RUNS; testRun++) {
			vector<vector<double>> A = generate_matrix(SIZE);
			vector<vector<double>> B = generate_matrix2(SIZE);
			double seqGauss, parGauss, seqJordan, parJordan, parJordanThread, parGaussThread,
				seqGaussSeidel, parGaussSeidel, parGaussSeidelThread, seqJacobi, parJacobi, parJacobiThread;
			run_test(A, B, SIZE, seqGauss, parGauss,
				parGaussThread, seqJordan, parJordan, parJordanThread,
				seqGaussSeidel, parGaussSeidel, parGaussSeidelThread,
				seqJacobi, parJacobi, parJacobiThread);

			totalSequentialTimeGauss += seqGauss;
			totalParallelTimeGauss += parGauss;
			totalParallelTimeGaussThread += parGaussThread;

			totalSequentialTimeJordan += seqJordan;
			totalParallelTimeJordan += parJordan;
			totalParallelTimeJordanThread += parJordanThread;

			totalSequentialTimeGaussSeidel += seqGaussSeidel;
			totalParallelTimeGaussSeidel += parGaussSeidel;
			totalParallelTimeGaussSeidelThread += parGaussSeidelThread;

			totalSequentialTimeJacobi += seqJacobi;
			totalParallelTimeJacobi += parJacobi;
			totalParallelTimeJacobiThread += parJacobiThread;
		}

		calculate_averages(SIZE, TEST_RUNS, totalSequentialTimeGauss, totalParallelTimeGauss,
			totalParallelTimeGaussThread, totalSequentialTimeJordan, totalParallelTimeJordan, totalParallelTimeJordanThread,
			totalSequentialTimeGaussSeidel, totalParallelTimeGaussSeidel, totalParallelTimeGaussSeidelThread,
			totalSequentialTimeJacobi, totalParallelTimeJacobi, totalParallelTimeJacobiThread);

		seqGaussResults.push_back(totalSequentialTimeGauss / TEST_RUNS);
		parGaussResults.push_back(totalParallelTimeGauss / TEST_RUNS);
		parGaussThreadResults.push_back(totalParallelTimeGaussThread / TEST_RUNS);

		seqJordanResults.push_back(totalSequentialTimeJordan / TEST_RUNS);
		parJordanResults.push_back(totalParallelTimeJordan / TEST_RUNS);
		parJordanThreadResults.push_back(totalParallelTimeJordanThread / TEST_RUNS);

		seqGaussSeidelResults.push_back(totalSequentialTimeGaussSeidel / TEST_RUNS);
		parGaussSeidelResults.push_back(totalParallelTimeGaussSeidel / TEST_RUNS);
		parGaussSeidelThreadResults.push_back(totalParallelTimeGaussSeidelThread / TEST_RUNS);

		seqJacobiResults.push_back(totalSequentialTimeJacobi / TEST_RUNS);
		parJacobiResults.push_back(totalParallelTimeJacobi / TEST_RUNS);
		parJacobiThreadResults.push_back(totalParallelTimeJacobiThread / TEST_RUNS);
	}
	for (int n : integration_intervals) {
		double totalMonteCarloTime = 0;
		double totalParMonteCarloTime = 0;
		double totalParMonteCarloThreadTime = 0;
		double totalSimpsonTime = 0;
		double totalParSimpsonTime = 0;
		double totalParSimpsonThreadTime = 0;

		for (int testRun = 0; testRun < TEST_RUNS; testRun++) {
			double monteCarloTime, parMonteCarloTime, parMonteCarloThreadTime;
			double simpsonTime, parSimpsonTime, parSimpsonThreadTime;
			run_test(n, monteCarloTime, parMonteCarloTime, parMonteCarloThreadTime, simpsonTime, parSimpsonTime, parSimpsonThreadTime);
			totalMonteCarloTime += monteCarloTime;
			totalParMonteCarloTime += parMonteCarloTime;
			totalParMonteCarloThreadTime += parMonteCarloThreadTime;
			totalSimpsonTime += simpsonTime;
			totalParSimpsonTime += parSimpsonTime;
			totalParSimpsonThreadTime += parSimpsonThreadTime;
		}

		calculate_averages(n, TEST_RUNS, totalMonteCarloTime, totalParMonteCarloTime, totalParMonteCarloThreadTime,
			totalSimpsonTime, totalParSimpsonTime, totalParSimpsonThreadTime);

		seqMonteCarloResults.push_back(totalMonteCarloTime / TEST_RUNS);
		parMonteCarloResults.push_back(totalParMonteCarloTime / TEST_RUNS);
		parMonteCarloThreadResults.push_back(totalParMonteCarloThreadTime / TEST_RUNS);

		seqSimpsonResults.push_back(totalSimpsonTime / TEST_RUNS);
		parSimpsonResults.push_back(totalParSimpsonTime / TEST_RUNS);
		parSimpsonThreadResults.push_back(totalParSimpsonThreadTime / TEST_RUNS);
	}

	string filename = "C:\\Users\\macie\\source\\repos\\projekt_wyklad\\results.csv";
	export_to_csv(filename, sizes, seqGaussResults, parGaussResults,
		parGaussThreadResults, seqJordanResults, parJordanResults, parJordanThreadResults,
		seqGaussSeidelResults, parGaussSeidelResults, parGaussSeidelThreadResults,
		seqJacobiResults, parJacobiResults, parJacobiThreadResults);

	string filename2 = "C:\\Users\\macie\\source\\repos\\projekt_wyklad\\results2.csv";
	export_to_csv(filename2, integration_intervals, seqMonteCarloResults, parMonteCarloResults, parMonteCarloThreadResults,
		seqSimpsonResults, parSimpsonResults, parSimpsonThreadResults);

	return 0;
}