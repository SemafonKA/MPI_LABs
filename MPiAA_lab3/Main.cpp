#include "mpi.h"
#include <iostream>
#include <windows.h>
#include <omp.h>

const int max_val = 500;
const int min_val = -500;

void make_array(int* arr, int sz) {
	for (int i = 0; i < sz; ++i)
		arr[i] = rand() % (max_val - min_val + 1) + min_val;
}

double get_normal(int* arr, int sz) {
	double ans = 0;

#pragma omp parallel for reduction(+ : ans)
	for (int i = 0; i < sz; ++i)
		ans += arr[i] * arr[i];

	return sqrt(ans);
}

int main(int argc, char** argv)
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0) {
		std::cout << "Size of MPI system is " << size << std::endl << std::endl;

#pragma omp parallel for num_threads(size)
		for (int i = 1; i < size; ++i) {
			double normal = 0;
			MPI_Request  req;

#pragma omp critical
			MPI_Irecv(&normal, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &req);

			int flag = 0;
			MPI_Status stat;
			while (!flag) {
#pragma omp critical
				MPI_Test(&req, &flag, &stat);
			}

#pragma omp critical
			std::cout << "Received rank is " << i << ", normal is " << normal << std::endl;
		}

		std::cout << std::endl << "Program is done" << std::endl;
	}
	else {
		srand(time(0) + rank * 28'228);
		const int vector_size = 100'000;
		const int sleep_time = 10'000;

		int* vector = new int[vector_size];

		make_array(vector, vector_size);
		double normal = get_normal(vector, vector_size);

		if (rank == 1) Sleep(sleep_time);

		MPI_Send(&normal, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

		delete[] vector;
	}

	MPI_Finalize();
	return 0;
}