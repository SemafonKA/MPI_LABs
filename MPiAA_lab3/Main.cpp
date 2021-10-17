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

		auto normals = new std::pair<MPI_Request, double>[size - 1];
		for (int i = 1; i < size; ++i) {
			MPI_Irecv(&(normals[i-1].second), 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &(normals[i-1].first));
		}

		int couted_count = 1;
		bool* couted = new bool[size - 1];
		for (int i = 0; i < size - 1; ++i) couted[i] = false;

		while (couted_count != size) {
			for (int i = 1; i < size; ++i) {
				if (couted[i - 1]) continue; // Если i-ый элемент уже был выведен, то скипаем

				int flag = 0;
            MPI_Status stat;
				MPI_Test(&(normals[i - 1].first), &flag, &stat);

				if (flag) { // Если i-ый элемент был получен
					std::cout << "Received rank is " << i << ", normal is " << normals[i-1].second << std::endl;
					++couted_count;
					couted[i - 1] = true;
				}
			}
		}

		delete[] couted;
		delete[] normals;

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