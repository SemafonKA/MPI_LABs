#include "mpi.h"
#include <iostream>

using namespace std;

/// <summary>
/// Линейная функция перемножения двух матриц произвольного размера
/// <para>Главное условие - совпадение размеров [m] и [o]</para>
/// </summary>
/// <param name="c"> - матрица, в которой будет хранится результат перемножения</param>
/// <param name="n"> - Число строк первой матрицы</param>
/// <param name="m"> - Число столбцов первой матрциы</param>
/// <param name="o"> - Число строк второй матрицы</param>
/// <param name="p"> - Число столбцов второй матрциы</param>
/// <param name="x"> - первая матрица</param>
/// <param name="y"> - вторая матрица</param>
void matrix_mult(double* c, int n, int m, int o, int p, double* x, double* y)
{
   for (int i = 0; i < n; ++i) {
      for (int j = 0; j < p; ++j) {
         c[i * p + j] = 0;
         for (int k = 0; k < m; ++k)
            c[i * p + j] += x[i * m + k] * y[k * p + j];
      }
   }
}

template<typename T>
void vector_print(const T* vec, const int size, const bool newLine = true) {
   cout << "[";
   for (int i = 0; i < size; ++i) {
      cout << vec[i];
      if (i != size - 1)
         cout << ", ";
   }
   cout << "]";

   if (newLine) cout << endl;
}

template<typename T>
inline void deleteNotNull(T*& vec) {
   if (vec != nullptr || vec != NULL) {
      delete[] vec;
   }
   vec = nullptr;
}

int main(int argc, char** argv)
{
   int rank, size;
   const int root = 0;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   int n = 0, m = 0;
   double* matrix = nullptr, * vector = nullptr;

   // Получаем нулевым процессом матрицу и вектор
   if (rank == root) {
      cout << "Enter size of matrix: " << endl
         << "[n]: ";
      cin >> n;
      cout << "[m]: ";
      cin >> m;
      cout << endl 
         << "Size of vector: " << endl
         << "[m]: " << m << endl << endl;

      matrix = new double[n * m];
      vector = new double[m];

      cout << "Enter matrix: " << endl;
      for (int i = 0; i < n * m; ++i) {
      	cin >> matrix[i];
      }	
      
      cout << "Enter vector: " << endl;
      for (int i = 0; i < m; ++i) {
      	cin >> vector[i];
      }

      // Печатаем результат линейного перемножения матрицы на вектор для проверки правильности
      double* linearResult = new double[n];
      matrix_mult(linearResult, n, m, m, 1, matrix, vector);
      cout << "Default linear result is: ";
      vector_print(linearResult, n);
      delete[] linearResult;
   }

   // Делимся размерами матрицы и вектора со всеми процессами
   MPI_Bcast(&n, 1, MPI_INTEGER, root, MPI_COMM_WORLD);
   MPI_Bcast(&m, 1, MPI_INTEGER, root, MPI_COMM_WORLD);

   if (rank != root) {
   	vector = new double[m];
   }

   // Делимся вектором со всеми процессами
   MPI_Bcast(vector, m, MPI_DOUBLE, root, MPI_COMM_WORLD);

   int* splitArray = new int[size];
   if (rank == root) {
   	int mainSplit = n / size;
   	int additionSplit = n % size;
   	for (int i = 0; i < size; ++i) {
   		splitArray[i] = mainSplit * m;
   		if (additionSplit > 0) {
   			splitArray[i] += m;
   			--additionSplit;
   		}
   	}
   }
   MPI_Bcast(splitArray, size, MPI_INTEGER, root, MPI_COMM_WORLD);

   int* displs = new int[size];
   if (rank == root) {
      displs[0] = 0;
      for (int i = 1; i < size; ++i) {
         displs[i] = displs[i - 1] + splitArray[i - 1];
      }
   }
   // Делаем с запасом в 1 элемент, чтобы избежать возможных ошибок
   double* splittedMatrix = new double[splitArray[rank]];

   MPI_Scatterv(matrix, splitArray, displs, MPI_DOUBLE, splittedMatrix, splitArray[rank], MPI_DOUBLE, root, MPI_COMM_WORLD);

   int partOfResultSize = splitArray[rank] / m;
   double* partOfResult = new double[partOfResultSize];
   matrix_mult(partOfResult, partOfResultSize, m, m, 1, splittedMatrix, vector);

   double* resultVec = nullptr;
   if (rank == root) {
      for (int i = 0; i < size; ++i) {
         splitArray[i] /= m;
         displs[i] /= m;
      }

      resultVec = new double[n];
   }

   MPI_Gatherv(partOfResult, partOfResultSize, MPI_DOUBLE, resultVec, splitArray, displs, MPI_DOUBLE, root, MPI_COMM_WORLD);

   if (rank == root) {
      cout << "Result is: ";
      vector_print(resultVec, n);
   }

   // Удаляем все массивы, если они существуют в этом процессе
   deleteNotNull(matrix);
   deleteNotNull(vector);
   deleteNotNull(splitArray);
   deleteNotNull(displs);
   deleteNotNull(splittedMatrix);
   deleteNotNull(partOfResult);
   deleteNotNull(resultVec);

   MPI_Finalize();
   return 0;
}