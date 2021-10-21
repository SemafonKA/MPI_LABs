#include "mpi.h"
#include <iostream>

using namespace std;

/// <summary>
/// Линейная функция перемножения двух матриц произвольного размера
/// <para>Главное условие - совпадение размеров [m] и [o]</para>
/// </summary>
/// <param name="c"> - матрица, в которой будет хранится результат перемножения</param>
/// <param name="n"> - Число строк первой матрицы</param>
/// <param name="m"> - Число столбцов первой матрицы</param>
/// <param name="o"> - Число строк второй матрицы</param>
/// <param name="p"> - Число столбцов второй матрицы</param>
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

/// <summary>
/// Функция вывода вектора на экран в виде [a0, a1, .., an], n = size - 1
/// </summary>
/// <typeparam name="T"> - тип данных вектора (должен поддерживаться cout)</typeparam>
/// <param name="vec"> - указатель на начало вектора,</param>
/// <param name="size"> - размер вектора,</param>
/// <param name="newLine"> - необязательный параметр, вставлять ли перенос строки в конце вывода</param>
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

/// <summary>
/// Удаление вектора по ненулевому указателю и зануление указателя.
/// Если передан nullptr, то ничего не происходит
/// </summary>
/// <typeparam name="T"> - тип данных вектора</typeparam>
/// <param name="vec"> - указатель на начало вектора</param>
template<typename T>
inline void deleteNotNull(T*& vec) {
   if (vec != nullptr) {
      delete[] vec;
      vec = nullptr;
   }
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
      cout << endl;

      cout << "Enter vector: " << endl;
      for (int i = 0; i < m; ++i) {
      	cin >> vector[i];
      }
      cout << endl;

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

   // Делимся вектором со всеми процессами
   if (rank != root) {
   	vector = new double[m];
   }
   MPI_Bcast(vector, m, MPI_DOUBLE, root, MPI_COMM_WORLD);

   // Говорим, сколько элементов матрицы достанется каждому процессу
   // по принципу: все процессы получают равное число строк, идущих друг за другом.
   // Если строк больше, чем число процессов, то лишние строки выдаются процессам по одной, начиная с первого.
   // Если строк меньше, чем число процессов, то последние процессы останутся без строк.
   int* splitArray = new int[size]; // Массив, где [i] элемент показывает, сколько элементов будет передано i-му процессу
   if (rank == root) {
   	int mainSplit = n / size;     // Равное деление на части
   	int additionSplit = n % size; // Оставшиеся от деления на части строки
   	for (int i = 0; i < size; ++i) {
   		splitArray[i] = mainSplit * m;
   		if (additionSplit > 0) {
   			splitArray[i] += m;
   			--additionSplit;
   		}
   	}
   }
   // Раздаём массив всем процессам
   MPI_Bcast(splitArray, size, MPI_INTEGER, root, MPI_COMM_WORLD);

   // Второй массив распределения, нужен для того, чтобы обеспечивать лёгкий сдвиг указателя.
   // [i] элемент показывает, начиная с какого места в матрице начнутся строки для i-го процесса
   int* displs = nullptr; 
   if (rank == root) {
      displs = new int[size];
      displs[0] = 0;
      for (int i = 1; i < size; ++i) {
         displs[i] = displs[i - 1] + splitArray[i - 1];
      }
   }
   // Каждому процессу создаём массив под выделенные для текущего процесса строки
   double* splittedMatrix = new double[splitArray[rank]];

   // Разделяем между ВСЕМИ процессами матрицу [martix] (она хранится только в [0] процессе, остальным она не нужна), 
   // распределяем их по [splitArray] и [displs] и записываем в [splittedMatrix] (у каждого процесса своя) 
   MPI_Scatterv(matrix, splitArray, displs, MPI_DOUBLE, splittedMatrix, splitArray[rank], MPI_DOUBLE, root, MPI_COMM_WORLD);

   // Здесь сохраним результат работы каждого процесса (перемножения части строк на вектор)
   int partOfResultSize = splitArray[rank] / m;
   double* partOfResult = new double[partOfResultSize];
   matrix_mult(partOfResult, partOfResultSize, m, m, 1, splittedMatrix, vector);

   // После произведения строк на вектор, каждые [m] столбцов строки превратились в одну, поэтому просто делим на [m]
   double* resultVec = nullptr;  // Вектор с результатом умножения, нужен только для главного процесса
   if (rank == root) {
      for (int i = 0; i < size; ++i) {
         splitArray[i] /= m;
         displs[i] /= m;
      }

      resultVec = new double[n];
   }
   // Склеиваем результаты СО ВСЕХ процессов в процесс [root] в массив [resultVec], распределяем по [splitArray] и [displs],
   // берём их из [partOfResult]
   MPI_Gatherv(partOfResult, partOfResultSize, MPI_DOUBLE, resultVec, splitArray, displs, MPI_DOUBLE, root, MPI_COMM_WORLD);

   // Выводим в терминал
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