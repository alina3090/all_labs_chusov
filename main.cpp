#include <complex>
#include <iostream>
#include <numbers>
#include <random>
#include <thread>
#include <vector>
#include <omp.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <immintrin.h>  // Заголовок для AVX/AVX2 интринсиков (Intel/MSVC/GCC)

using namespace std;

/**
 * Транспонирование матрицы с использованием AVX2
 *
 * @param out - выходная транспонированная матрица
 * @param in - входная матрица
 * @param rows - количество строк входной матрицы
 * @param cols - количество столбцов входной матрицы
 *
 * Оптимизация: обрабатывает блоки 4x4 за раз используя 256-битные регистры AVX2
 */
void transposeAVX2(double* out, const double* in, size_t rows, size_t cols)
{
    const size_t blockSize = 4; // 4 double = 256 бит (размер AVX2 регистра)

    size_t i, j;
    // Основной цикл: обрабатываем блоки 4x4
    for (i = 0; i + blockSize <= rows; i += blockSize) {
        for (j = 0; j + blockSize <= cols; j += blockSize) {

            // === Загружаем блок 4×4 из входной матрицы ===
            // Каждая строка загружается как вектор из 4 double
            __m256d row0 = _mm256_loadu_pd(&in[(i + 0) * cols + j]); // a0 a1 a2 a3
            __m256d row1 = _mm256_loadu_pd(&in[(i + 1) * cols + j]); // b0 b1 b2 b3
            __m256d row2 = _mm256_loadu_pd(&in[(i + 2) * cols + j]); // c0 c1 c2 c3
            __m256d row3 = _mm256_loadu_pd(&in[(i + 3) * cols + j]); // d0 d1 d2 d3

            // === Разбиваем и переставляем пары строк ===
            // unpacklo берет младшие половины: [a0,a1] и [b0,b1] -> [a0,b0,a1,b1]
            __m256d t0 = _mm256_unpacklo_pd(row0, row1); // a0 b0 a1 b1
            __m256d t1 = _mm256_unpackhi_pd(row0, row1); // a2 b2 a3 b3
            __m256d t2 = _mm256_unpacklo_pd(row2, row3); // c0 d0 c1 d1
            __m256d t3 = _mm256_unpackhi_pd(row2, row3); // c2 d2 c3 d3

            // === Собираем верхнюю и нижнюю части блока ===
            // permute2f128 переставляет 128-битные половины регистров
            // 0x20 = 0010 0000 - берем нижнюю половину t0 и нижнюю половину t2
            __m256d r0 = _mm256_permute2f128_pd(t0, t2, 0x20); // a0 b0 c0 d0
            __m256d r1 = _mm256_permute2f128_pd(t1, t3, 0x20); // a2 b2 c2 d2
            __m256d r2 = _mm256_permute2f128_pd(t0, t2, 0x31); // a1 b1 c1 d1
            __m256d r3 = _mm256_permute2f128_pd(t1, t3, 0x31); // a3 b3 c3 d3

            // === Сохраняем транспонированный блок ===
            // Теперь столбцы стали строками
            _mm256_storeu_pd(&out[(j + 0) * rows + i], r0);
            _mm256_storeu_pd(&out[(j + 1) * rows + i], r1); 
            _mm256_storeu_pd(&out[(j + 2) * rows + i], r2); 
            _mm256_storeu_pd(&out[(j + 3) * rows + i], r3); 
        }

        // === Хвостовые столбцы (если cols не кратно 4) ===
        // Обрабатываем оставшиеся столбцы скалярно
        for (; j < cols; ++j) {
            for (size_t ii = 0; ii < blockSize && (i + ii) < rows; ++ii)
                out[j * rows + (i + ii)] = in[(i + ii) * cols + j];
        }
    }

    // === Хвостовые строки (если rows не кратно 4) ===
    // Обрабатываем оставшиеся строки скалярно
    for (; i < rows; ++i) {
        for (j = 0; j < cols; ++j)
            out[j * rows + i] = in[i * cols + j];
    }
}

/**
 * Вывод матрицы в консоль
 */
void matrixPrint(double* matrix, size_t rows, size_t cols) {
    cout << "Matrix:\n";
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++)
            cout << matrix[r * cols + c] << ' ';
        cout << endl;
    }
}

/**
 * Обычное скалярное умножение матриц (O(n³))
 *
 * @param result - результирующая матрица размером rowsA × colsB
 * @param matA - первая матрица размером rowsA × sharedDim
 * @param matB - вторая матрица размером sharedDim × colsB
 */
void matrixMul(double* result, const double* matA, const double* matB,
    size_t sharedDim, size_t rowsA, size_t colsB) {
    for (size_t r1 = 0; r1 < rowsA; r1++)
        for (size_t c2 = 0; c2 < colsB; c2++) {
            double accum = 0;
            // Скалярное произведение строки A и столбца B
            for (size_t i = 0; i < sharedDim; i++)
                accum += matA[r1 * sharedDim + i] * matB[i * colsB + c2];
            result[r1 * colsB + c2] = accum;
        }
}

/**
 * Умножение матриц с AVX2 оптимизацией
 *
 * Оптимизация:
 * 1. Транспонируем матрицу B для последовательного доступа к памяти
 * 2. Используем SIMD для параллельного умножения 4 double за раз
 */
void matrixMulAVX2(double* result, const double* matA, const double* matB,
    size_t sharedDim, size_t rowsA, size_t colsB) {

    // Создаем транспонированную копию матрицы B
    // Это нужно для последовательного доступа к памяти (кэш-эффективность)
    std::vector<double> TransposeT(colsB * sharedDim, 0.0);

    // Используем AVX2 транспонирование только если размеры кратны 4
    if (sharedDim % 4 == 0 && colsB % 4 == 0)
        transposeAVX2(TransposeT.data(), matB, colsB, sharedDim);
    else {
        // Иначе используем обычное скалярное транспонирование
        for (size_t r = 0; r < sharedDim; ++r)
            for (size_t c = 0; c < colsB; ++c)
                TransposeT[c * sharedDim + r] = matB[r * colsB + c];
    }

    const double* matT = TransposeT.data();
    double temp[4]; // Временный массив для извлечения данных из SIMD регистра

    // Основное умножение
    for (size_t r1 = 0; r1 < rowsA; r1++)
        for (size_t c2 = 0; c2 < colsB; c2++) {
            __m256d sum_vec = _mm256_setzero_pd(); // Обнуляем сумму (4x double)
            size_t k = 0;

            // === SIMD цикл: обрабатываем по 4 элемента ===
            for (; k + 3 < sharedDim; k += 4) {
                // Загружаем 4 элемента строки из A
                __m256d x = _mm256_loadu_pd(&matA[r1 * sharedDim + k]);
                // Загружаем 4 элемента строки из транспонированной B
                // (это соответствует столбцу в исходной B)
                __m256d y = _mm256_loadu_pd(&matT[c2 * sharedDim + k]);
                // Умножаем и добавляем к сумме
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(x, y));
            }

            // === Горизонтальное суммирование элементов регистра ===
            _mm256_storeu_pd(temp, sum_vec);
            double result_val = temp[0] + temp[1] + temp[2] + temp[3];

            // === Хвостовые элементы (скалярно) ===
            for (; k < sharedDim; ++k) {
                result_val += matA[r1 * sharedDim + k] * matT[c2 * sharedDim + k];
            }

            result[r1 * colsB + c2] = result_val;
        }
}

/**
 * Умножение матриц с AVX2 и gather инструкциями
 *
 * gather позволяет загружать несмежные данные из памяти по индексам
 * Здесь используется для прямого доступа к столбцам B без транспонирования
 */
void matrixMulAVX2_gather(double* result, const double* matA, const double* matB,
    size_t sharedDim, size_t rowsA, size_t colsB) {

    double temp[4];
    for (size_t r1 = 0; r1 < rowsA; r1++) {
        for (size_t c2 = 0; c2 < colsB; c2++) {
            __m256d sum_vec = _mm256_setzero_pd();
            size_t k = 0;

            // === SIMD цикл с gather ===
            for (; k + 3 < sharedDim; k += 4) {
                // Загружаем 4 элемента из A (последовательно)
                __m256d a_vec = _mm256_loadu_pd(&matA[r1 * sharedDim + k]);

                // Создаем индексы для gather: 
                // нужны элементы matB[k][c2], matB[k+1][c2], ...
                // в линейной памяти: k*colsB + c2, (k+1)*colsB + c2, ...
                __m256i indices = _mm256_setr_epi64x(
                    k * colsB + c2,
                    (k + 1) * colsB + c2,
                    (k + 2) * colsB + c2,
                    (k + 3) * colsB + c2
                );
                // gather загружает 4 double по заданным индексам
                // scale = sizeof(double) = 8 (множитель индексов)
                __m256d b_vec = _mm256_i64gather_pd(matB, indices, sizeof(double));

                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec));
            }

            // Горизонтальное суммирование
            _mm256_storeu_pd(temp, sum_vec);
            double result_val = temp[0] + temp[1] + temp[2] + temp[3];

            // Хвостовые элементы
            for (; k < sharedDim; ++k) {
                result_val += matA[r1 * sharedDim + k] * matB[k * colsB + c2];
            }

            result[r1 * colsB + c2] = result_val;
        }
    }
}

/**
 * Тест производительности для прямоугольных матриц
 */
void speedtest(size_t rowsA, size_t sharedDim, size_t colsB) {
    // Инициализация матриц
    std::vector<double> A = std::vector<double>(rowsA * sharedDim, 1.0);
    std::vector<double> B = std::vector<double>(colsB * sharedDim);
    std::vector<double> R1(rowsA * colsB, 0.0); // Результат обычного умножения
    std::vector<double> R2(rowsA * colsB, 0.0); // Результат AVX2
    std::vector<double> R3(rowsA * colsB, 0.0); // Результат AVX2+Gather

    // Заполняем B значениями 0, 1, 2, ...
    for (size_t i = 0; i < colsB * sharedDim; ++i) {
        B[i] = (double)i;
    }

    // === Замер обычного умножения ===
    auto t1 = std::chrono::steady_clock::now();
    matrixMul(R1.data(), A.data(), B.data(), sharedDim, rowsA, colsB);
    auto t2 = std::chrono::steady_clock::now();

    // === Замер AVX2 с транспонированием ===
    auto t3 = std::chrono::steady_clock::now();
    matrixMulAVX2(R2.data(), A.data(), B.data(), sharedDim, rowsA, colsB);
    auto t4 = std::chrono::steady_clock::now();

    // === Замер AVX2 с gather ===
    auto t5 = std::chrono::steady_clock::now();
    matrixMulAVX2_gather(R3.data(), A.data(), B.data(), sharedDim, rowsA, colsB);
    auto t6 = std::chrono::steady_clock::now();

    // Вывод результатов
    cout << "Duration of synchronous calc: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << endl;
    cout << "Duration of AVX2 calc: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms" << endl;
    cout << "Duration of AVX2_Gather calc: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() << " ms" << endl;
}

/**
 * Тест производительности для квадратных матриц с записью в CSV
 */
void speedtest_avg_openmp(size_t n) {
    std::cout << "======= SPEEDTEST MATRIX MUL ==========" << std::endl;

    // Получаем путь к директории выше текущей
    std::cout << "Current dir: " << std::filesystem::current_path() << std::endl;
    auto base_dir = std::filesystem::current_path().parent_path();
    std::ofstream output(base_dir.append("output.csv"));

    if (!output.is_open()) {
        std::cout << "Error while opening file" << std::endl;
        return;
    }

    // Заголовок CSV
    output << "Sync,AVX2,AVX2Acceleration,Gather,GatherAcceleration\n";

    // Инициализация матриц n×n
    std::vector<double> A(n * n, 1.0);
    std::vector<double> B(n * n);
    std::vector<double> R1(n * n, 0.0);
    std::vector<double> R2(n * n, 0.0);
    std::vector<double> R3(n * n, 0.0);

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (double)i;
    }

    // === Замеры времени ===
    auto t1 = std::chrono::steady_clock::now();
    matrixMul(R1.data(), A.data(), B.data(), n, n, n);
    auto t2 = std::chrono::steady_clock::now();

    auto t3 = std::chrono::steady_clock::now();
    matrixMulAVX2(R2.data(), A.data(), B.data(), n, n, n);
    auto t4 = std::chrono::steady_clock::now();

    auto t5 = std::chrono::steady_clock::now();
    matrixMulAVX2_gather(R3.data(), A.data(), B.data(), n, n, n);
    auto t6 = std::chrono::steady_clock::now();

    // Расчет ускорений
    double sync_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    double avx2_time = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
    double gather_time = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();

    // Вывод в консоль
    std::cout << "Sync time: " << sync_time
        << " ms\t| AVX2 time: " << avx2_time
        << " ms\t| AVX2 Acceleration: " << sync_time / avx2_time
        << "x\t| Gather time: " << gather_time
        << " ms\t| Gather Acceleration: " << sync_time / gather_time
        << "x" << endl;

    // Запись в CSV
    output << sync_time << ","
        << avx2_time << ","
        << sync_time / avx2_time << ","
        << gather_time << ","
        << sync_time / gather_time
        << endl;

    output.close();
}

int main() {
    // === Тест 1: Простая проверка корректности ===
    cout << "Test 1 (all ones * all twos):" << endl;
    std::size_t rowsA = 2, sharedDim = 3, colsB = 2;
    std::vector<double> A(rowsA * sharedDim, 1.0);  // [1,1,1; 1,1,1]
    std::vector<double> B(sharedDim * colsB, 2.0);  // [2,2; 2,2; 2,2]
    std::vector<double> R(rowsA * colsB, 0.0);

    matrixMul(R.data(), A.data(), B.data(), sharedDim, rowsA, colsB);
    matrixPrint(R.data(), rowsA, colsB);
    // Ожидаем: [6,6; 6,6] (каждый элемент = 1*2 + 1*2 + 1*2 = 6)

    // === Тест 2: Проверка с разными значениями ===
    cout << "\nTest 2 (specific values):" << endl;
    rowsA = 2; sharedDim = 3; colsB = 2;
    // Матрица A: 2×3
    A = std::vector<double>({ 1.0, 2.0, -3.0,   // первая строка
                             -2.0, 13.0, -2.0 }); // вторая строка
    // Матрица B: 3×2
    B = std::vector<double>({ 3.0, 4.0,    // первая строка
                             5.0, -1.0,   // вторая строка
                             4.0, 4.0 });  // третья строка
    R = std::vector<double>(rowsA * colsB, 0.0);

    matrixMul(R.data(), A.data(), B.data(), sharedDim, rowsA, colsB);
    matrixPrint(R.data(), rowsA, colsB);
    // Ожидаем:
    // [1*3 + 2*5 + (-3)*4,  1*4 + 2*(-1) + (-3)*4] = [1, -10]
    // [-2*3 + 13*5 + (-2)*4, -2*4 + 13*(-1) + (-2)*4] = [51, -29]

    // === Тест производительности ===
    cout << "\nRunning performance test..." << endl;
    speedtest_avg_openmp(1000); // Матрицы 1000×1000

    return 0;
}