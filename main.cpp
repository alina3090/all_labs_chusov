#include <complex>
#include <iostream>
#include <numbers>
#include <random>
#include <thread>
#include <vector>
#include <omp.h>
#include <filesystem>
#include <fstream>

using namespace std::numbers;

constexpr size_t MAX_TASK_DEPTH = 100;

void print_vector(const std::vector<std::complex<double>>& v) {
    for (std::size_t i = 0; i < v.size(); i++)
    {
        std::cout << "[" << i << "] " << std::fixed << v[i] << "\n";
    }
};

void randomize_vector(std::vector<std::complex<double>>& v) {
    std::uniform_real_distribution<double> unif(0, 100000);
    static std::random_device rd;
    std::default_random_engine re(rd());
    for (auto & i : v)
    {
        i = unif(re);
    }
};

bool approx_equal(const std::vector<std::complex<double>>& v,
                  const std::vector<std::complex<double>>& u) {
    for (std::size_t i = 0; i < v.size(); i++)
    {
        if (std::abs(v[i] - u[i]) > 0x1P-10)
        {
            std::cout << "Mismatch at index " << i
                      << ": " << v[i] << " != " << u[i]
                      << " (diff = " << std::abs(v[i] - u[i]) << ")"
                      << std::endl;
            return false;
        }
    }
    return true;
};


void dft_generic(const std::complex<double>* input, std::complex<double>* output, size_t n, std::complex<double> w, int inverse) {

    for (size_t k = 0; k < n; k++) {
        std::complex<double> sum(0.0, 0.0);
        std::complex<double> w_k = std::pow(w, static_cast<double>(k));
        std::complex<double> w_power = 1.0;

        for (size_t m = 0; m < n; m++) {
            sum += input[m] * w_power;
            w_power *= w_k;
        }

        output[k] = (inverse == -1) ? sum / static_cast<double>(n) : sum;
    }
}

void dft(const std::complex<double>* time, std::complex<double>* spectrum, size_t n, std::complex<double> w)
{
    dft_generic(time, spectrum, n, w, 1);
}

void idft(const std::complex<double>* spectrum, std::complex<double>* restored, size_t n, std::complex<double> w)
{
    dft_generic(spectrum, restored, n, w, -1);
}

void test_dft_correctness(size_t n) {
    std::cout << "======= Test DFT ==========" << std::endl;

    std::vector<std::complex<double>> original(n);
    if (n < 40)
        randomize_vector(original);

    std::cout << std::endl << "====== Original signal =======" << std::endl;
    if (n < 40)
        print_vector(original);

    std::vector<std::complex<double>> spectrum(n);
    std::complex<double> w1 = std::polar(1.0, -2.0 * pi_v<double> / n);

    auto fft_start = std::chrono::high_resolution_clock::now();
    dft(original.data(), spectrum.data(), n, w1);
    auto fft_end = std::chrono::high_resolution_clock::now();
    auto fft_time = std::chrono::duration_cast<std::chrono::milliseconds>(fft_end - fft_start);

    std::cout << std::endl << "===== Spectrum =======" << std::endl;
    if (n < 40)
        print_vector(spectrum);

    std::cout << std::endl << "====== DFT TIME =======" << std::endl;
    std::cout << "DFT time: " << fft_time.count() << " ms" << std::endl;

    // 3. Выполняем обратное DFT
    std::vector<std::complex<double>> restored(n);
    std::complex<double> w2 = std::polar(1.0, 2.0 * pi_v<double> / n);

    auto ifft_start = std::chrono::high_resolution_clock::now();
    idft(spectrum.data(), restored.data(), n, w2);
    auto ifft_end = std::chrono::high_resolution_clock::now();
    auto ifft_time = std::chrono::duration_cast<std::chrono::milliseconds>(ifft_end - ifft_start);

    std::cout << std::endl << "====== Restored signal =========" << std::endl;
    if (n < 40)
        print_vector(restored);

    std::cout << std::endl << "====== iDFT TIME =======" << std::endl;
    std::cout << "iDFT time: " << ifft_time.count() << " ms" << std::endl;

    std::cout << std::endl << "====== Check =========" << std::endl;
    if (!approx_equal(original, restored)) {
        std::cout << std::endl << "====== Error =========" << std::endl;
    } else {
        std::cout << std::endl << "====== OK =========" << std::endl;
    }
}

void fft_openmp_core(std::complex<double>* data, size_t n, int inverse, size_t depth = 0) {

    if (n <= 1) return;

    std::vector<std::complex<double>> even(n/2), odd(n/2);

#pragma omp parallel for if(depth == 0 && n > 1000)
    for (size_t i = 0; i < n/2; i++) {
        even[i] = data[2 * i];
        odd[i] = data[2 * i + 1];
    }

#pragma omp task shared(even) if(depth < MAX_TASK_DEPTH && n > 1000)
    {
        fft_openmp_core(even.data(), n / 2, inverse, depth + 1);
    }

#pragma omp task shared(odd) if(depth < MAX_TASK_DEPTH && n > 1000)
    {
        fft_openmp_core(odd.data(), n / 2, inverse, depth + 1);
    }

#pragma omp taskwait

#pragma omp parallel for if(depth == 0 && n > 1000)
    for (size_t i = 0; i < n / 2; i++) {
        double angle = (inverse == 1) ? -2.0 * pi_v<double> * i / n : 2.0 * pi_v<double> * i / n;
        std::complex<double> w = std::polar(1.0, angle);

        std::complex<double> t = w * odd[i];
        data[i] = even[i] + t;
        data[i + n / 2] = even[i] - t;
    }
}

void fft_recursive(std::complex<double>* data, size_t n, int inverse) {
#pragma omp parallel
#pragma omp single nowait
    {
        fft_openmp_core(data, n, inverse, 0);
    }
}

void fft_openmp(std::complex<double>* data, size_t n) {
    fft_recursive(data, n, 1);
}

void ifft_openmp(std::complex<double>* data, size_t n) {
    fft_recursive(data, n, -1);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        data[i] /= static_cast<double>(n);
    }
}


void test_fft_openmp_correctness(size_t n) {
    std::cout << "======= Test OPEN MP ==========" << std::endl;

    std::vector<std::complex<double>> original(n);
    if (n < 40)
        randomize_vector(original);

    std::cout << std::endl << "====== Original signal =======" << std::endl;
    if (n < 40)
        print_vector(original);

    std::vector<std::complex<double>> original_copy = original;
    std::vector<std::complex<double>> spectrum = original;

    auto fft_start = std::chrono::high_resolution_clock::now();
    fft_openmp(spectrum.data(), n);
    auto fft_end = std::chrono::high_resolution_clock::now();
    auto fft_time = std::chrono::duration_cast<std::chrono::milliseconds>(fft_end - fft_start);

    std::cout << std::endl << "====== Spectrum =======" << std::endl;
    if (n < 40)
        print_vector(spectrum);

    std::cout << std::endl << "====== OPEN MP TIME =======" << std::endl;
    std::cout << "FFT time: " << fft_time.count() << " ms" << std::endl;


    auto ifft_start = std::chrono::high_resolution_clock::now();
    ifft_openmp(spectrum.data(), n);
    auto ifft_end = std::chrono::high_resolution_clock::now();
    auto ifft_time = std::chrono::duration_cast<std::chrono::milliseconds>(ifft_end - ifft_start);

    std::cout << std::endl << "====== Restored signal =========" << std::endl;
    if (n < 40)
        print_vector(spectrum);

    std::cout << std::endl << "====== INVERSE OPEN MP TIME =======" << std::endl;
    std::cout << "Total time: " << (fft_time + ifft_time).count() << " ms" << std::endl;

    if (!approx_equal(original_copy, spectrum)) {
        std::cout << std::endl << "====== Error =========" << std::endl;
    } else {
        std::cout << std::endl << "====== OK =========" << std::endl;
    }
}


void speedtest_fft_openmp(size_t n, size_t exp_count) {
    std::cout << "======= SPEEDTEST OPEN MP FFT ==========" << std::endl;

    std::cout << "Current dir: " << std::filesystem::current_path() << std::endl;
    auto base_dir = std::filesystem::current_path().parent_path();
    std::ofstream output(base_dir.append("output.csv"));
    if (!output.is_open())
    {
        std::cout << "Error while opening file" << std::endl;
        return;
    }

    output << "T,Time,Avg,Acceleration\n";

    std::vector<std::complex<double>> original(n);
    randomize_vector(original);

    std::vector<std::complex<double>> original_copy = original;

    double time_sum_1;
    for (int thread_num = 1; thread_num <= std::thread::hardware_concurrency(); thread_num++) {
        omp_set_num_threads(thread_num);

        double time_sum = 0;
        auto t0 = std::chrono::steady_clock::now();

        original = original_copy;

        for (size_t exp = 0; exp < exp_count; exp++) {

            std::vector<std::complex<double>> spectrum = original;
            std::vector<std::complex<double>> restored(n);

            auto t1 = std::chrono::high_resolution_clock::now();
            fft_openmp(spectrum.data(), n);
            ifft_openmp(spectrum.data(), n);
            auto t2 = std::chrono::high_resolution_clock::now();
            time_sum += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

            if (!approx_equal(original_copy, spectrum)) {
                std::cout << "Warning: FFT/IFFT mismatch in experiment " << exp + 1
                          << " with " << thread_num << " thread_num\n";
            }
        }


        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);

        if (thread_num == 1) {
            time_sum_1 = time_sum;
        }

    std::cout << "FFT: T = " << thread_num << "\t| total experiment time: " << total_time.count() << "\t| avg fft time = "
              << time_sum / exp_count << "\tacceleration = " << (time_sum_1 / exp_count) / (time_sum / exp_count) << "\n";

    output << thread_num << "," << total_time.count() << "," << time_sum / exp_count << "," << (time_sum_1 / exp_count) / (time_sum / exp_count) << std::endl;

    }

    output.close();
}

int main()
{
    //    test_dft_correctness(1 << 4);

    //    std::cout << std::endl << std::endl;

    //    test_fft_openmp_correctness(1 << 4);

    speedtest_fft_openmp(1 << 14, 5);
    return 0;
}