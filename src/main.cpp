#include <pybind11/pybind11.h>
#include <matplot/matplot.h>
#include <pybind11/stl_bind.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <matplot/matplot.h>
#include <vector>
#include <cmath>
#include <complex>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#define pi 3.1415926535897932

std::vector<double> generate_sin(double freq, double t_start, double t_end, int num_samples) {
    std::vector<double> y(num_samples);
    double dt = (t_end - t_start) / (num_samples - 1);

    for (int i = 0; i < num_samples; ++i) {
        double t = t_start + i * dt;
        y[i] = std::sin(2 * pi * freq * t);
    }

    return y;
}
std::vector<double> generate_cos(double freq, double t_start, double t_end, int num_samples) {
    std::vector<double> y(num_samples);
    double dt = (t_end - t_start) / (num_samples - 1);

    for (int i = 0; i < num_samples; ++i) {
        double t = t_start + i * dt;
        y[i] = std::cos(2 * pi * freq * t);
    }

    return y;
}
std::vector<double> generate_sawtooth(double freq, double t_start, double t_end, int num_samples) {
    std::vector<double> y(num_samples);
    double dt = (t_end - t_start) / (num_samples - 1);
    double T = 1.0 / freq;

    for (int i = 0; i < num_samples; ++i) {
        double t = t_start + i * dt;
        double mod = fmod(t, T); 
        y[i] = 2.0 * (mod / T) - 1.0; 
    }

    return y;
}

std::vector<double> generate_square(double freq, double t_start, double t_end, int num_samples) {
    std::vector<double> y(num_samples);
    double dt = (t_end - t_start) / (num_samples - 1);
    double T = 1.0 / freq;

    for (int i = 0; i < num_samples; ++i) {
        double t = t_start + i * dt;
        double mod = fmod(t, T);
        y[i] = (mod < T / 2.0) ? 1.0 : -1.0;
    }

    return y;
}

std::vector<std::complex<double>> dft(const std::vector<double>& input) {
    size_t N = input.size();
    std::vector<std::complex<double>> out(N);
    for (size_t k = 0; k < N; ++k) {
        std::complex<double> sum = 0;
        for (size_t n = 0; n < N; ++n) {
            double angle = -2.0 * pi * k * n / N;
            sum += input[n]*std::polar(1.0, angle);
        }
        out[k] = sum;
    }
    return out;
}

std::vector<double> idft(const std::vector<std::complex<double>>& input) {
    size_t N = input.size();
    std::vector<double> out(N);
    for (size_t n = 0; n < N; ++n) {
        std::complex<double> sum = 0;
        for (size_t k = 0; k < N; ++k) {
            double angle = 2.0 * pi * k * n / N;
            sum += input[k] * std::polar(1.0, angle);
        }
        out[n] = sum.real() / N;
    }
    return out;
}
std::vector<double> filter_1d(const std::vector<double>& signal, const std::vector<double>& kernel) {
    size_t n = signal.size();
    size_t k = kernel.size();
    std::vector<double> y(n, 0.0);
    int offset = k / 2;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
            int id = i + j - offset;
            if (id >= 0 && id < n) {
                y[i] += signal[id] * kernel[j];
            }
        }
    }
    return y;
}
std::vector<std::vector<double>> filter_2d(const std::vector<std::vector<double>>& img, const std::vector<std::vector<double>>& kernel) {
    size_t rows = img.size();
    size_t cols = img[0].size();
    size_t k_rows = kernel.size();
    size_t k_cols = kernel[0].size();
    int row_offset = k_rows / 2;
    int col_offset = k_cols / 2;

    std::vector<std::vector<double>> output(rows, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (size_t ki = 0; ki < k_rows; ++ki) {
                for (size_t kj = 0; kj < k_cols; ++kj) {
                    int ni = i + ki - row_offset;
                    int nj = j + kj - col_offset;
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                        sum += img[ni][nj] * kernel[ki][kj];
                    }
                }
            }
            output[i][j] = sum;
        }
    }

    return output;
}
void plot_signal(const std::vector<double>& y) {
    using namespace matplot;
    std::vector<double> x(y.size());
    for (size_t i = 0; i < y.size(); ++i)
        x[i] = static_cast<double>(i);

    plot(x, y);
    title("Signal Visualization");
    xlabel("Sample");
    ylabel("Amplitude");
    show();
}
namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
     m.def("generate_sin", &generate_sin);
     m.def("generate_cos", &generate_cos );
     m.def("generate_sawtooth", &generate_sawtooth);
     m.def("generate_square", &generate_square );
     m.def("plot_signal", &plot_signal);
     m.def("dft", &dft);
     m.def("idft", &idft);
     m.def("filter_1d", &filter_1d);
     m.def("filter_2d", &filter_2d);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
