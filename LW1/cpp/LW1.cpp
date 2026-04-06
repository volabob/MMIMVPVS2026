#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;

double kernel(double z) {
    return (fabs(z) <= 1.0) ? 0.75 * (1.0 - z * z) : 0.0;
}

double predict_nw(double x_new, const vector<double>& x_train, const vector<double>& y_train, double h) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < x_train.size(); ++i) {
        double z = (x_new - x_train[i]) / h;
        double k = kernel(z);
        num += k * y_train[i];
        den += k;
    }
    return (den > 1e-12) ? num / den : 0.0;
}

pair<vector<double>, vector<double>> generate_data(int n = 80) {
    mt19937 gen(42);
    uniform_real_distribution<double> ux(-3.0, 7.0);
    normal_distribution<double> noise(0.0, 0.3);
    vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = ux(gen);
        y[i] = 0.1 * (x[i] - 4.0) * cos(x[i]) + 0.5 * x[i] + noise(gen);
    }
    return {x, y};
}

double compute_delta(const vector<double>& x) {
    vector<double> xs = x;
    sort(xs.begin(), xs.end());
    double max_d = 0.0;
    for (size_t i = 1; i < xs.size(); ++i) {
        max_d = max(max_d, xs[i] - xs[i - 1]);
    }
    return max_d;
}

double loo_mse(double beta, const vector<double>& x, const vector<double>& y, double delta) {
    size_t n = x.size();
    double h = delta / beta;
    double mse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        vector<double> xt, yt;
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                xt.push_back(x[j]);
                yt.push_back(y[j]);
            }
        }
        double pred = predict_nw(x[i], xt, yt, h);
        mse += (y[i] - pred) * (y[i] - pred);
    }
    return mse / n;
}

int main() {
    std::setlocale(LC_ALL, "ru_RU.UTF-8");
    std::cout << std::fixed << std::setprecision(4);
    auto start = chrono::high_resolution_clock::now();

    cout << "=== Непараметрическая регрессия (C++ / Nadaraya-Watson) ===\n\n";

    auto [x, y] = generate_data();
    double delta = compute_delta(x);
    cout << "Δ (max spacing) = " << fixed << setprecision(4) << delta << "\n";

    // Перебор β
    vector<double> betas, mses;
    double best_beta = 0.1, best_mse = 1e9;
    for (double b = 0.1; b <= 2.01; b += 0.1) {
        double mse = loo_mse(b, x, y, delta);
        betas.push_back(b);
        mses.push_back(mse);
        if (mse < best_mse) {
            best_mse = mse;
            best_beta = b;
        }
    }

    cout << "Лучший β = " << best_beta << "\n";
    cout << "LOOCV MSE = " << fixed << setprecision(4) << best_mse << "\n\n";

    // Финальная модель
    double h_opt = delta / best_beta;
    vector<double> x_plot, y_plot;
    for (double xi = -3.0; xi <= 7.0; xi += 0.02) {
        x_plot.push_back(xi);
        y_plot.push_back(predict_nw(xi, x, y, h_opt));
    }

    // Сохранение для графиков
    ofstream fdata("data_points.csv");
    for (size_t i = 0; i < x.size(); ++i) fdata << x[i] << "," << y[i] << "\n";
    fdata.close();

    ofstream freg("regression_curve.csv");
    for (size_t i = 0; i < x_plot.size(); ++i) freg << x_plot[i] << "," << y_plot[i] << "\n";
    freg.close();

    ofstream fmse("mse_vs_beta.csv");
    for (size_t i = 0; i < betas.size(); ++i) fmse << betas[i] << "," << mses[i] << "\n";
    fmse.close();

    auto end = chrono::high_resolution_clock::now();
    double sec = chrono::duration<double>(end - start).count();
    cout << "Время выполнения: " << fixed << setprecision(2) << sec << " сек\n";
    cout << "Данные сохранены в *.csv файлы\n";

    return 0;
}
