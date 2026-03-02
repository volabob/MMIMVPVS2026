import numpy as np
import matplotlib.pyplot as plt
import time

def generate_data(n=80, seed=42):
    """Генерация выборки"""
    np.random.seed(seed)
    x = np.random.uniform(-3, 7, n)
    noise = np.random.normal(0, 0.3, n)
    y = 0.1 * (x - 4) * np.cos(x) + 0.5 * x + noise
    return x, y

def compute_delta(x):
    """Δ = max{(x_{i+1} - x_i)} после сортировки (неравномерные отсчёты)"""
    x_sorted = np.sort(x)
    return np.max(np.diff(x_sorted))

def kernel(z):
    """Epanechnikov: K(z) из формулы (4.7.4)"""
    return np.where(np.abs(z) <= 1, 0.75 * (1 - z**2), 0.0)

def predict_nw(x_new, x_train, y_train, h):
    """Nadaraya-Watson оценка (формула 4.7.3)"""
    x_new = np.atleast_1d(x_new)
    diffs = (x_new[:, np.newaxis] - x_train) / h          # z = (x - x_i)/h
    k = kernel(diffs)
    sum_k = np.sum(k, axis=1, keepdims=True)
    weights = k / (sum_k + 1e-12)
    y_pred = np.sum(weights * y_train, axis=1)
    return y_pred[0] if len(y_pred) == 1 else y_pred

def loo_mse(beta, x, y, delta):
    """Leave-One-Out MSE для данного β"""
    n = len(x)
    h = delta / beta                                      # h = Δ / β (формула 4.7.5)
    y_pred_loo = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        y_pred_loo[i] = predict_nw(x[i], x[mask], y[mask], h)
    return np.mean((y - y_pred_loo)**2)

def find_best_beta(x, y, delta):
    """Перебор β ∈ [0.1, 2.0] шаг 0.1"""
    betas = np.arange(0.1, 2.01, 0.1)
    mses = [loo_mse(beta, x, y, delta) for beta in betas]
    best_idx = np.argmin(mses)
    return betas[best_idx], mses[best_idx], betas, mses

def main():
    start = time.time()
    
    # 1. Генерация данных
    x, y = generate_data()
    delta = compute_delta(x)
    print(f"Δ (max spacing) = {delta:.4f}")
    
    # 2. Поиск лучшего β через LOOCV
    best_beta, best_mse, betas, mses = find_best_beta(x, y, delta)
    print(f"Лучший β = {best_beta:.1f}, LOOCV MSE = {best_mse:.4f}")
    
    # 3. Финальная модель на всех данных
    h_opt = delta / best_beta
    x_plot = np.linspace(-3, 7, 500)
    y_plot = predict_nw(x_plot, x, y, h_opt)
    
    comp_time = time.time() - start
    print(f"Время вычислений: {comp_time:.3f} секунд")
    
    # 4. Графики
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    axs[0].scatter(x, y, alpha=0.7, s=30, label='Данные (n=80)')
    axs[0].plot(x_plot, y_plot, 'r-', lw=2.5, label=f'НП регрессия (β={best_beta:.1f})')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Непараметрическая регрессия (Nadaraya-Watson)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(betas, mses, 'bo-', markersize=4)
    axs[1].axvline(best_beta, color='red', linestyle='--', label=f'Лучший β = {best_beta:.1f}')
    axs[1].set_xlabel('Параметр размытия ядра β')
    axs[1].set_ylabel('MSE (leave-one-out)')
    axs[1].set_title('Зависимость ошибки от β')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Оптимальная ширина окна h = {h_opt:.4f} (по формуле 4.7.5)")

if __name__ == "__main__":
    main()
