import numpy as np
import matplotlib.pyplot as plt
import time


def generate_data(n=80, seed=42):
    """Генерация синтетических данных"""
    np.random.seed(seed)
    x = np.random.uniform(-3, 7, n)
    noise = np.random.normal(0, 0.3, n)
    y = 0.1 * (x - 4) * np.cos(x) + 0.5 * x + noise
    return x, y


def epanechnikov_kernel(z):
    """Ядро Епанечникова"""
    return np.where(np.abs(z) <= 1, 0.75 * (1 - z**2), 0.0)


def compute_delta(x):
    """Вычисление максимального расстояния между соседними точками (Δ)"""
    x_sorted = np.sort(x)
    deltas = np.diff(x_sorted)
    return np.max(deltas)


def nadaraya_watson_predict(x_train, y_train, x_pred, beta, delta):
    """Предсказание методом Надрая-Ватсона"""
    h = delta / beta
    y_pred = np.zeros_like(x_pred)
    
    for i, x0 in enumerate(x_pred):
        z = (x0 - x_train) / h
        K = epanechnikov_kernel(z)
        denominator = np.sum(K)
        
        if denominator > 1e-10:
            y_pred[i] = np.sum(K * y_train) / denominator
        else:
            # Если все веса нулевые — берём среднее
            y_pred[i] = np.mean(y_train)
    
    return y_pred


def loo_mse(x, y, beta, delta):
    """Leave-One-Out Cross Validation (скользящий экзамен)"""
    n = len(x)
    y_pred_loo = np.zeros(n)
    
    for i in range(n):
        x_train = np.delete(x, i)
        y_train = np.delete(y, i)
        y_pred_loo[i] = nadaraya_watson_predict(x_train, y_train, [x[i]], beta, delta)[0]
    
    mse = np.mean((y - y_pred_loo) ** 2)
    return mse


def find_best_beta(x, y, delta, beta_range=np.arange(0.1, 2.1, 0.1)):
    """Поиск оптимального beta через LOOCV"""
    mses = []
    for beta in beta_range:
        mse = loo_mse(x, y, beta, delta)
        mses.append(mse)
    
    best_idx = np.argmin(mses)
    best_beta = beta_range[best_idx]
    best_mse = mses[best_idx]
    
    return best_beta, best_mse, beta_range, mses


def main():
    print("=== Непараметрическая регрессия Nadaraya-Watson (LW1) ===\n")
    
    start_time = time.time()
    
    # 1. Генерация данных
    x, y = generate_data(n=80)
    delta = compute_delta(x)
    print(f"Максимальный шаг Δ = {delta:.4f}\n")
    
    # 2. Поиск лучшего beta
    print("Поиск оптимального параметра β...")
    best_beta, best_mse, betas, mses = find_best_beta(x, y, delta)
    
    print(f"Лучшее β = {best_beta:.1f}")
    print(f"Минимальная MSE (LOOCV) = {best_mse:.4f}\n")
    
    # 3. Построение финальной модели
    x_sorted = np.sort(x)
    y_pred = nadaraya_watson_predict(x, y, x_sorted, best_beta, delta)
    
    # 4. Визуализация
    plt.figure(figsize=(14, 10))
    
    # График 1: Данные + регрессия
    plt.subplot(2, 1, 1)
    plt.scatter(x, y, alpha=0.7, label='Исходные данные (с шумом)', color='blue')
    plt.plot(x_sorted, y_pred, 'r-', linewidth=2.5, label=f'НПР (β={best_beta:.1f})')
    plt.title('Непараметрическая регрессия Nadaraya-Watson')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 2: Зависимость MSE от β
    plt.subplot(2, 1, 2)
    plt.plot(betas, mses, 'o-', color='green', linewidth=2)
    plt.axvline(x=best_beta, color='red', linestyle='--', alpha=0.7)
    plt.title('Зависимость ошибки LOOCV от параметра размытия β')
    plt.xlabel('β')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time:.2f} секунд")


if __name__ == "__main__":
    main()    for i in range(n):
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
