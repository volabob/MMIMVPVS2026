import numpy as np
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
from joblib import Parallel, delayed

def gaussian_kernel(z):
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)

def nadaraya_watson_single(x_target, X, Y, h):
    distances = (x_target - X) / h
    weights = gaussian_kernel(distances)
    sum_weights = np.sum(weights)
    if sum_weights == 0:
        return 0.0
    return np.sum(Y * weights) / sum_weights

def evaluate_bandwidth_worker(args):
    h, X_train, Y_train, X_val, Y_val = args
    predictions = [nadaraya_watson_single(x, X_train, Y_train, h) for x in X_val]
    mse = mean_squared_error(Y_val, predictions)
    return h, mse

def parallel_by_bandwidth(X_train, Y_train, X_val, Y_val, h_grid, n_jobs=-1):
    tasks = [(h, X_train, Y_train, X_val, Y_val) for h in h_grid]
    with Pool(processes=n_jobs if n_jobs > 0 else None) as pool:
        results = pool.map(evaluate_bandwidth_worker, tasks)
    best_h, best_mse = min(results, key=lambda item: item[1])
    return best_h, results

def compute_partial_sums(x_target, X_chunk, Y_chunk, h):
    distances = (x_target - X_chunk) / h
    weights = gaussian_kernel(distances)
    return np.sum(Y_chunk * weights), np.sum(weights)

def parallel_by_sum(x_target, X, Y, h, n_chunks=4):
    X_chunks = np.array_split(X, n_chunks)
    Y_chunks = np.array_split(Y, n_chunks)
    
    partial_results = Parallel(n_jobs=n_chunks)(
        delayed(compute_partial_sums)(x_target, X_chunks[i], Y_chunks[i], h)
        for i in range(n_chunks)
    )
    
    total_num = sum(res[0] for res in partial_results)
    total_den = sum(res[1] for res in partial_results)
    
    if total_den == 0:
        return 0.0
    return total_num / total_den

if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.random.uniform(-5, 5, 1000)
    Y_train = np.sin(X_train) + np.random.normal(0, 0.1, 1000)
    
    X_val = np.random.uniform(-5, 5, 200)
    Y_val = np.sin(X_val) + np.random.normal(0, 0.1, 200)
    
    h_grid = np.linspace(0.1, 2.0, 20)
    
    print("Executing Variant 1...")
    best_h, all_results = parallel_by_bandwidth(X_train, Y_train, X_val, Y_val, h_grid)
    print(f"Best h found: {best_h:.4f}")
    
    print("\nExecuting Variant 2...")
    single_x = 1.5
    pred_val = parallel_by_sum(single_x, X_train, Y_train, h=best_h, n_chunks=4)
    print(f"Prediction for x={single_x}: {pred_val:.4f}")
