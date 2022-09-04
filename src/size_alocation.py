import numpy as np
from scipy.optimize import minimize
from functools import partial

def g_ab(x, a, b):
    return b * np.exp(-x*a)

def f_ab(x, a, b, T):
    return np.abs(np.sum(g_ab(x, a, b)) - T)


def size_allocator(total_size, sizes, min_size=10):
    N = len(sizes)
    
    assert total_size >= N
    
    sizes = np.asarray(sizes)
    
    min_size =  min(min(min_size, total_size // N), sizes.min())
    
    new_sizes = np.array([min_size] * N)
    
    total_size = total_size - sum(new_sizes)
    
    dist = np.ones(N)
    
    a=dist.astype(np.float32)
    b=(sizes - new_sizes).astype(np.float32)
    T=total_size
    
    f = partial(f_ab, a=a, b=b, T=T)
    k = minimize(f, x0=np.array([1.0], dtype=np.float32), method='Nelder-Mead', tol=1e-3).x[0]

    delta = np.ceil(g_ab(k, a=a, b=b)).astype(int)
    
    new_sizes += delta
    
    new_sizes = np.minimum(new_sizes, sizes)
    
    return new_sizes


if __name__ == "__main__":
    a = np.array([3, 1, 1, 1], dtype=np.float32)
    b = np.array([10, 20, 1, 4], dtype=np.float32)
    T = 15
    f = partial(f_ab, a=a,b=b, T=T)
    res = minimize(f, x0=[1.0], method='Nelder-Mead', tol=1e-3)
    print(res)
