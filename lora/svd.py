# 该文件为奇异值分解，是为了验证 lora 方式的有效性
import torch
import numpy as np
_ = torch.manual_seed(0)

d, k = 10, 10
w_rank = 2

# 这种方式能够产生一个低秩矩阵
w = torch.randn(d, w_rank) @ torch.randn(w_rank, k)
print(w) # 10 * 10
print(np.linalg.matrix_rank(w)) # 秩为 2

# svd(奇异值分解)
U, S, V = torch.svd(w)

U_r = U[:, :w_rank]
S_r = torch.diag(S[:w_rank])
V_r = V[:, :w_rank].t()


B = U_r @ S_r
A = V_r

print(B.shape) # 10,2
print(A.shape) # 2, 10

# 比较验证
bias = torch.randn(d)
x = torch.randn(d)

y = w @ x + bias
y_prime = (B @ A) @ x + bias

print(y)
print(y_prime)

print(w.nelement)
print((B.nelement() + A.nelement()))



