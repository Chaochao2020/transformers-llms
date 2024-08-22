import numpy as np

# 抑制科学记数法的显示
np.set_printoptions(suppress=True)

# 生成一组在 -50 到 150 之间均匀分布的 20 个随机数
params = np.random.uniform(low=-50, high=150, size=20)

# 将最大值 + 1， 最小值 - 1 和 0， 加入到其中
params[0] = params.max() + 1
params[1] = params.min() - 1
params[2] = 0

# 保留两位小数
params = np.round(params, 2)

print(params)

def clamp(params_q: np.array, lower_bound: int, upper_bound: int) -> np.array:
    params_q[params_q < lower_bound] = lower_bound
    params_q[params_q > upper_bound] = upper_bound
    return params_q

def asymmetric_quantization(params: np.array, bits: int) -> tuple[np.array, float, int]:
    # Calculate the scale and zero point
    alpha = np.max(params)
    beta = np.min(params)
    scale = (alpha - beta) / (2**bits-1)
    zero = -1*np.round(beta / scale)
    lower_bound, upper_bound = 0, 2**bits-1
    # Quantize the parameters
    quantized = clamp(np.round(params / scale + zero), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale, zero

def asymmetric_dequantize(params_q: np.array, scale: float, zero: int) -> np.array:
    return (params_q - zero) * scale

def symmetric_dequantize(params_q: np.array, scale: float) -> np.array:
    return params_q * scale

def symmetric_quantization(params: np.array, bits: int) -> tuple[np.array, float]:
    # Calculate the scale
    alpha = np.max(np.abs(params))
    scale = alpha / (2**(bits-1)-1)
    lower_bound = -2**(bits-1)
    upper_bound = 2**(bits-1)-1
    # Quantize the parameters
    quantized = clamp(np.round(params / scale), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale

def quantization_error(params: np.array, params_q: np.array):
    # calculate the MSE
    return np.mean((params - params_q)**2)

(asymmetric_q, asymmetric_scale, asymmetric_zero) = asymmetric_quantization(params, 8)
(symmetric_q, symmetric_scale) = symmetric_quantization(params, 8)

print(f'Original:')
print(np.round(params, 2))
print('')
print(f'Asymmetric scale: {asymmetric_scale}, zero: {asymmetric_zero}')
print(asymmetric_q)
print('')
print(f'Symmetric scale: {symmetric_scale}')
print(symmetric_q)