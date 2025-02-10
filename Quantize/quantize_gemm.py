import numpy as np


def calculate_quant_params(arr_fp32):
    min_val_fp32 = np.min(arr_fp32)
    max_val_fp32 = np.max(arr_fp32)
    
    # The float range must include 0.f, so that 0.f can be represented within the quant range. 
    # Otherwise, the quantized value corresponding to 0.f will exceed the quant range.
    min_val_fp32 = min(0, min_val_fp32)
    max_val_fp32 = max(0, max_val_fp32)

    range_fp32 = max_val_fp32 - min_val_fp32

    scale_fp32 = range_fp32 / 255.0

    zeropoint_u8 = np.clip(0 - np.round(min_val_fp32 / scale_fp32).astype(np.uint32), 0, 255).astype(np.uint8)

    return scale_fp32, zeropoint_u8


def quantize(arr_fp32, scale_fp32, zeropoint_u8):
    arr_u8 = np.clip(np.round(arr_fp32 / scale_fp32).astype(np.uint32)+ zeropoint_u8.astype(np.uint32), 0, 255).astype(np.uint8) 
    return arr_u8


def dequantize(arr_u8, scale_fp32, zeropoint_u8):
    arr_fp32 = (arr_u8.astype(np.uint32) - np.uint32(zeropoint_u8)).astype(np.float32) * scale_fp32
    return arr_fp32


# Post-training quantization, PTQ

def quantized_gemm_PTQ(qA, qB, A_scale, A_zeropoint, B_scale, B_zeropoint):
    M, K = qA.shape
    K, N = qB.shape

    C = np.zeros((M, N), dtype=np.float32)

    for m0 in range(M):
        for n0 in range(N):
            sum = np.uint32(0)
            for k0 in range(K):
                sum += (qA[m0, k0] - A_zeropoint).astype(np.uint32) * (qB[k0, n0] - B_zeropoint).astype(np.uint32)
            C[m0, n0] = (sum * A_scale * B_scale)

    C_scale, C_zeropoint = calculate_quant_params(C)

    qC = quantize(C, C_scale, C_zeropoint)

    return qC, C_scale, C_zeropoint


def PTQ(A, B):
    A_scale, A_zeropoint = calculate_quant_params(A)
    B_scale, B_zeropoint = calculate_quant_params(B)

    qA = quantize(A, A_scale, A_zeropoint)
    qB = quantize(B, B_scale, B_zeropoint)

    qC, C_scale, C_zeropoint = quantized_gemm_PTQ(qA, qB, A_scale, A_zeropoint, B_scale, B_zeropoint)

    C = dequantize(qC, C_scale, C_zeropoint)

    return C
    
# Quantization-aware training, QAT

def quantized_gemm_QAT(qA, qB, A_scale, A_zeropoint, B_scale, B_zeropoint, C_scale, C_zeropoint):
    M, K = qA.shape
    K, N = qB.shape

    qC = np.zeros((M, N), dtype=np.uint8)

    for m0 in range(M):
        for n0 in range(N):
            sum = np.uint32(0)
            for k0 in range(K):
                sum += (qA[m0, k0] - A_zeropoint).astype(np.uint32) \
                    * (qB[k0, n0] - B_zeropoint).astype(np.uint32)

            tmpVal_fp32 = A_scale * B_scale * sum.astype(np.float32) / C_scale
            qC[m0, n0] = np.clip(np.round(tmpVal_fp32).astype(np.uint32)+ C_zeropoint.astype(np.uint32), 0, 255).astype(np.uint8)

    return qC


def QAT(A, B, A_scale, A_zeropoint, B_scale, B_zeropoint, C_scale, C_zeropoint):
    qA = quantize(A, A_scale, A_zeropoint)
    qB = quantize(B, B_scale, B_zeropoint)

    qC = quantized_gemm_QAT(qA, qB, A_scale, A_zeropoint, B_scale, B_zeropoint, C_scale, C_zeropoint)

    C = dequantize(qC, C_scale, C_zeropoint)

    return C

if __name__ == "__main__":
    A = np.array([
        [1.1, 1.4],
        [1.1, 1.2]
    ], dtype=np.float32)

    B = np.array([
        [1.1, 1.3],
        [1.4, 1.6],
    ], dtype=np.float32)

    # Ref, float

    refC = A@B

    print(f"Ref C \n{refC}")

    # PTQ

    ptqC = PTQ(A, B)

    print(f"PTQ C \n{ptqC}")

    # QAT

    A_scale, A_zeropoint = calculate_quant_params(A)
    B_scale, B_zeropoint = calculate_quant_params(B)
    C_scale, C_zeropoint = calculate_quant_params(refC)

    qatC = QAT(A, B, A_scale, A_zeropoint, B_scale, B_zeropoint, C_scale, C_zeropoint);

    print(f"QAT C \n{qatC}")
