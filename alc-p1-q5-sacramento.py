import numpy as np


def resolve_lu(A, b):

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)

    # L começa como identidade (diagonal = 1); U começa como cópia de A
    L = np.eye(n)
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            U[i, j] = A[i, j]

    # ---------- Eliminação de Gauss: constrói L e U ----------
    for k in range(n):
        pivo = U[k, k]
        if abs(pivo) < 1e-12:
            raise Exception(
                f"Pivô nulo encontrado na posição ({k}, {k}). "
                "A decomposição LU sem pivoteamento não é possível; "
                "utilize uma função alternativa (ex.: LU com pivoteamento parcial)."
            )
        for i in range(k + 1, n):
            m = U[i, k] / pivo          # multiplicador m_ik
            L[i, k] = m                 # o multiplicador vai direto para L
            for j in range(k, n):
                U[i, j] = U[i, j] - m * U[k, j]   # Li <- Li - m * Lk
            U[i, k] = 0.0               # zera exatamente abaixo do pivô

    # ---------- Substituição progressiva: Ly = b ----------
    y = np.zeros(n)
    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma += L[i, j] * y[j]
        y[i] = b[i] - soma              # L[i,i] = 1, não precisa dividir

    # ---------- Substituição regressiva: Ux = y ----------
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i, j] * x[j]
        x[i] = (y[i] - soma) / U[i, i]

    return L, U, x


if __name__ == "__main__":
    A = [[2, 1, 1],
         [4, 3, 3],
         [8, 7, 9]]
    b = [1, 2, 3]

    L, U, x = resolve_lu(A, b)
    print("L =\n", L)
    print("U =\n", U)
    print("x =", x)

    # Teste do pivô nulo
    try:
        resolve_lu([[0, 1], [1, 0]], [1, 2])
    except Exception as e:
        print("Exceção:", e)