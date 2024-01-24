import numpy as np


def inti_GMM(dataset, K):
    N, D = np.shape(dataset)
    val_max = np.max(dataset, axis=0)
    val_min = np.min(dataset, axis=0)
    centers = np.linspace(val_min, val_max, num=K + 2)

    mus = centers[1:-1, :]
    sigmas = np.array([0.5 * np.eye(D) for i in range(K)])
    ws = 1.0 / K * np.ones(K)

    return mus, sigmas, ws


# 计算一个高斯的pdf
# x: 数据 [N,D]
# sigma 方差 [D,D]
# mu 均值 [1,D]
def getPdf(x, mu, sigma, eps=1e-12):

    try:
        N, D = np.shape(x)
    except ValueError:
        N = 1
        D = x.shape[0]

    if D == 1:
        sigma = sigma + eps
        A = 1.0 / sigma
        det = np.fabs(sigma[0])
    else:
        sigma = sigma + eps * np.eye(D)
        A = np.linalg.inv(sigma)
        det = np.fabs(np.linalg.det(sigma))

    # 计算系数
    factor = (2.0 * np.pi) ** (D / 2.0) * det ** 0.5

    # 计算 pdf
    dx = x - mu
    if N == 1:
        pdf = (np.exp(-0.5 * np.dot(np.dot(dx, A), dx)) + eps) / factor
    else:
        pdf = [(np.exp(-0.5 * np.dot(np.dot(dx[i], A), dx[i])) + eps) / factor for i in range(N)]

    return pdf


def train_GMM_step(dataset, mus, sigmas, ws):
    N, D = np.shape(dataset)
    K, D = np.shape(mus)
    # 计算样本在每个成分上的pdf
    pdfs = np.zeros([N, K])
    for k in range(K):
        pdfs[:, k] = getPdf(dataset, mus[k], sigmas[k])

    # 获取r
    r = pdfs * np.tile(ws, (N, 1))
    r_sum = np.tile(np.sum(r, axis=1, keepdims=True), (1, K))
    r = r / r_sum

    # 进行参数的更新
    for k in range(K):
        r_k = r[:, k]
        N_k = np.sum(r_k)
        r_k = r_k[:, np.newaxis]  # [N,1]

        # 更新mu
        mu = np.sum(dataset * r_k, axis=0) / N_k  # [D,1]

        # 更新sigma
        dx = dataset - mu
        sigma = np.zeros([D, D])
        for i in range(N):
            sigma = sigma + r_k[i, 0] * np.outer(dx[i], dx[i])
        sigma = sigma / N_k

        # 更新w
        w = N_k / N
        mus[k] = mu
        sigmas[k] = sigma
        ws[k] = w
    return mus, sigmas, ws


def train_GMM(dataset, K, m):
    mus, sigmas, ws = inti_GMM(dataset, K)

    for i in range(m):
        # print("Step ",i)
        mus, sigmas, ws = train_GMM_step(dataset, mus, sigmas, ws)
    # return mus, sigms, ws
    return mus, sigmas


def getlogPdfFromeGMM(datas, mus, sigmas, ws):
    N, D = np.shape(datas)
    K, D = np.shape(mus)

    weightedlogPdf = np.zeros([N, K])

    for k in range(K):
        temp = getPdf(datas, mus[k], sigmas[k], eps=1e-12)
        weightedlogPdf[:, k] = np.log(temp) + np.log(ws[k])

    return weightedlogPdf, np.sum(weightedlogPdf, axis=1)


def clusterByGMM(datas, mus, sigmas, ws):
    weightedlogPdf, _ = getlogPdfFromeGMM(datas, mus, sigmas, ws)
    labs = np.argmax(weightedlogPdf, axis=1)
    return labs
