"""Python implementation of eT2QFNN (evolving Type-2 Quantum Fuzzy Neural Network)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from sklearn.mixture import GaussianMixture


EPS = 1e-12
MIN_COV = 1e-8
EXP_CLIP = 50.0


@dataclass
class RuleInfo:
    nrule: np.ndarray
    K: int
    wr: np.ndarray


@dataclass
class NetworkInfo:
    I: int
    miu: np.ndarray
    ns: int
    b: float
    theta_up: np.ndarray
    theta_low: np.ndarray
    omega_up: np.ndarray
    omega_low: np.ndarray
    qleft: np.ndarray
    qright: np.ndarray


def _safe_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x, -EXP_CLIP, EXP_CLIP))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + _safe_exp(-x))


def _safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return numer / (denom + EPS)


def _cov_floor(cov_diag: np.ndarray) -> np.ndarray:
    return np.maximum(cov_diag, MIN_COV)


def _prod_except(values: np.ndarray, index: int) -> float:
    if values.size == 1:
        return 1.0
    if index == 0:
        return float(np.prod(values[1:]))
    if index == values.size - 1:
        return float(np.prod(values[:-1]))
    return float(np.prod(np.concatenate([values[:index], values[index + 1 :]])))


def _dekf_gain(P: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
    S = R + H.T @ P @ H
    S = S + EPS * np.eye(S.shape[0])
    try:
        return P @ H @ np.linalg.solve(S, np.eye(S.shape[0]))
    except LinAlgError:
        return P @ H @ np.linalg.pinv(S)


def eT2QFNN(
    data: np.ndarray,
    I: int,
    lr: float,
    mode: int,
    fix_themodel: int,
    random_state: int | None = 0,
    plot: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict, Dict, Dict]:
    start_time = time.time()
    if verbose:
        print("eT2QFNN: running ...")

    data = np.asarray(data, dtype=float)
    N, mn = data.shape
    M = mn - I
    x = data[:, :I]
    z = data[:, I:]

    K = 1
    nrule = np.zeros(N, dtype=int)
    nrule[0] = K

    ns = int(round(0.5 * I))
    if ns == 1:
        ns = 2

    rng = np.random.default_rng(random_state)
    sample_size = min(fix_themodel, 30)
    kk = rng.permutation(fix_themodel)[:sample_size]
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="diag",
        random_state=random_state,
    )
    gmm.fit(data[kk, :I])

    gmm_weight = gmm.weights_.astype(float)
    gmm_miu = gmm.means_.T
    gmm_var = gmm.covariances_.astype(float)

    center = np.zeros(I)
    var = np.zeros(I)
    for model in range(gmm.n_components):
        center += gmm_weight[model] * gmm_miu[:, model]
        var += gmm_weight[model] * gmm_var[model]

    m = [x[0].copy()]
    grow = 1
    if verbose:
        print("The new rule no 1 is FORMED around sample 1")

    if mode == 1:
        b = 10.0
    elif mode == 2:
        b = 5.0
    else:
        b = 1.0

    sigma_up = [_cov_floor(var)]
    sigma_low = [_cov_floor(0.7 * sigma_up[0])]

    nr = np.arange(1, ns + 1, dtype=float)
    theta_up = [np.zeros((I, ns))]
    theta_low = [np.zeros((I, ns))]
    D_up = np.sqrt(sigma_up[0])
    D_low = 0.7 * D_up
    for i in range(I):
        theta_up[0][i, :] = nr * D_up[i] / ((ns + 1) / 2)
        theta_low[0][i, :] = nr * D_low[i] / ((ns + 1) / 2)

    omega_up = [0.6 * np.ones((M, I + 1))]
    omega_low = [0.4 * np.ones((M, I + 1))]
    nomega_up = [float(np.linalg.norm(omega_up[0]))]
    nomega_low = [float(np.linalg.norm(omega_low[0]))]
    qleft = 0.4 * np.ones(M)
    qright = 0.7 * np.ones(M)
    index = 0

    y = np.zeros((N, M))
    out = np.zeros((N, M))
    e = np.zeros((N, M))
    wr = np.zeros(N)

    Z = 2 * M * (2 + I) + I * (2 * ns + 1)
    P = [np.eye(Z)]
    R = lr * np.eye(M)

    for k in range(N):
        if k < fix_themodel:
            m_candidate = x[k].copy()
            sigma_candidate = _cov_floor((x[k] - center) ** 2)
            sigma_up.append(sigma_candidate)
            sigma_low.append(_cov_floor(0.7 * sigma_candidate))
            omega_up.append(omega_up[index].copy())
            omega_low.append(omega_low[index].copy())

            D_up = np.sqrt(sigma_candidate)
            D_low = 0.7 * D_up
            theta_up.append(np.zeros((I, ns)))
            theta_low.append(np.zeros((I, ns)))
            for i in range(I):
                theta_up[-1][i, :] = nr * D_up[i] / ((ns + 1) / 2)
                theta_low[-1][i, :] = nr * D_low[i] / ((ns + 1) / 2)

            nomega_up.append(float(np.linalg.norm(omega_up[-1])))
            nomega_low.append(float(np.linalg.norm(omega_low[-1])))
            nqleft = float(np.linalg.norm(qleft))
            nqright = float(np.linalg.norm(qright))

            N_up = np.zeros((gmm.n_components, K + 1))
            N_low = np.zeros((gmm.n_components, K + 1))
            E = np.zeros(K + 1)

            for i in range(K + 1):
                n_up = np.zeros(gmm.n_components)
                n_low = np.zeros(gmm.n_components)
                for j in range(gmm.n_components):
                    diff = m_candidate if i == K else m[i]
                    diff = diff - gmm_miu[:, j]
                    denom_up = 0.5 * (sigma_up[i] if i < K else sigma_candidate) + gmm_var[j]
                    denom_low = 0.5 * (sigma_low[i] if i < K else 0.7 * sigma_candidate) + gmm_var[j]
                    n_up[j] = float(np.exp(-0.5 * np.sum((diff ** 2) / (denom_up + EPS))))
                    n_low[j] = float(np.exp(-0.5 * np.sum((diff ** 2) / (denom_low + EPS))))

                N_up[:, i] = n_up
                N_low[:, i] = n_low

                sigma_up_i = sigma_up[i] if i < K else sigma_candidate
                sigma_low_i = sigma_low[i] if i < K else 0.7 * sigma_candidate
                det_up = float(np.prod(_cov_floor(sigma_up_i)))
                det_low = float(np.prod(_cov_floor(sigma_low_i)))
                common_up = (np.pi ** (I / 2)) * (det_up ** 0.5) * float(N_up[:, i] @ gmm_weight)
                common_low = (np.pi ** (I / 2)) * (det_low ** 0.5) * float(N_low[:, i] @ gmm_weight)

                Er = nqright * nomega_up[i] * (common_up**0.5) + (1 - nqright) * nomega_low[i] * (common_low**0.5)
                El = nqleft * nomega_up[i] * (common_up**0.5) + (1 - nqleft) * nomega_low[i] * (common_low**0.5)
                E[i] = abs(Er) + abs(El)

            if (E[K] > 0.7 * np.sum(E[:K])) and k > 0:
                K += 1
                P.append(np.eye(Z))
                for j in range(K - 1):
                    P[j] = P[j] * ((K**2 + 1) / (K**2))
                grow = 1
                m.append(m_candidate)
                if verbose:
                    print(f"The new rule no {K} is FORMED around sample {k + 1}")
            else:
                grow = 0
                sigma_up.pop()
                sigma_low.pop()
                omega_up.pop()
                omega_low.pop()
                theta_up.pop()
                theta_low.pop()
                nomega_up.pop()
                nomega_low.pop()
            nrule[k] = K

        Ql = np.zeros((I, K))
        Qu = np.zeros((I, K))
        Ru = np.zeros(K)
        Rl = np.zeros(K)

        for j in range(K):
            for i in range(I):
                yi = np.zeros(ns)
                yu = np.zeros(ns)
                for r in range(ns):
                    if x[k, i] <= m[j][i]:
                        t_low = b * (x[k, i] - m[j][i] + abs(theta_low[j][i, r]))
                        t_up = b * (x[k, i] - m[j][i] + abs(theta_up[j][i, r]))
                    else:
                        t_low = b * (m[j][i] - x[k, i] + abs(theta_low[j][i, r]))
                        t_up = b * (m[j][i] - x[k, i] + abs(theta_up[j][i, r]))
                    yi[r] = float(_sigmoid(t_low))
                    yu[r] = float(_sigmoid(t_up))
                Ql[i, j] = float(np.mean(yi))
                Qu[i, j] = float(np.mean(yu))
            Ru[j] = float(np.prod(Qu[:, j]))
            Rl[j] = float(np.prod(Ql[:, j]))

        sumrurl = float(np.sum(Ru) + np.sum(Rl))

        if k < fix_themodel and grow == 0:
            Ra = 0.5 * (Ru + Rl)
            index = int(np.argmax(Ra[:K]))
            wr[k] = index + 1
            omega_low_winner = omega_low[index]
            omega_up_winner = omega_up[index]
            theta = np.concatenate(
                [
                    omega_low_winner.reshape(M * (I + 1)),
                    omega_up_winner.reshape(M * (I + 1)),
                    qleft,
                    qright,
                    m[index],
                    theta_low[index].reshape(I * ns),
                    theta_up[index].reshape(I * ns),
                ]
            )
        elif k >= fix_themodel:
            wr[k] = wr[k - 1]

        Xe = np.concatenate([[1.0], x[k]])
        phi_low = np.zeros((M, K))
        phi_up = np.zeros((M, K))
        for o in range(M):
            for j in range(K):
                phi_low[o, j] = float(omega_low[j][o] @ Xe)
                phi_up[o, j] = float(omega_up[j][o] @ Xe)

        for o in range(M):
            yl = ((1 - qleft[o]) * (Rl @ phi_low[o]) + qleft[o] * (Ru @ phi_low[o]))
            yr = ((1 - qright[o]) * (Rl @ phi_up[o]) + qright[o] * (Ru @ phi_up[o]))
            y[k, o] = float(_safe_divide(yl + yr, sumrurl))

            if mode == 1:
                out[k, o] = 1.0 if y[k, o] > 0.5 else 0.0
            elif mode == 2:
                ww = int(np.argmax(y[k]))
                out[k, :] = 0.0
                out[k, ww] = 1.0

        e[k] = z[k] - y[k]

        if k < fix_themodel and grow == 0:
            dRlm = np.zeros(ns)
            dRum = np.zeros(ns)
            dRldtl = np.zeros((I, ns))
            dRudtu = np.zeros((I, ns))
            dRldm = np.zeros(I)
            dRudm = np.zeros(I)

            for i in range(I):
                Qlw_prod = _prod_except(Ql[:, index], i)
                Quw_prod = _prod_except(Qu[:, index], i)
                for r in range(ns):
                    t = -b * (x[k, i] - m[index][i] + abs(theta_low[index][i, r]))
                    drlm = b * float(_safe_exp(t)) / (1.0 + float(_safe_exp(t))) ** 2
                    t = -b * (x[k, i] - m[index][i] + abs(theta_up[index][i, r]))
                    drum = b * float(_safe_exp(t)) / (1.0 + float(_safe_exp(t))) ** 2

                    side_sign = -1.0 if x[k, i] <= m[index][i] else 1.0
                    dRlm[r] = side_sign * drlm
                    dRum[r] = side_sign * drum

                    theta_l = theta_low[index][i, r]
                    theta_u = theta_up[index][i, r]
                    theta_sign_l = 1.0 if theta_l >= 0 else -1.0
                    theta_sign_u = 1.0 if theta_u >= 0 else -1.0

                    t_l = -b * (x[k, i] - m[index][i] + abs(theta_l))
                    t_u = -b * (x[k, i] - m[index][i] + abs(theta_u))
                    base_l = b * float(_safe_exp(t_l)) / (1.0 + float(_safe_exp(t_l))) ** 2
                    base_u = b * float(_safe_exp(t_u)) / (1.0 + float(_safe_exp(t_u))) ** 2

                    sign_l = (1.0 if x[k, i] <= m[index][i] else -1.0) * theta_sign_l
                    sign_u = (1.0 if x[k, i] <= m[index][i] else -1.0) * theta_sign_u

                    C = (1.0 / ns) * sign_l * base_l
                    D = (1.0 / ns) * sign_u * base_u
                    dRldtl[i, r] = Qlw_prod * C
                    dRudtu[i, r] = Quw_prod * D

                dRldm[i] = Qlw_prod * float(np.mean(dRlm))
                dRudm[i] = Quw_prod * float(np.mean(dRum))

            dydw_low = np.zeros((I + 1, M))
            dydw_up = np.zeros((I + 1, M))
            dydql = np.zeros(M)
            dydqr = np.zeros(M)
            dyldRl = np.zeros(M)
            dyldRu = np.zeros(M)
            dyrdRl = np.zeros(M)
            dyrdRu = np.zeros(M)
            dymdm = np.zeros((I, M))
            dymdtl = np.zeros((I, ns, M))
            dymdtu = np.zeros((I, ns, M))

            for o in range(M):
                dydw_low[:, o] = ((1 - qleft[o]) * Rl[index] + qleft[o] * Ru[index]) * Xe / (sumrurl + EPS)
                dydw_up[:, o] = ((1 - qright[o]) * Rl[index] + qright[o] * Ru[index]) * Xe / (sumrurl + EPS)

                dydql[o] = ((-Rl @ phi_low[o]) + (Ru @ phi_low[o])) / (sumrurl + EPS)
                dydqr[o] = ((-Rl @ phi_up[o]) + (Ru @ phi_up[o])) / (sumrurl + EPS)

                dyldRl[o] = ((1 - qleft[o]) * phi_low[o, index]) / (sumrurl + EPS) - (
                    ((1 - qleft[o]) * (Rl @ phi_low[o]) + qleft[o] * (Ru @ phi_low[o]))
                    / ((sumrurl + EPS) ** 2)
                )
                dyldRu[o] = (qleft[o] * phi_low[o, index]) / (sumrurl + EPS) - (
                    ((1 - qleft[o]) * (Rl @ phi_low[o]) + qleft[o] * (Ru @ phi_low[o]))
                    / ((sumrurl + EPS) ** 2)
                )
                dyrdRl[o] = ((1 - qright[o]) * phi_up[o, index]) / (sumrurl + EPS) - (
                    ((1 - qright[o]) * (Rl @ phi_up[o]) + qright[o] * (Ru @ phi_up[o]))
                    / ((sumrurl + EPS) ** 2)
                )
                dyrdRu[o] = (qright[o] * phi_up[o, index]) / (sumrurl + EPS) - (
                    ((1 - qright[o]) * (Rl @ phi_up[o]) + qright[o] * (Ru @ phi_up[o]))
                    / ((sumrurl + EPS) ** 2)
                )

                dymdm[:, o] = (
                    dyldRl[o] * dRldm
                    + dyldRu[o] * dRudm
                    + dyrdRl[o] * dRldm
                    + dyrdRu[o] * dRudm
                )
                dymdtl[:, :, o] = dyldRl[o] * dRldtl + dyrdRl[o] * dRldtl
                dymdtu[:, :, o] = dyldRu[o] * dRudtu + dyrdRu[o] * dRudtu

            H = np.zeros((Z, M))
            offset = 0
            for o in range(M):
                H[offset + o * (I + 1) : offset + (o + 1) * (I + 1), o] = dydw_low[:, o]
            offset += M * (I + 1)

            for o in range(M):
                H[offset + o * (I + 1) : offset + (o + 1) * (I + 1), o] = dydw_up[:, o]
            offset += M * (I + 1)

            for o in range(M):
                H[offset + o, o] = dydql[o]
            offset += M

            for o in range(M):
                H[offset + o, o] = dydqr[o]
            offset += M

            H[offset : offset + I, :] = dymdm
            offset += I

            for o in range(M):
                H[offset : offset + I * ns, o] = dymdtl[:, :, o].reshape(I * ns)
            offset += I * ns

            for o in range(M):
                H[offset : offset + I * ns, o] = dymdtu[:, :, o].reshape(I * ns)

            G = _dekf_gain(P[index], H, R)
            P[index] = (np.eye(Z) - G @ H.T) @ P[index]
            theta = theta + G @ e[k]

            ptr = 0
            omega_low[index] = theta[ptr : ptr + M * (I + 1)].reshape(M, I + 1)
            ptr += M * (I + 1)
            omega_up[index] = theta[ptr : ptr + M * (I + 1)].reshape(M, I + 1)
            ptr += M * (I + 1)
            nomega_up[index] = float(np.linalg.norm(omega_up[index]))
            nomega_low[index] = float(np.linalg.norm(omega_low[index]))

            qleft = theta[ptr : ptr + M]
            ptr += M
            qright = theta[ptr : ptr + M]
            ptr += M
            m[index] = theta[ptr : ptr + I]
            ptr += I
            theta_low[index] = theta[ptr : ptr + I * ns].reshape(I, ns)
            ptr += I * ns
            theta_up[index] = theta[ptr : ptr + I * ns].reshape(I, ns)

            min_low = np.min(np.abs(theta_low[index]), axis=1) * ((ns + 1) / 2)
            min_up = np.min(np.abs(theta_up[index]), axis=1) * ((ns + 1) / 2)
            sigma_low[index] = _cov_floor(min_low**2)
            sigma_up[index] = _cov_floor(min_up**2)

            if verbose:
                print(f"The rule no {index + 1} is UPDATED around sample {k + 1}")

    y[0] = z[0]
    e[0] = 0.0

    if mode in (1, 2):
        y = out.copy()
        if mode == 1 and M >= 1:
            y[0, 0] = 1.0 if y[0, 0] > 0.5 else 0.0
        elif mode == 2:
            ww = int(np.argmax(y[0]))
            y[0] = 0.0
            y[0, ww] = 1.0

    rule = RuleInfo(nrule=nrule, K=K, wr=wr)
    network = NetworkInfo(
        I=I,
        miu=np.column_stack(m[:K]),
        ns=ns,
        b=b,
        theta_up=np.stack(theta_up[:K], axis=2),
        theta_low=np.stack(theta_low[:K], axis=2),
        omega_up=np.stack(omega_up[:K], axis=2),
        omega_low=np.stack(omega_low[:K], axis=2),
        qleft=qleft,
        qright=qright,
    )

    performance: Dict[str, float | np.ndarray] = {}
    elapsed = time.time() - start_time
    performance["time"] = elapsed

    if mode == 3:
        if M == 1:
            mse_train = float(np.mean(e[:fix_themodel, 0] ** 2))
            rmse_train = float(np.sqrt(mse_train))
            ndei_train = float(_safe_divide(rmse_train, np.std(z[:fix_themodel, 0]) + EPS))

            mse_test = float(np.mean(e[fix_themodel:, 0] ** 2))
            rmse_test = float(np.sqrt(mse_test))
            ndei_test = float(_safe_divide(rmse_test, np.std(z[fix_themodel:, 0]) + EPS))

            mse_total = float(np.mean(e[:, 0] ** 2))
            rmse_total = float(np.sqrt(mse_total))
            ndei_total = float(_safe_divide(rmse_total, np.std(z[:, 0]) + EPS))

            if verbose:
                print("Number of data | MSE | RMSE | NDEI")
                print(f"Train: {fix_themodel} | {mse_train:.6f} | {rmse_train:.6f} | {ndei_train:.6f}")
                print(f"Valid: {N - fix_themodel} | {mse_test:.6f} | {rmse_test:.6f} | {ndei_test:.6f}")
                print(f"Total: {N} | {mse_total:.6f} | {rmse_total:.6f} | {ndei_total:.6f}")

            performance["RMSE_training"] = rmse_train
            performance["NDEI_training"] = ndei_train
            performance["RMSE_validation"] = rmse_test
            performance["NDEI_validation"] = ndei_test
        else:
            rmse_train = np.zeros(M)
            ndei_train = np.zeros(M)
            rmse_test = np.zeros(M)
            ndei_test = np.zeros(M)
            for o in range(M):
                mse_train = float(np.mean(e[:fix_themodel, o] ** 2))
                rmse_train[o] = float(np.sqrt(mse_train))
                ndei_train[o] = float(_safe_divide(rmse_train[o], np.std(z[:fix_themodel, o]) + EPS))

                mse_test = float(np.mean(e[fix_themodel:, o] ** 2))
                rmse_test[o] = float(np.sqrt(mse_test))
                ndei_test[o] = float(_safe_divide(rmse_test[o], np.std(z[fix_themodel:, o]) + EPS))

                mse_total = float(np.mean(e[:, o] ** 2))
                rmse_total = float(np.sqrt(mse_total))
                ndei_total = float(_safe_divide(rmse_total, np.std(z[:, o]) + EPS))

                if verbose:
                    print(f"Output {o + 1}")
                    print("Number of data | MSE | RMSE | NDEI")
                    print(f"Train: {fix_themodel} | {mse_train:.6f} | {rmse_train[o]:.6f} | {ndei_train[o]:.6f}")
                    print(f"Valid: {N - fix_themodel} | {mse_test:.6f} | {rmse_test[o]:.6f} | {ndei_test[o]:.6f}")
                    print(f"Total: {N} | {mse_total:.6f} | {rmse_total:.6f} | {ndei_total:.6f}")

            performance["RMSE_training"] = rmse_train
            performance["NDEI_training"] = ndei_train
            performance["RMSE_validation"] = rmse_test
            performance["NDEI_validation"] = ndei_test

        if verbose:
            print(f"Rule: {K} | Computation_Time: {elapsed:.4f}s")
    else:
        count_train = 0
        for k in range(fix_themodel):
            if np.array_equal(z[k], y[k]):
                count_train += 1
        count_valid = 0
        for k in range(fix_themodel, N):
            if np.array_equal(z[k], y[k]):
                count_valid += 1

        rate_train = float(_safe_divide(count_train, fix_themodel))
        rate_valid = float(_safe_divide(count_valid, N - fix_themodel))

        if verbose:
            print("Number of data | Classification rate")
            print(f"Train: {fix_themodel} | {rate_train:.6f}")
            print(f"Valid: {N - fix_themodel} | {rate_valid:.6f}")
            print(f"Rule: {K} | Computation_Time: {elapsed:.4f}s")

        performance["classification_rate_training"] = rate_train
        performance["classification_rate_validation"] = rate_valid

    if plot and mode == 3:
        import matplotlib.pyplot as plt

        def _shade_train_valid(ax, split_index: int, total: int) -> None:
            train_end = max(0, min(split_index, total))
            total = max(train_end, total)
            ax.axvspan(0, train_end, color="#d8f3dc", alpha=0.25, zorder=0)
            ax.axvspan(train_end, total, color="#ffe5e5", alpha=0.25, zorder=0)

        plt.figure()
        plt.plot(wr)
        _shade_train_valid(plt.gca(), fix_themodel, N)
        plt.title("winning rule")
        plt.xlabel("iteration")
        plt.ylabel("Rule no.")
        plt.xlim([0, N])

        for o in range(M):
            plt.figure()
            plt.plot(e[:, o])
            _shade_train_valid(plt.gca(), fix_themodel, N)
            plt.xlabel("iteration")
            plt.ylabel("error")
            plt.xlim([0, N])

        for o in range(M):
            plt.figure()
            plt.plot(z[:, o], "-b", linewidth=0.5)
            plt.plot(y[:, o], "-.r", linewidth=1)
            _shade_train_valid(plt.gca(), fix_themodel, N)
            plt.xlabel("iteration")
            plt.ylabel("y")
            plt.legend(["target", "output"], loc="best")
            plt.xlim([0, N])
            plt.ylim([np.min(y[:, o]) - 0.1, np.max(y[:, o]) + 0.1])
        plt.show()

    if verbose:
        print("eT2QFNN: finished")

    return y, rule.__dict__, network.__dict__, performance
