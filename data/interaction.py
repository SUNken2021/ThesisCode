import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Lattice:
    def __init__(self, M, N, lamb, epsilon=0):
        self.M = M  # 光晶格的数量
        self.N = N  # 每个光晶格中玻色子数量的平均值
        self.lamb = lamb  # 单个光晶格内部玻色子相互作用强度
        self.set_potential(epsilon)  # 计算约束光晶格的势场

    def set_potential(self, epsilon):
        self.epsilon = epsilon  # 约束光晶格的势场强度
        self.V = np.zeros([self.M])
        mid = self.M//2
        for i in range(self.M):
            self.V[i] = epsilon/2 * (i - mid)**2

    # 生成初始态
    # Ni:各晶格初始态的玻色子数量, type:初始态的类型
    def initial_state(self, Ni, type: str):
        Ni = np.array(Ni)
        if type == "Mott insulating":
            phi = np.random.uniform(0, 2*np.pi, (self.M))
        elif type == "Modulated phase":
            phi = np.arange(0, self.M)*np.pi
        self.n_0 = np.sqrt(Ni/self.N)  # 模长
        self.phi_0 = phi  # 相位
        self.psi_0 = self.n_0*np.exp(1j*self.phi_0)  # 波函数

    def psi2real(self, psi):
        return np.concatenate((psi.real, psi.imag))

    def real2psi(self, real):
        return real[:self.M] + 1j*real[self.M:]

    # 演化微分方程(实数版)
    def func(self, real_psi, t):
        diff = np.zeros_like(real_psi)
        psi = self.real2psi(real_psi)  # 转换为复数
        for i in range(self.M):
            # 相互作用项
            interaction = (psi[self.M-1] if i == 0 else psi[i-1]) +\
                (psi[0] if i == self.M-1 else psi[i+1])
            # 势能项(多个光晶格情形)
            potential = self.V[i] * psi[i]
            # 单个光晶格内部相互作用
            energy = self.lamb * psi[i] * (abs(psi[i])**2)
            # 导数项
            _diff = -1j*(potential + energy - interaction)
            diff[i] = _diff.real
            diff[i+self.M] = _diff.imag
        return diff

    def cross_conj(self, a, b):
        return np.conj(a)*b + np.conj(b)*a

    def Dgt(self, psit, g="nearest"):
        D = np.zeros([psit.shape[0]])
        if g == "nearest":
            for i in range(self.M-1):
                D = D + self.cross_conj(psit[:, i], psit[:, i+1])
            D = D + self.cross_conj(psit[:, 0], psit[:, self.M-1])
            D = D/self.lamb
        elif g == "variance":
            for i in range(self.M):
                D = D + (abs(psit[:, i])**2-1)**2
            D = D*self.lamb/2
        elif g == "global":
            for i in range(self.M):
                for j in range(i+1, self.M):
                    D = D + self.cross_conj(psit[:, i], psit[:, j])
            D = D/self.lamb
        elif g == "ring":
            for i in range(self.M-1):
                D = D + self.cross_conj(psit[:, i], psit[:, i+1])
            D = D + self.cross_conj(psit[:, 0], psit[:, self.M-1])
            D = D/self.lamb
        return D/self.M
        # return D

    # 计算相关程度
    def count_relation(self, t, n_time, g="nearest", type="Mott insulating"):
        # Ni = np.zeros([self.M])
        # Ni[0] = self.N
        Ni = np.ones([self.M])*self.N

        D = np.zeros_like(t, dtype=float)
        n_sampling = n_time if type == "Mott insulating" else int(1e2)
        n_report = int(5e2)
        for i in range(n_sampling):
            if (i+1) % n_report == 0:
                print(i+1)
            self.initial_state(Ni, type)
            result = odeint(self.func, self.psi2real(self.psi_0), t)
            result = result[:, :self.M] + 1j*result[:, self.M:]
            _D = self.Dgt(result, g)
            D = (D*i+_D)/(i+1)
        D = D - np.ones_like(D)*D[0]
        return abs(D)

    def count_n2(self, t, type="test"):
        Ni = np.ones([self.M])*self.N  # 初始态平均分布
        n2 = np.zeros_like(t, dtype=complex)  # 平均的n2

        n_sampling = int(1e3)
        n_report = int(1e2)
        for i in range(n_sampling):
            if (i+1) % n_report == 0:
                print(i+1)

            self.initial_state(Ni, "Mott insulating")
            # self.initial_state(Ni, "Modulated phase")
            result = odeint(self.func, self.psi2real(self.psi_0), t)
            result = result[:, :self.M] + 1j*result[:, self.M:]
            if type == "test":
                _n2 = (
                    np.ones_like(result[:, 0]) -
                    abs(result[:, 0])**2
                )**2
            else:
                _n2 = self.cross_conj(result[:, 0], result[:, 1])/self.lamb
            n2 = (n2*i+_n2)/(i+1)
        return n2.real

    def count_evolution(self, t, type):
        Ni = np.ones([self.M])*self.N
        self.initial_state(Ni, type)
        result = odeint(self.func, self.psi2real(self.psi_0), t)
        result = result[:, :self.M] + 1j*result[:, self.M:]
        return result
