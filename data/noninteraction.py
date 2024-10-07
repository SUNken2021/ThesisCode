# 无相互作用系统能带的计算

import numpy as np
import matplotlib.pyplot as plt

# 质量、约化普朗克常数、归一化能量单位
Mass = 1
hbar = 1
E_R = (hbar*np.pi)**2/2/Mass
# 计算的相邻布里渊区的数量
kspace_size = 101


class Energy_band():
    def __init__(self, kspace_size, lattice_height,
                 k_list=np.linspace(-np.pi, np.pi, 201)):
        self.kspace_size = kspace_size  # 截断后的k空间包含的布里渊区的数量
        self.lattice_height = lattice_height  # 光晶格深度
        potential_list = lattice_height/2*(
            np.cos(2*np.pi*np.linspace(0, 1, kspace_size))+1)  # 势能曲线

        self.k_list = k_list  # 单个布里渊区中取的所有k值

        # 计算势能矩阵
        self.V_matrix = self.potential_matrix(potential_list)

        # 计算所有k值对应的哈密顿量及其本征值和本征向量
        self.count_eigen()

    def potential_matrix(self, potential_list):
        # 通过傅里叶变化将势能函数转化为势能矩阵
        length = self.kspace_size
        fft_potential = np.fft.fft(potential_list)/length
        freq = np.fft.fftfreq(np.size(potential_list, 0), 1)
        index = np.argsort(freq)
        fft_potential = np.real(fft_potential[index])

        V = np.zeros((length, length))
        index_max = length//2
        for i in range(length):
            for j in range(length):
                t = i-j+index_max
                if t >= 0 and t < length:
                    V[i, j] = fft_potential[t]
        return V

    def dynamic_matrix(self, k):
        # 动能矩阵
        length = self.kspace_size
        index_max = length//2
        diag = 2*np.pi*np.linspace(-index_max, index_max, length) + k
        diag = diag*hbar

        return np.diag((diag)**2/2/Mass)

    def count_eigen(self):
        length = self.kspace_size
        k_len = self.k_list.size

        self.Hamitonian_matrixs = np.zeros((k_len, length, length))
        self.eigen_values = np.zeros((k_len, length))
        self.eigen_vector = np.zeros((k_len, length, length))

        for i in range(k_len):
            k = self.k_list[i]
            self.Hamitonian_matrixs[i, :, :] = self.dynamic_matrix(k) +\
                self.V_matrix
            value, vector = np.linalg.eig(self.Hamitonian_matrixs[i, :, :])

            index = np.argsort(value)  # 对特征值进行排序
            self.eigen_values[i, :] = value[index]
            self.eigen_vector[i, :, :] = np.transpose(vector)[index, :]

    def band_plot(self, n_bands=5):
        for i in range(n_bands):
            plt.plot(self.k_list/np.pi, self.eigen_values[:, i]/E_R,
                     label=f"Band {i}")

        plt.plot(self.k_list/np.pi,
                 self.lattice_height/E_R*np.ones_like(self.k_list),
                 label="Lattice Depth", linestyle="--")

        plt.xlabel("Quasi-momentum $q/k$")
        plt.ylabel("Energy $E/E_r$")
        plt.legend(loc="upper right")

    def band_population_plot(self,ki,n,judge=True):
        q=self.k_list[ki]/np.pi
        self.population_plot(q,self.eigen_vector[ki,n,:])
        plt.xlabel("Momentum($\hbar k$)")
        if judge:
            plt.ylabel("Probablity")
        plt.title(f"$q$=0, Band {n}")
        # plt.title(f"$q$={np.around(q,decimals=0)}, Band {n}")

    def population_plot(self,q,k_weight):
        length=self.kspace_size
        index_max=length//2
        x=np.linspace(q-index_max*2,q+index_max*2,length)
        population=np.conj(k_weight)*k_weight
        plt.ylim([0,1.0])
        plt.bar(x,population,color='black',width=1.6)
        plt.xticks(x)

    def time_evolution(self,ki,k_weight0,ts):
        t_num=ts.size
        eigen_vector=self.eigen_vector[ki,:,:]
        eigen_value=self.eigen_values[ki,:]

        band_weight0=np.dot(k_weight0,np.linalg.inv(eigen_vector))

        band_weight=np.zeros((t_num,self.kspace_size),dtype=complex)
        for i in range(self.kspace_size):
            E_i=eigen_value[i]
            band_weight[:,i]=np.transpose(np.exp(-E_i*ts/hbar*1j))*band_weight0[i]

        k_weight=np.dot(band_weight,eigen_vector)
        return band_weight,k_weight

    def ignore_phase(k_weight):
        population=np.abs(k_weight)**2
        return population

    def bandk_evolution_plot(ts,band_weight,k_weight,q):
        _kspace_size=k_weight.shape[1]
        plt.subplot(1,2,1)
        for i in range(_kspace_size):
            plt.plot(ts,np.abs(band_weight[:,i])**2,label=f"{i}")
        plt.legend()

        plt.subplot(1,2,2)
        population=Energy_band.ignore_phase(k_weight)

        index_max=_kspace_size//2

        momentum=np.linspace(-index_max*2+q,index_max*2+q,_kspace_size)

        for i in range(0, kspace_size):
            plt.plot(ts,population[:,i],label=f"{momentum[i]}$\hbar k$")
        plt.legend()
