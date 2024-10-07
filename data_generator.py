import numpy as np
import matplotlib.pyplot as plt
import json as js
import argparse
import random

from data.noninteraction import Energy_band
from data.interaction import Lattice


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# 计算最小的周期
def count_freq(kspace_size, lattice_height):
    omega = np.pi**2/2*(kspace_size**2+lattice_height/2)
    return f"omega:{omega}, T:{1/omega}"


# 生成非相互作用的数据
def noninteraction_generate():
    with open("./settings/noninteraction.json") as f:
        gene_set = js.load(f)
    gene_set = dict2namespace(gene_set)

    dataformat = gene_set.dataformat
    Mass = dataformat.Mass
    hbar = dataformat.hbar
    E_R = (hbar*np.pi)**2/2/Mass
    k_list = np.linspace(-np.pi, np.pi, dataformat.k_list_num)

    V0_max, V0_min = dataformat.lattice_height.Max, \
        dataformat.lattice_height.Min
    n_V0 = gene_set.n_lattice_height
    rd = np.random.random(n_V0)
    V0_list = (rd*(V0_max-V0_min)+V0_min*np.ones(n_V0))*E_R

    bandlist = []
    print("count energy band:")
    for i in range(n_V0):
        if i % 50 == 0:
            print(f"band{i}")
        bandlist.append(Energy_band(dataformat.kspace_size,
                                    lattice_height=V0_list[i],
                                    k_list=k_list))

    step = gene_set.time.step
    tlength = gene_set.time.length
    ts = np.arange(tlength)*step

    n_weight = gene_set.n_time_evolution  # 对于每种光晶格要选取多少个初始能量分布
    data = np.zeros([n_V0*n_weight, tlength, dataformat.kspace_size])  # 数据的大小

    print("time evolution:")
    for i in range(n_V0):
        if i % 50 == 0:
            print(f"band{i}")
        for j in range(n_weight):
            while True:
                real, imag = np.random.random(dataformat.kspace_size), \
                    np.random.random(dataformat.kspace_size)
                norm = np.sqrt((real**2+imag**2).sum())
                if norm != 0:
                    break
            k_weight0 = (real+1j*imag)/norm  # 选取初始动量分布

            ki = np.random.randint(0, dataformat.k_list_num)
            band_weight, k_weight =\
                bandlist[i].time_evolution(ki, k_weight0, ts)

            # Energy_band.bandk_evolution_plot(ts,band_weight,k_weight)
            # plt.show()
            # exit()
            population = Energy_band.ignore_phase(k_weight)

            idx = i*n_weight+j
            data[idx, :, :] = population

    np.save("./data/noninteraction_data.npy", data)
    # np.save("./data/noninteraction_data_energy.npy", V0_list)
    # np.save("./data/data_evolution_test.npy",data)
    # np.save("./data/data_energy_test.npy",V0_list)


# 生成相互作用的数据
def interaction_generate():
    with open("./settings/interaction.json") as f:
        gene_set = js.load(f)
    gene_set = dict2namespace(gene_set)

    M = gene_set.M
    N = gene_set.N
    lambs = gene_set.lambs
    n_lamb = gene_set.n_lamb
    n_time_evolution = gene_set.n_time_evolution

    time_set = gene_set.time
    step, length = time_set.step, time_set.length
    t = np.arange(0, step*length, step)

    data = np.zeros([n_lamb*n_time_evolution, length, M])
    for i in range(n_lamb):
        if i % 20 == 0:
            print(f"band{i}")
        band = Lattice(M, N, random.uniform(lambs.Min, lambs.Max))
        for j in range(n_time_evolution):
            # 波函数
            psi_evolution = band.count_evolution(t, "Mott insulating")
            # 玻色子数量
            num_evolution = np.square(np.abs(psi_evolution))
            # 概率分布
            prob_evolution = num_evolution/M
            idx = i*n_time_evolution + j
            data[idx, :, :] = prob_evolution

    np.save("./data/interaction_data.npy", data)


if __name__ == "__main__":
    noninteraction_generate()
    interaction_generate()
