import neurons
import networks
import trackers

import matplotlib.pyplot as plt
import numpy as np


def task_2():
    X = neurons.XNeurons()
    net = networks.Network([X])

    spike = trackers.SpikeTracker(X)

    net.run_for(2)

    plt.figure(figsize=(20,12))
        
    plt.imshow(spike.data, aspect='auto', cmap='Greys',  interpolation='nearest')
    plt.xlabel("Time Step")
    plt.ylabel("Input Neuron")
    plt.show()

    totals = np.mean(np.sum(spike.data, axis=1) * X.delta_time / 2)
    print("Mean rate (Hz): " + str(totals))


def task_3():
    X = neurons.XNeurons(N=1)
    E = neurons.LIFNeurons(N=1)

    E.add_input_population(X, [[.9]])

    voltage = trackers.VoltageTracker(E)
    spike_X = trackers.SpikeTracker(X)
    spike_E = trackers.SpikeTracker(E)

    net = networks.Network([X, E])
    net.run(20000)

    plt.plot(voltage.data[0])
    plt.plot(spike_E.data[0], linewidth=0.2)
    plt.ylim([0, 1])
    plt.xlabel("Time Step")
    plt.ylabel("Membrane Potential")
    plt.show()

    #plt.imshow([spike_X.data[0], spike_E.data[0]], interpolation='nearest', aspect='auto')
    x = np.where(spike_X.data[0]>0)
    e = np.where(spike_E.data[0]>0)
    plt.scatter(x, np.zeros_like(x))
    plt.scatter(e, np.ones_like(e))
    plt.xlabel("Time Step")
    plt.legend(("Input Spike Train", "Output Spike Train"))
    plt.show()


def task_4_1(w=1, K=100):
    # K = int(input("Choose K: "))

    X = neurons.XNeurons(N=K)
    E = neurons.NonResettingLIFNeurons(N=1)

    C = np.ones([1, K]) * w/K
    E.add_input_population(X, C)
    voltage = trackers.VoltageTracker(E)

    net = networks.Network([X, E])
    net.run_for(2)

    plt.plot(net.time_axis, voltage.data[0])
    plt.xlabel("Time (s)")
    plt.ylabel("Membrane Potential")
    plt.show()


def mu_prediction(K=1, w=1, tau=2e-2, rate=10):
    return tau * rate * w + (K*0)


def var_prediction(K=1, w=1, tau=2e-2, rate=10):
    return (tau * rate * w**2) / (2 * K)


def check_mu_var(K=1, sim_time=10, w=1):
    X = neurons.XNeurons(N=K)
    E = neurons.NonResettingLIFNeurons(N=1)

    C = np.ones([1, K]) * w/K
    E.add_input_population(X, C)
    voltage = trackers.VoltageTracker(E)

    net = networks.Network([X, E])
    net.run_for(sim_time)

    trn = int(0.1 / net.delta_time)

    ss = np.var(voltage.data[0, trn:])
    mu = np.mean(voltage.data[0, trn:])

    return mu, ss


def task_4_3(resolution=100, w=1):
    K_range = [1, 10, 100, 1000]
    K_axis = np.geomspace(0.5, 1000, resolution+1)

    plt.plot(K_axis, mu_prediction(K=K_axis, w=w))
    plt.plot(K_axis, var_prediction(K=K_axis, w=w))

    mu_list = []
    ss_list = []

    for K in K_range:
        print("Simulating with K = " + str(K))
        mu, ss = check_mu_var(K=K, w=w)

        mu_list.append(mu)
        ss_list.append(ss)

    plt.scatter(K_range, mu_list)
    plt.scatter(K_range, ss_list)

    plt.ylim([0, w*.3])
    plt.xscale('log')
    plt.legend(("Mean Prediction", "Variance Prediction", "Mean Simulation", "Variance Simulation"))
    plt.xlabel("K")

    plt.show()


def task_4_4():
    task_4_1(w=5)


def task_4_5(w=4.25, K=100, sim_time=10, window_time=0.1):
    X = neurons.XNeurons(N=K)
    E = neurons.LIFNeurons(N=1)

    C = np.ones([1, K]) * w/K
    E.add_input_population(X, C)
    voltage = trackers.VoltageTracker(E)
    spike = trackers.SpikeTracker(E)

    net = networks.Network([X, E])
    net.run_for(sim_time)

    firing_rate = np.count_nonzero(spike.data) / sim_time

    window_size = int(window_time / net.delta_time)
    moving_sum = np.convolve(spike.data[0] * net.delta_time, np.ones(window_size))
    fano = np.var(moving_sum) / np.mean(moving_sum)

    print("Firing Rate: " + str(firing_rate) + " Hz")
    print("Fano Factor: " + str(fano))

    plt.plot(net.time_axis, voltage.data[0])
    plt.plot(net.time_axis, spike.data[0], linewidth=0.2)
    plt.ylim([0,1])
    plt.ylabel("Membrane Potential")
    plt.xlabel("Time (s)")

    plt.show()


def task_5(w=12, K=100, sim_time=10, window_time=0.1):
    XE = neurons.XNeurons(N=K)
    XI = neurons.XNeurons(N=K)
    E = neurons.LIFNeurons(N=1)

    C = np.ones([1, K]) * w/K
    E.add_input_population(XE, C)
    E.add_input_population(XI, -C)

    voltage = trackers.VoltageTracker(E)
    spike = trackers.SpikeTracker(E)

    net = networks.Network([XE, XI, E])
    net.run_for(sim_time)

    firing_rate = np.count_nonzero(spike.data) / sim_time

    window_size = int(window_time / net.delta_time)
    moving_sum = np.convolve(spike.data[0] * net.delta_time, np.ones(window_size))
    fano = np.var(moving_sum) / np.mean(moving_sum)

    print("Firing Rate: " + str(firing_rate) + " Hz")
    print("Fano Factor: " + str(fano))

    plt.plot(net.time_axis, voltage.data[0])
    plt.plot(net.time_axis, spike.data[0], linewidth=0.2)
    plt.ylim([-1.2, 1.2])
    plt.ylabel("Membrane Potential")
    plt.xlabel("Time (s)")

    plt.show()


def task_6_1(rx=10, tau=2e-2, K=100, J_EE=2, J_IE=4.4, J_EI=-2.5, J_II=-3, J_EX=1, J_IX=0.2):
    Jr = np.array([[J_EE, J_EI], [J_IE, J_II]])
    Jx = np.array([J_EX, J_IX])

    D = np.eye(2) / (tau * np.sqrt(K))

    Rr = np.linalg.solve(D - Jr, Jx * rx)

    print(Rr)


def task_6_2(N=1000, K=100, sim_time=2, rx=10, J_EE=2, J_IE=4.4, J_EI=-2.5, J_II=-3, J_EX=1, J_IX=0.2):
    X = neurons.XNeurons(N, rate=rx)
    E = neurons.LIFNeurons(N)
    I = neurons.LIFNeurons(N)

    E.link(X, J_EX, K)
    E.link(I, J_EI, K)
    E.link(E, J_EE, K)

    I.link(X, J_IX, K)
    I.link(I, J_II, K)
    I.link(E, J_IE, K)

    spike_E = trackers.SpikeTracker(E)
    spike_I = trackers.SpikeTracker(I)

    net = networks.Network([X, E, I])
    net.run_for(sim_time)

    rate_E = np.count_nonzero(spike_E.data) / (N * sim_time)
    rate_I = np.count_nonzero(spike_I.data) / (N * sim_time)

    print("Mean excitatory firing rate: " + str(rate_E) + "Hz")
    print("Mean inhibitory firing rate: " + str(rate_I) + "Hz")

    return (rate_E, rate_I)


def task_6_3():
    rx_range = [1, 5, 10, 20]
    results = np.zeros((len(rx_range), 2))

    for i, rx in enumerate(rx_range):
        print(rx)

        results[i, :] = task_6_2(rx=rx)

    plt.plot(rx_range, results)
    plt.ylabel("Firing Rate (Hz)")
    plt.xlabel("Input rate rx (Hz)")
    plt.legend(("Exitatory Neurons", "Inhibitory Neurons"))
    plt.show()


def task_6_4():
    N = 300
    K_range = np.linspace(10, 250, 11, dtype=np.int32)
    results = np.zeros((len(K_range), 2))

    for i, K in enumerate(K_range):
        print(N, K)

        results[i, :] = task_6_2(N=N, K=K)

    plt.plot(K_range, results)
    plt.ylabel("Firing Rate (Hz)")
    plt.xlabel("Inputs Per Neuron K")
    plt.legend(("Exitatory Neurons", "Inhibitory Neurons"))
    plt.show()


def task_6_5():
    N_range = [40, 50, 60, 70, 80, 100, 250, 500, 1000]
    results = np.zeros((len(N_range), 2))

    for i, N in enumerate(N_range):
        print(N, N//10)

        results[i, :] = task_6_2(N=N, K=40)

    plt.plot(N_range, results)
    plt.ylabel("Firing Rate (Hz)")
    plt.xlabel("Population Size N")
    plt.legend(("Exitatory Neurons", "Inhibitory Neurons"))
    plt.show()


if __name__ == '__main__':
    task_6_5()

# Firing rate 9.9
# 4.5 fano factor 0.42 firing rate 9.8
# 5 fano factor 1.06 firing rate 5.1
# 6 E 4.70 I 9.86