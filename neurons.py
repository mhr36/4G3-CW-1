import numpy as np


class Population:
    '''
    Generic neuron population. Includes methods for processing trackers.

    add_tracker(): Add a tracker object to record information

    step(): Update method called on each time step

    '''

    takes_input = False

    def __init__(self, N=1000, delta_time=1e-4):
        self.N = N
        self.delta_time = delta_time
        self.output = np.zeros(self.N)
        self.trackers = []

    def set_output(self, out):
        self.output = out
        return out

    def add_tracker(self, tracker):
        self.trackers.append(tracker)

    def init_trackers(self, k):
        for tracker in self.trackers:
            tracker.init(k)

    def step(self):
        for tracker in self.trackers:
            tracker.track()

    def receive(self):
        pass


class XNeurons(Population):
    '''
    Input neuron population. Produces Poisson output with the specified rate.
    '''

    def __init__(self, N=1000, delta_time=1e-4, rate=10):
        super().__init__(N, delta_time)
        self.rate = rate

    def step(self):
        super().step()
        self.set_output(np.random.binomial(1, self.rate * self.delta_time, self.N) / self.delta_time)
        return self.output


class LIFNeurons(Population):
    '''
    Integrate and fire neuron population.

    link(source, J, K): Add a source population which inputs into this population. J is synaptic weight and K is number of random connections per neuron.
    '''

    takes_input = True

    def __init__(self, N=1000, delta_time=1e-4, tau=2e-2, V_threshold=1):
        super().__init__(N, delta_time)
        self.tau = tau
        self.net_input = np.zeros(self.N)

        self.input_populations = []
        self.C_inputs = []

        self.V = np.zeros(self.N)
        self.V_threshold = V_threshold

    def add_input_population(self, population, C):
        self.input_populations.append(population)
        self.C_inputs.append(np.array(C))

    def receive(self):
        self.net_input = np.zeros(self.N)

        for i, pop in enumerate(self.input_populations):
            self.net_input += self.C_inputs[i] @ pop.output

    def step(self):
        super().step()

        self.V *= (1 - (self.delta_time / self.tau))
        self.V += self.net_input * self.delta_time
        spikes = self.clip()

        return self.set_output(spikes)

    def clip(self):
        exceeded = self.V >= self.V_threshold
        #negative = self.V < 0

        self.V[exceeded] = 0
        #self.V[negative] = 0

        return exceeded / self.delta_time

    def link(self, source, J, K):
        C = J * make_C(self.N, source.N, K) / np.sqrt(K)
        self.add_input_population(source, C)


class NonResettingLIFNeurons(LIFNeurons):
    def clip(self):
        exceeded = self.V >= self.V_threshold
        return exceeded / self.delta_time


def make_C(N_dest=100, N_source=100, K=10):
    '''Randomly produce a matrix of zeros with K ones per row.'''
    C = np.zeros((N_dest, N_source))
    for row in C:
        row.put(np.random.choice(N_source, K, False), 1)
    return C