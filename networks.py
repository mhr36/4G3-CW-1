import numpy as np
from tqdm import tqdm

class Network:
    '''
    Takes a list of neuron populations and simulates their behaviour using integrate and fire model.
    '''

    def __init__(self, populations, delta_time=1e-4):
        self.populations = populations
        self.delta_time = delta_time
        self.time_axis = []

        self.EI_populations = []
        for pop in self.populations:
            pop.delta_time = self.delta_time

            if pop.takes_input:
                self.EI_populations.append(pop)

    def run(self, k=1):
        self.time_axis = np.linspace(0, k*self.delta_time, k+1)

        for pop in self.populations:
            pop.init_trackers(k)

        for i in tqdm(range(k)):
            for pop in self.populations:
                pop.receive()

            for pop in self.populations:
                pop.step()

        print("\nSimulation complete\n")
    
    def run_for(self, time):
        k = int(time / self.delta_time)
        self.run(k)
