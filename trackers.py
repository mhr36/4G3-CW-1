import numpy as np

class Tracker:
    def __init__(self, population):
        self.source = population
        population.add_tracker(self)

    def init(self, k):
        self.data = np.zeros([self.source.N, k+1])
        self.pointer = 0
        self.track()

    def track(self):
        self.data[:, self.pointer] = self.get_source_state()
        self.pointer += 1

    def get_source_state(self):
        return np.zeros([self.source.N])

class VoltageTracker(Tracker):
    def get_source_state(self):
        return self.source.V

class SpikeTracker(Tracker):
    def get_source_state(self):
        return self.source.output