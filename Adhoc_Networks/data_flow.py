import numpy as np

FLOW_COLORS = ['tab:cyan', 'tab:purple', 'tab:pink']*10

class Data_Flow():
    def __init__(self, flow_id, src, dest):
        self.flow_id = flow_id
        self.src = src
        self.dest = dest
        self._links = []
        self.exclude_nodes = None
        self.frontier_node = src
        self.bottleneck_rate = -1
        self._n_reprobes = 0 # number of reprobes during routing this flow
        self.plot_color = FLOW_COLORS[flow_id]

    def add_link(self, tx, band, rx, state, action):
        assert tx == self.frontier_node
        self._links.append((tx, band, rx, state, action))
        if tx==rx: # it's a reprobe
            self._n_reprobes += 1
        else:
            self.frontier_node = rx
            self.exclude_nodes = np.append(self.exclude_nodes, rx)
        return

    def get_links(self):
        return self._links.copy() # make sure it's not altered from outside


    def destination_reached(self):
        return self.frontier_node == self.dest

    def get_number_of_reprobes(self):
        return self._n_reprobes

    def reset(self):
        self._links = []
        self.exclude_nodes = None
        self.frontier_node = self.src
        self.bottleneck_rate = None
        self._n_reprobes = 0
        return
