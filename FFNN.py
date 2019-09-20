import Layer

class FFNN:
    def __init__(self, batch_size=1, n_hidden_layers=1, nb_nodes=2, learning_rate=0.1, momentum=0.1, epoch=1):
        self.batch_size = batch_size
        self.n_hidden_layers = n_hidden_layers
        self.nb_nodes = nb_nodes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epoch = epoch

    def fit(self, data):
        # Get number of features
        n_rows = data.shape[0]
        n_features = data.shape[1]
        # Create list of layers (while initiating random weights)
        self.layer_list = []
        self.layer_list.append(Layer(n_neuron=self.nb_nodes, n_input=n_features))
        self.layer_list = [Layer(n_neuron=self.nb_nodes, n_input=self.nb_nodes) for i in range(self.n_hidden_layers - 1)]
        self.layer_list.append(Layer(n_neuron=1, n_input=n_features))

        # for e in self.epoch:
        #     for i in range(0, n_rows/self.batch_size, self.batch_size):


        # Loop:
        #      - Feed forward
        #      - Learn
        pass

    def predict(self):
        pass
