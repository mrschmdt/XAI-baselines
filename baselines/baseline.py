from network import NeuralNetwork

class Baseline():
    def __init__(self, classification_model: NeuralNetwork):
        self.classification_model: NeuralNetwork = classification_model
        self.num_inputs = self.classification_model.get_number_of_input_features()
    def get_baseline(self):
        pass