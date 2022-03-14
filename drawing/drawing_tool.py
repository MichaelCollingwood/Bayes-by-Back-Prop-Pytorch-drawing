import re

from drawing.drawing_components.neural_network import NeuralNetwork
from drawing.drawing_components.layer import LayerDetails


class DrawNN():
    """."""

    def __init__(
        self, 
        net_params
    ) -> None:
        """Input: (pytorch) net.named_params()."""

        self.params = dict(net_params)

        # Find layer ids
        stringed_keys = "".join(list(self.params.keys()))
        layer_numbers = set(re.findall("\d+", stringed_keys))
        self.layer_ids = sorted([int(num) for num in layer_numbers])

    def draw(
        self, 
        label
    ) -> None:
        """."""

        layer_widths = []

        init_mus_name = f"layer{self.layer_ids[0]}.weight_mus"
        init_layer_width = self.params[init_mus_name].shape[0]
        layer_widths.append(init_layer_width)

        for id in self.layer_ids:
            layer_mus_name = f"layer{id}.weight_mus"
            layer_width = self.params[layer_mus_name].shape[1]
            layer_widths.append(layer_width)
            
            # Just checking if can make specific weight edges untrainable
            print("hello ", self.params[layer_mus_name])
            print(self.params[layer_mus_name][0,0])

        network = NeuralNetwork(
            input_neurons=layer_widths[0],  # number of neurons in input layer
            max_neurons=max(layer_widths)   # max number of neurons in a layer
        )

        for id in self.layer_ids:
            layer_details = LayerDetails()
            layer_details.set_params(id, self.params)
            
            network.add_layer(layer_details)

        network.draw(label)
