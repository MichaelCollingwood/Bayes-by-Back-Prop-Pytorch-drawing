from __future__ import annotations
from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np
from typing import Optional
import logging
from enum import Enum

from drawing.drawing_components.neuron import Neuron


class LayerType(Enum):
    INPUT_LAYER = 0
    HIDDEN_LAYER = 1
    OUTPUT_LAYER = 2


class LayerDetails():
    """."""

    def __init__(
        self, 
        shape: tuple = (None, None)
    ) -> None:
        """."""

        self.shape: tuple = shape
        
    def set_params(
        self,
        layer_id: int,
        params: dict
    ) -> None:
        """."""
        
        layer_name = f"layer{layer_id}"
        
        self.weight_mus = params[f"{layer_name}.weight_mus"]
        self.weight_rhos = params[f"{layer_name}.weight_rhos"]
        self.bias_mus = params[f"{layer_name}.bias_mus"]
        self.bias_rhos = params[f"{layer_name}.bias_rhos"]
        self.shape = params[f"{layer_name}.weight_mus"].shape


class Layer():
    """Layer drawing class."""

    def __init__(
        self, 
        network, 
        layer_details: LayerDetails, 
        max_neurons: int
    ) -> None:
        """Initialises Layer drawing class."""

        self.layer_details = layer_details
        self.im_dims = {
            "layer_spacing": 100,
            "neuron_spacing": 30,
            "neuron_radius": 1,
            "max_neurons": max_neurons
        }

        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()

        self.neurons = self.__intialise_neurons(layer_details.shape[1])

    def __intialise_neurons(
        self, 
        n_neurons: int
    ) -> list:
        """Initialises Layer neurons."""

        x = self.__calculate_left_margin(n_neurons)

        neurons = []
        for _ in range(n_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.im_dims["neuron_spacing"]

        return neurons

    def __calculate_left_margin(
        self, 
        n_neurons: int
    ) -> float:
        """Calculates left margin so layer is centered."""

        neurons_less_than_max = self.im_dims["max_neurons"] - n_neurons

        return self.im_dims["neuron_spacing"] * neurons_less_than_max / 2

    def __calculate_layer_y_position(self) -> None:
        """Calculates layer y position."""

        if self.previous_layer:
            return self.previous_layer.y + self.im_dims["layer_spacing"]
        else:
            return 0

    def __get_previous_layer(
        self, 
        network
    ) -> Optional[Layer]:
        """Retrieve previous layer."""

        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(
        self, 
        neuron1: Neuron, 
        neuron2: Neuron, 
        sigma: float
    ) -> None:
        """."""

        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.im_dims["neuron_radius"] * sin(angle)
        y_adjustment = self.im_dims["neuron_radius"] * cos(angle)
        line = pyplot.Line2D(
            xdata = (neuron1.x - x_adjustment, neuron2.x + x_adjustment),
            ydata = (neuron1.y - y_adjustment, neuron2.y + y_adjustment),
            color = 'k',
            dashes = (sigma, sigma)
        )

        pyplot.gca().add_line(line)

    def draw(
        self, 
        layer_type: LayerType
    ) -> None:
        """."""
        
        # HAVE TO FIND A NEW WAY TO LINK NEURONS

        for i, neuron in enumerate(self.neurons):
            neuron.draw(self.im_dims["neuron_radius"])

            if self.previous_layer:
                for j, prev_neuron in enumerate(self.previous_layer.neurons):
                    w_rho = self.layer_details.weight_rhos[j, i].detach().numpy()

                    self.__line_between_two_neurons(
                        neuron,
                        prev_neuron,
                        np.log(1 + np.exp(w_rho))
                    )
        
        # Plot layer names
        x_text = self.im_dims["max_neurons"] * self.im_dims["neuron_spacing"]
        pyplot.text(x_text, self.y, str(layer_type), fontsize=12)
