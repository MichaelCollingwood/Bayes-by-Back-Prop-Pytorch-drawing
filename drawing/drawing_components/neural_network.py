from matplotlib import pyplot
import logging

from drawing.drawing_components.layer import Layer, LayerDetails, LayerType


class NeuralNetwork():
    """Neural Network drawing class."""

    def __init__(
        self, 
        input_neurons: int, 
        max_neurons: int
    ) -> None:
        """Initialises Neural Network drawing class."""
        
        self.logger = logging.getLogger("NeuralNetwork")
        self.logger.debug("Initialise Neural Network drawing class.")

        self.max_neurons = max_neurons

        self.layers = []
        
        self.logger.debug("Append initial layer.")
        self.layers.append(
            Layer(
                self,
                LayerDetails(shape=(None, input_neurons)),
                self.max_neurons
            )
        )

    def add_layer(
        self, 
        layer_details: LayerDetails
    ) -> None:
        """Adds Layer to Neural Network."""
        
        self.logger.debug("Append Layer to <NeuralNetwork>.layers.")

        self.layers.append(
            Layer(
                self, 
                layer_details,
                self.max_neurons
            )
        )

    def draw(
        self, 
        label: str
    ) -> None:
        """Draws Neural Network."""
        
        self.logger.debug("Draw Neural Network.")

        for i, layer in enumerate(self.layers):
            
            if i == 0:
                layer_type = LayerType.INPUT_LAYER
            elif i == len(self.layers)-1:
                layer_type = LayerType.OUTPUT_LAYER
            else:
                layer_type = LayerType.HIDDEN_LAYER
            
            layer.draw(layer_type)

        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title(label, fontsize=15)
        pyplot.show()