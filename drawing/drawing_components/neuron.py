from matplotlib import pyplot
import logging


class Neuron():
    """."""

    def __init__(self, x, y):
        """."""

        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        """."""

        circle = pyplot.Circle(
            (self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)