from ClasicLayer import Classic

class Layer:
    """Abstract class that represents network layer"""
    def __init__(self, size):
        assert isinstance(size, tuple), "size must be tuple"
        self.size = size

    def forward(self, x):
        """Does forward pass through layer should return numpy array"""
        raise NotImplemented

def layerCreator(description):
    size, layer, activation = description
    if layer == "classic":
        return Classic(size, activation)
    raise NotImplemented # Layer type was not yet implemented
