class ILayer:
    """Abstract class that represents network layer"""
    def __init__(self, size):
        assert isinstance(size, tuple), "size must be tuple"
        self.size = size

    def forward(self, x):
        """Does forward pass through layer should return numpy array"""
        raise NotImplemented

