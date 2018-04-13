from classicLayer import Classic

def layerCreator(description):
    size, layer, activation = description
    if layer == "classic":
        return Classic(size, activation)
    raise NotImplemented # Layer type was not yet implemented
