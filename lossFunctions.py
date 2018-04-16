def mse(y, prediction):
    diff = y - prediction
    return (diff ** 2).sum() / len(prediction)
