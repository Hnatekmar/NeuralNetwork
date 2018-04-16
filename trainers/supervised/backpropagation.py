import lossFunctions 

def backpropagation(model, x, y):
    result = model.forward(x)
    error = lossFunctions.mse(y, result)

