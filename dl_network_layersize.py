def dl_network_layersize(net, X):

    '''
    show the size of each layer in the network

    net: a neural network
    X: a randomly generated tensor that fits the first layer of the network

    code from https://space.bilibili.com/1567748478
    '''

    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)