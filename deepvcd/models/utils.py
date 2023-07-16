def get_layer_weights(weights_h5_file=None, layer_name=None):
    if weights_h5_file is None or layer_name is None:
        return None
    else:
        g = weights_h5_file[layer_name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]
        return weight_names, weight_values


def get_layer_index(model, layer_name):
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            return idx