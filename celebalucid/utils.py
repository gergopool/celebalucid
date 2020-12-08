import pkgutil

def load_layer_info():
    text = pkgutil.get_data(__name__, 'res/layer_info.txt').decode('utf-8')
    data = []
    lines = text.split('\n')
    for line in lines:
        layer_name, n_channels = line.split(' ')
        n_channels = int(n_channels)
        data.append([layer_name, n_channels])
    return data