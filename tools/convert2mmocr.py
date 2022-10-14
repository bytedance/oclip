import torch


def convert_param_name(model_path, save_path):
    state_dict = torch.load(model_path)

    backbone_dict = {}
    for k, v in state_dict['state_dict'].items():
        if k.startswith('module.visual.') and 'attnpool' not in k:
            new_k = k[len('module.visual.'):]
            # update parameter name in stem layers
            if new_k.startswith('conv1'):
                new_k =  new_k.replace('conv1', 'stem.0')
            elif new_k.startswith('bn1'):
                new_k =  new_k.replace('bn1', 'stem.1')
            elif new_k.startswith('conv2'):
                new_k =  new_k.replace('conv2', 'stem.2')
            elif new_k.startswith('bn2'):
                new_k =  new_k.replace('bn2', 'stem.3')
            elif new_k.startswith('conv3'):
                new_k =  new_k.replace('conv3', 'stem.4')
            elif new_k.startswith('bn3'):
                new_k =  new_k.replace('bn3', 'stem.5')
            
            # update parameter name in bottleneck blocks
            new_k =  new_k.replace('downsample.1', 'downsample.2').replace('downsample.0', 'downsample.1')

            print(new_k, v.shape)
            backbone_dict[new_k] = v

    torch.save(backbone_dict, save_path)
    return


if __name__ == '__main__':
    model_path = 'path/to/model.pt'
    save_path = 'path/to/save/model.pth'
    convert_param_name(model_path, save_path)
