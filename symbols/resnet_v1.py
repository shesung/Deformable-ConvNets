
import mxnet as mx


conv_idx = 0
bn_idx = 0
relu_idx = 0

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256,
                  memonger=False, use_global_stats=True, use_dilated=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    global conv_idx
    global bn_idx
    global relu_idx
    dilate = (2,2) if use_dilated else ()
    pad_dilate = (2,2) if use_dilated else (1,1)
    if bottle_neck:
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter*0.25), kernel=(1,1), stride=stride, pad=(0,0),
                                   no_bias=True, workspace=workspace, name='%s_conv%d'%(name, conv_idx))
        conv_idx += 1
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, use_global_stats=use_global_stats, eps=2e-5, momentum=bn_mom,
                               name='%s_batchnorm%d'%(name, bn_idx))
        bn_idx += 1
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name='%s_relu%d'%(name, relu_idx))
        relu_idx += 1
        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(3,3), stride=(1,1),
                                   pad=pad_dilate, dilate=dilate, cudnn_off=use_dilated,
                                   no_bias=True, workspace=workspace, name='%s_conv%d'%(name, conv_idx))
        conv_idx += 1
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, use_global_stats=use_global_stats, eps=2e-5, momentum=bn_mom,
                               name='%s_batchnorm%d'%(name, bn_idx))
        bn_idx += 1
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name='%s_relu%d'%(name, relu_idx))
        relu_idx += 1
        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name='%s_conv%d'%(name, conv_idx))
        conv_idx += 1
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, use_global_stats=use_global_stats, eps=2e-5, momentum=bn_mom,
                               name='%s_batchnorm%d'%(name, bn_idx))
        bn_idx += 1

        if dim_match:
            shortcut = data
        else:
            conv1sc = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name='%s_conv%d'%(name, conv_idx))
            conv_idx += 1
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, use_global_stats=use_global_stats, eps=2e-5, momentum=bn_mom,
                                        name='%s_batchnorm%d'%(name, bn_idx))
            bn_idx += 1
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        add = mx.sym.Activation(data=bn3 + shortcut, act_type='relu', name='%s_relu%d'%(name, relu_idx))
        relu_idx += 1
    else:
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name='%s_conv%d'%(name, conv_idx))
        conv_idx += 1
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, use_global_stats=use_global_stats, eps=2e-5,
                               name='%s_batchnorm%d'%(name, bn_idx))
        bn_idx += 1
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name='%s_relu%d'%(name, relu_idx))
        relu_idx += 1
        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=(1,1),
                                   pad=pad_dilate, dilate=dilate, cudnn_off=use_dilated,
                                   no_bias=True, workspace=workspace, name='%s_conv%d'%(name, conv_idx))
        conv_idx += 1
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, use_global_stats=use_global_stats, eps=2e-5,
                               name='%s_batchnorm%d'%(name, bn_idx))
        bn_idx += 1

        if dim_match:
            shortcut = data
        else:
            conv1sc = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name='%s_conv%d'%(name, conv_idx))
            conv_idx += 1
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, use_global_stats=use_global_stats, eps=2e-5,
                                        name='%s_batchnorm%d'%(name, bn_idx))
            bn_idx += 1
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        add = mx.sym.Activation(data=bn2 + shortcut, act_type='relu', name='%s_relu%d'%(name, relu_idx))
        relu_idx += 1
    return add


def get_resnet(data, num_layers, strides=[1,2,2,2]):
    global conv_idx
    global bn_idx
    global unit_idx
    memonger = True
    bn_mom = 0.9
    workspace = 512
    use_dilated_in_stage5 = False

    if num_layers >= 50:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False

    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 50:
        units = [3, 4, 6, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, use_global_stats=True, eps=2e-5, momentum=bn_mom, name='batchnorm0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    output_layers = []
    num_stages = 4
    for i in range(num_stages):
        conv_idx = 0
        bn_idx = 0
        relu_idx = 0
        use_dilated = (i == 3 and use_dilated_in_stage5)
        body = residual_unit(body,
                             num_filter  = filter_list[i+1],
                             stride      = (strides[i], strides[i]),
                             dim_match   = False if i > 0 else True,
                             use_dilated = use_dilated,
                             name        = 'stage%d' % (i + 1),
                             bottle_neck = bottle_neck,
                             workspace   = workspace,
                             memonger    = memonger)
        for j in range(units[i]-1):
            body = residual_unit(body,
                                 num_filter  = filter_list[i+1],
                                 stride      = (1,1),
                                 dim_match   = True,
                                 use_dilated = use_dilated,
                                 name        = 'stage%d' % (i + 1),
                                 bottle_neck = bottle_neck,
                                 workspace   = workspace,
                                 memonger    = memonger)
        output_layers.append(body)
    return output_layers

