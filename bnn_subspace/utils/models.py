"""
Creates ResNet with filter response norm

Resnet code taken from https://github.com/izmailovpavel/neurips_bdl_starter_kit/blob/main/pytorch_models.py
"""

from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
import utils


def get_curve_model(problem_setting,curve_checkpoint_path,**kwargs):
    """Construct curve subspace model based on problem setting"""

    base_net = utils.models.get_model(problem_setting=problem_setting,**kwargs)

    curve_checkpoint = utils.save.load_net_and_opt_checkpoint(curve_checkpoint_path)

    net = utils.models.construct_sampling_curve_from_checkpoint(
        curve_checkpoint=curve_checkpoint, base_net=base_net)

    return net




class FilterResponseNorm_layer(nn.Module):
    def __init__(self, num_filters, eps=1e-6):
        super(FilterResponseNorm_layer, self).__init__()
        self.eps = eps
        par_shape = (1, num_filters, 1, 1)  # [1,C,1,1]
        self.tau = torch.nn.Parameter(
            torch.zeros(par_shape)
        )  #self.register_parameter("tau", torch.zeros(par_shape))
        self.beta = torch.nn.Parameter(
            torch.zeros(par_shape)
        )  #self.register_parameter("beta", torch.zeros(par_shape))
        self.gamma = torch.nn.Parameter(
            torch.ones(par_shape)
        )  #self.register_parameter("gamma", torch.ones(par_shape))

    def forward(self, x):

        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        x = x * 1 / torch.sqrt(nu2 + self.eps)
        y = self.gamma * x + self.beta
        z = torch.max(y, self.tau)
        return z


class resnet_block(nn.Module):
    def __init__(self,
                 normalization_layer,
                 input_size,
                 num_filters,
                 kernel_size=3,
                 strides=1,
                 activation=torch.nn.Identity,
                 use_bias=True):
        super(resnet_block, self).__init__()
        # input size = C, H, W
        p0 = int(strides * (input_size[2] - 1) + kernel_size - input_size[2])
        if p0 % 2 == 0:
            p0 /= 2
            p2 = p0
        else:
            p2 = (p0 + 1) / 2
            p0 = (p0 - 1) / 2
        # height padding
        p1 = strides * (input_size[1] - 1) + kernel_size - input_size[1]
        if p1 % 2 == 0:
            p1 /= 2
            p3 = p1
        else:
            p3 = (p1 + 1) / 2
            p1 = (p1 - 1) / 2
        self.pad1 = torch.nn.ZeroPad2d((int(p0), int(p1), int(p2), int(p3)))
        self.conv1 = torch.nn.Conv2d(input_size[0],
                                     num_filters,
                                     kernel_size=kernel_size,
                                     stride=strides,
                                     padding=0,
                                     bias=use_bias)
        self.norm1 = normalization_layer(num_filters)
        self.activation1 = activation()

    def forward(self, x):

        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation1(out)

        return out


class stacked_resnet_block(nn.Module):
    def __init__(self, normalization_layer, num_filters, input_num_filters,
                 stack, res_block, activation, use_bias):
        super(stacked_resnet_block, self).__init__()
        self.stack = stack
        self.res_block = res_block
        strides = 1
        if stack > 0 and res_block == 0:  # first layer but not first stack
            strides = 2  # downsample
        self.res1 = resnet_block(normalization_layer=normalization_layer,
                                 num_filters=num_filters,
                                 input_size=(input_num_filters, 32, 32),
                                 strides=strides,
                                 activation=activation,
                                 use_bias=use_bias)
        self.res2 = resnet_block(normalization_layer=normalization_layer,
                                 num_filters=num_filters,
                                 input_size=(num_filters, 32, 32),
                                 use_bias=use_bias)
        if stack > 0 and res_block == 0:  # first layer but not first stack
            # linear projection residual shortcut connection to match changed dims
            self.res3 = resnet_block(normalization_layer=normalization_layer,
                                     num_filters=num_filters,
                                     input_size=(input_num_filters, 32, 32),
                                     strides=strides,
                                     kernel_size=1,
                                     use_bias=use_bias)

        self.activation1 = activation()

    def forward(self, x):
        y = self.res1(x)
        y = self.res2(y)
        if self.stack > 0 and self.res_block == 0:
            x = self.res3(x)

        out = self.activation1(x + y)
        return out


class make_resnet_fn(nn.Module):
    def __init__(self,
                 num_classes,
                 depth,
                 normalization_layer,
                 width=16,
                 use_bias=True,
                 activation=torch.nn.ReLU(inplace=True)):
        super(make_resnet_fn, self).__init__()
        self.num_res_blocks = (depth - 2) // 6
        self.normalization_layer = normalization_layer
        self.activation = activation
        self.use_bias = use_bias
        self.width = width
        if (depth - 2) % 6 != 0:
            raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

        # first res_layer
        self.layer1 = resnet_block(normalization_layer=normalization_layer,
                                   num_filters=width,
                                   input_size=(3, 32, 32),
                                   kernel_size=3,
                                   strides=1,
                                   activation=torch.nn.Identity,
                                   use_bias=True)
        # stacks
        self.stacks = self._make_res_block()
        # avg pooling
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=(8, 8),
                                           stride=8,
                                           padding=0)
        # linear layer
        self.linear1 = nn.Linear(768, num_classes)

    def forward(self, x):
        # first res_layer
        out = self.layer1(x)  # shape out torch.Size([5, 16, 32, 32])
        out = self.stacks(out)
        out = self.avgpool1(out)
        out = torch.flatten(out, start_dim=1)
        logits = self.linear1(out)
        return logits

    def _make_res_block(self):
        layers = []
        num_filters = self.width
        input_num_filters = num_filters
        for stack in range(3):
            for res_block in range(self.num_res_blocks):
                layers.append(
                    stacked_resnet_block(self.normalization_layer, num_filters,
                                         input_num_filters, stack, res_block,
                                         self.activation, self.use_bias))
                input_num_filters = num_filters
            num_filters *= 2
        return nn.Sequential(*layers)


def make_resnet20_frn_fn(data_info, activation=torch.nn.ReLU):
    num_classes = data_info["num_classes"]
    return make_resnet_fn(num_classes,
                          depth=20,
                          normalization_layer=FilterResponseNorm_layer,
                          activation=activation)

class CNN1d(nn.Module):
    """
    CNN-LSTM for sentiment classification 
    taken from https://github.com/bentrevett/pytorch-sentiment-analysis
    """

    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embedding_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        embedded = self.embedding(text)
        
        embedded = embedded.permute(0, 2, 1)
        
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
            
        return self.fc(cat)

# pytorch version
def get_model(problem_setting,**kwargs):

    if problem_setting=='cifar10':
        return _get_cifar10_model(model_name='resnet20_frn_swish',data_info={"num_classes":10})
    elif problem_setting=='imdb':
        return _get_imdb_model(TEXT=kwargs['TEXT'])

    else:
        raise NotImplementedError("Problem setting {} not implemented".format(problem_setting))

def _get_imdb_model(TEXT):
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    EMBEDDING_DIM = 100
    VOCAB_SIZE=len(TEXT.vocab)

    model = CNN1d(
                vocab_size=VOCAB_SIZE,
                embedding_dim=EMBEDDING_DIM,
                n_filters=100,
                filter_sizes=[3,4,5],
                output_dim=2,
                dropout=0.5,
                pad_idx=PAD_IDX
            )
            
    #copy pretrained embeddings
    model.embedding.weight.data.copy_(TEXT.vocab.vectors) 
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    return model

def _get_cifar10_model(model_name, data_info, **kwargs):
    _MODEL_FNS = {
        "resnet20_frn":
        make_resnet20_frn_fn,
        "resnet20_frn_swish":
        partial(make_resnet20_frn_fn, activation=torch.nn.SiLU),
    }
    net_fn = _MODEL_FNS[model_name](data_info, **kwargs)
    return net_fn

def construct_curve_model(base_net, endpoint_1_dict, endpoint_2_dict):
    #construct a CurveModel given base net and two checkpoint paths
    

    def trim_module_name(name):
        #remove prefix so names align
        if name[:7]=='module.': 
            return name[7:]
        else:
            return name

    #change dict_names since SWA model appends endpoint
    new_endpoint_1_dict = {trim_module_name(key):endpoint_1_dict[key] for key in endpoint_1_dict} 
    new_endpoint_2_dict = {trim_module_name(key):endpoint_2_dict[key] for key in endpoint_2_dict} 


    #construct curve net
    curve_net = CurveModel(net=base_net, dict_1=new_endpoint_1_dict, dict_2=new_endpoint_2_dict)

    return curve_net



class CurveModel(torch.nn.Module):
    #take two existing state dicts as input
    #construct a subspace curve nn as in Garipov et al. DNN Loss Surfaces and Mode Connectivity

    def __init__(self, net, dict_1, dict_2, device = None, *args, **kwargs):
        super(CurveModel,self).__init__(*args, **kwargs)
        self.dict_1 = dict_1
        self.dict_2 = dict_2
        self.device = device 
        self.net = net

        self.midpoint_dict = {key:torch.nn.Parameter(0.5*dict_1[key]+0.5*dict_2[key],requires_grad=True) for key in self.net.state_dict()}

    def parameters(self):
        return self.midpoint_dict.values()

    def cuda(self):
        self.to('cuda')

    def to(self, device):

        self.device=device
        self.net.to(device)

        for key in self.midpoint_dict.keys():
            self.dict_1[key] = self.dict_1[key].to(device)
            self.dict_2[key] = self.dict_2[key].to(device)
            self.midpoint_dict[key] = self.midpoint_dict[key].to(device)


    @torch.no_grad()
    def update_weights(self,t=None):
        #only set t if we are not given a fixed value    
        if t is None:
            if self.training:
                #during training randomly sample t value
                t = torch.rand(1).to(self.device)
            else:
                #during testing eval midpoint 
                #unless otherwise specified
                t = 0.5

        new_state_dict = {}
        for key,param in self.net.state_dict().items():
            #take interpolant t to assign model weights, then do forward prop
            if t < 0.5:
                coeff = t / 0.5
                to_copy = (coeff * self.midpoint_dict[key] +(1.0 - coeff) * self.dict_1[key])

            else:
                coeff = (t - 0.5) / 0.5
                to_copy = (1 - coeff) * self.midpoint_dict[key] + coeff * self.dict_2[key]

            new_state_dict[key]=to_copy

        self.net.load_state_dict(new_state_dict)

    def forward(self, input, t=None):
        '''Perform forward propagation after interpolating between endpoints and midpoint'''
        self.update_weights(t)

        return self.net.forward(input)


#flatten and unflatten taken from https://raw.githubusercontent.com/wjmaddox/drbayes/master/subspace_inference/utils.py
def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i=0
    for tensor in likeTensorList:
        #n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:,i:i+n].view(tensor.shape))
        i+=n
    return outList

class SubspaceModel(torch.nn.Module):
    #take three state dicts as input and constructs curve subspace
    #construct a subspace curve nn as in Garipov et al. DNN Loss Surfaces and Mode Connectivity
    def __init__(self, net, w_hat, P, device = None, *args, **kwargs):
        super(SubspaceModel,self).__init__(*args, **kwargs)

        self.w_hat = w_hat
        self.P = P
        self.device = device 
        self.net = net

    def cuda(self):
        self.to('cuda')

    def to(self, device):

        self.device=device
        self.net.to(device)

        self.w_hat = self.w_hat.to(device)
        self.P = self.P.to(device)

    @torch.no_grad()
    def update_weights(self,vec):
        #set weights to point on subspace
        assert vec.shape[0]==1 and vec.shape[1]==2
        new_weights = vec@self.P+self.w_hat 

        unflattened_weights = unflatten_like(new_weights,[val for val in self.net.state_dict().values()])

        for (key,val),unflattened_weight in zip(self.net.state_dict().items(),unflattened_weights):
            val.mul_(0).add_(unflattened_weight)
   
    @torch.no_grad()
    def get_proj_grad(self):
        return flatten([x.grad for x in self.net.parameters()]).unsqueeze(0)@self.P.T

    def forward(self, input, vec=None):
        '''Perform forward propagation after interpolating between endpoints and midpoint'''
        if vec is not None:
            self.update_weights(vec=vec)

        return self.net.forward(input)


def construct_sampling_curve_from_checkpoint(curve_checkpoint,base_net):
    endpoint_1 = curve_checkpoint['endpoint_1']
    endpoint_2 = curve_checkpoint['endpoint_2']
    midpoint = curve_checkpoint['midpoint']

    weight_keys = base_net.state_dict().keys()

    w_0 = flatten([endpoint_1[key] for key in weight_keys])
    w_1 = flatten([endpoint_2[key] for key in weight_keys])
    w_mid = flatten([midpoint[key] for key in weight_keys])

    #construct basis vectors following https://arxiv.org/pdf/1907.07504.pdf
    w_hat = (w_1+w_0)/2.0
    v1 = (w_0-w_hat)
    v2 = (w_mid-w_hat)
    P = torch.stack([v1,v2])

    net = SubspaceModel(net=base_net,w_hat=w_hat,P=P)

    return net
