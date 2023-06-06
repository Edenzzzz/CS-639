"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from ps2_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU



def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        total_data_num, channels, data_height, data_width = x.shape
        total_kernal, _, kernel_height, kernel_width = w.shape
        stride = conv_param['stride']
        pad = conv_param['pad']
        # Note: here we use // to achieve the "floor" function
        conved_height = 1 + (data_height + 2 * pad - kernel_height) // stride
        conved_width = 1 + (data_width + 2 * pad - kernel_width) // stride
        out = torch.zeros(total_data_num, total_kernal, conved_height, conved_width).to(x)

        # pad the last two dimensions in input x 
        padded_version = torch.nn.functional.pad(x, (pad, pad, pad, pad), 'constant', 0)

        # Convolution
        # Note: Although for-loops are computational consuming, they express the
        # convolution operations as clear as poosible !!
        for data in range(total_data_num):
          image = padded_version[data]
          for kernel in range(total_kernal):
            conv_weight = w[kernel]

            # process rows
            start_idx_row = 0
            end_idx_row = start_idx_row + kernel_height - 1
            for row in range(conved_height):
              # process cols
              start_idx_col = 0
              end_idx_col = start_idx_col + kernel_width - 1
              for col in range(conved_width):
                target_pixels = image[ :, start_idx_row:end_idx_row + 1, start_idx_col:end_idx_col + 1]
                out[data, kernel, row, col] = (target_pixels * conv_weight).sum() + b[kernel]
                start_idx_col += stride
                end_idx_col += stride
              start_idx_row += stride
              end_idx_row += stride
        
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        total_data, total_kernal, conved_height, conved_width = dout.shape
        _, channels, kernal_height, kernal_width = cache[1].shape
        _, _, img_height, img_width = cache[0].shape
        dw = torch.zeros_like(cache[1])
        dx = torch.zeros_like(cache[0])
        db = torch.zeros_like(cache[2])
        pad = cache[3]['pad']
        stride = cache[3]['stride']
        padded_version = torch.nn.functional.pad(cache[0], (pad, pad, pad, pad), 'constant', 0)
        padded_dx = torch.nn.functional.pad(dx, (pad, pad, pad, pad), 'constant', 0)
        # Note: since we wanna make it clear as possible, we need to use a bunch
        # of for-loops which are very computational-consuming!!!
        for data in range(total_data):
          org_image = padded_version[data] # channel * img_height * img_width
          for kernal in range(total_kernal): # total F kernals
            dout_this_kernel = dout[data, kernal] # conved_height * conved_width
            corresponding_org_img_pixels = torch.zeros_like(dout_this_kernel)
            dw_this_kernal = dw[kernal] # channel * kernel_height * kernel_width
            weights_this_kernel = cache[1][kernal]
          
            # process each channal 
            for cha in range(channels):
              org_image_this_channel = org_image[cha]
              padded_dx_this_channel = padded_dx[data, cha]
              dw_this_kernal_this_channel = dw_this_kernal[cha]
              weights_this_kernel_this_channel = weights_this_kernel[cha]
              
              for kernal_rol in range(kernal_height):
                for kernal_col in range(kernal_width):
                  start_rol = kernal_rol
                  for row in range(conved_height):
                    start_col = kernal_col
                    for col in range(conved_width):
                      corresponding_org_img_pixels[row, col] = org_image_this_channel[start_rol, start_col]
                      padded_dx_this_channel[start_rol, start_col] += (dout_this_kernel[row, col] * weights_this_kernel_this_channel[kernal_rol, kernal_col])
                      start_col += stride
                    start_rol += stride
                  dw_this_kernal_this_channel[kernal_rol, kernal_col] += (corresponding_org_img_pixels * dout_this_kernel).sum()
            
            # then we update the db
            db_this_kernal = db[kernal]
            for row in range(conved_height):
              for col in range(conved_width):
                db_this_kernal += dout_this_kernel[row, col]
            
            # finally, get rid of the zero-padding of the padded_dx 
            _, _, padded_dx_height, padded_dx_width = padded_dx.shape 
            dx = padded_dx[:, :, pad : padded_dx_height - pad, pad : padded_dx_width - pad]

        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        total_data, channels, height, width = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        pooled_height = 1 + (height - pool_height) // stride
        pooled_width = 1 + (width - pool_width) // stride
        out = torch.zeros(total_data, channels, pooled_height, pooled_width).to(x)

        for data in range(total_data):
          org_img = x[data]
          for cha in range(channels):
            cur_cha = org_img[cha]
            start_idx_row = 0
            end_idx_row = start_idx_row + pool_height - 1 
            for row in range(pooled_height):
              start_idx_col = 0
              end_idx_col = start_idx_col + pool_width - 1
              for col in range(pooled_width):
                out[data, cha, row, col] = cur_cha[start_idx_row:end_idx_row + 1, start_idx_col:end_idx_col + 1].max()
                start_idx_col += stride
                end_idx_col += stride
              start_idx_row += stride
              end_idx_row += stride

        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        x = cache[0]
        pool_param = cache[1]
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        total_data, channels, pooled_height, pooled_width = dout.shape
        _, _, height, width = x.shape
        dx = torch.zeros_like(x)

        for data in range(total_data):
          img = x[data]
          for cha in range(channels):
            cur_cha = img[cha]
            start_idx_row = 0
            end_idx_row = start_idx_row + pool_height - 1
            for row in range(pooled_height):
              start_idx_col = 0
              end_idx_col = start_idx_col + pool_width - 1
              for col in range(pooled_width):
                target_area = cur_cha[start_idx_row:end_idx_row + 1, start_idx_col:end_idx_col + 1]
                # pickout the max item in the target area (pool_height * pool_width)
                max_idx_row, max_idx_col = 0, 0
                max_val = target_area[0, 0]

                for h in range(pool_height):
                  for w in range(pool_width):
                    if (target_area[h, w] > max_val):
                      max_val = target_area[h, w]
                      max_idx_row, max_idx_col = h, w
                      
                dx[data, cha, start_idx_row:end_idx_row + 1, start_idx_col:end_idx_col + 1][max_idx_row, max_idx_col] += dout[data, cha, row, col]
                start_idx_col += stride
                end_idx_col += stride
              start_idx_row += stride
              end_idx_row += stride
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in thedictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        channel, input_height, input_width = input_dims

        # this is the flatten vector after being max-pooled
        after_pool_dim = num_filters * (input_height // 2) * (input_width // 2)
        # Conv Layer 
        self.params['W1'] = weight_scale * torch.randn((num_filters, channel, filter_size, filter_size), dtype = self.dtype).to(device)
        self.params['b1'] = torch.zeros(num_filters, dtype = self.dtype).to(device)
        # 100 neurons hidden layer
        self.params['W2'] = weight_scale * torch.randn((after_pool_dim, hidden_dim), dtype = self.dtype).to(device)
        self.params['b2'] = torch.zeros(hidden_dim, dtype = self.dtype).to(device)
        # 10 classes output (before softmax classification)
        self.params['W3'] = weight_scale * torch.randn((hidden_dim, num_classes), dtype = self.dtype).to(device)
        self.params['b3'] = torch.zeros(num_classes, dtype = self.dtype).to(device)

        # Another Reasonable Solution: 
        # Since we're using ReLU, we can initialize the weight by multiplying 
        # a factor torch.sqrt(2 / neuron_last_layer)

        # and if we're using tanh, the factor becames to 
        # torch.sqrt(neuron_last_layer)


        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above                                                              #
        ######################################################################
        
        # we use the sandwich layers
        conv_out, conv_cache = Conv_ReLU_Pool.forward(X, W1, b1, conv_param=conv_param, pool_param=pool_param)
        
        # The flattening operation is included in the Linear.forward()
        # then we flatten the conv_out 
        # flatten_vec = conv_out.reshape(-1, 1)

        # then the linear 
        hidden_linear_out, hidden_linear_cache = Linear.forward(conv_out, W2, b2)

        # then the ReLu
        relued_out, relued_cache = ReLU.forward(hidden_linear_out)

        # then the output linear layer
        output_linear_out, output_linear_cache = Linear.forward(relued_out, W3, b3)
        
        scores = output_linear_out
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        # calculate the loss PLUS the weight decay
        loss, grad_x = softmax_loss(output_linear_out, y) 
        loss += (self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2) + torch.sum(W3 * W3)))

        grad_x, grads['W3'], grads['b3'] = Linear.backward(grad_x, output_linear_cache)
        grads['W3'] = (grads['W3'] + 2 * self.reg * W3)
        
        grad_x = ReLU.backward(grad_x, relued_cache)
        grad_x, grads['W2'], grads['b2'] = Linear.backward(grad_x, hidden_linear_cache)
        grads['W2'] = (grads['W2'] + 2 * self.reg * W2)
        
        grad_x, grads['W1'], grads['b1'] = Conv_ReLU_Pool.backward(grad_x, conv_cache)
        grads['W1'] = (grads['W1'] + 2 * self.reg * W1)
        
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights, #
        # biases, and batchnorm scale and shift parameters should be        #
        # stored in the dictionary self.params.                             #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        # Batchnorm scale (gamma) and shift (beta) parameters should be     #
        # initilized to ones and zeros respectively.                        #
        #####################################################################
        # Replace "pass" statement with your code
        in_channels, H, W = input_dims
        K = 3
        for idx, filters in enumerate(num_filters):
            if weight_scale == "kaiming":
                self.params["W" + str(idx + 1)] = kaiming_initializer(Din=in_channels, Dout=filters, K=K, device=device, dtype=dtype)
            else:
                self.params["W" + str(idx + 1) ] = torch.randn(filters, in_channels, K, K, dtype=dtype, device=device) * weight_scale
            self.params["b" + str(idx + 1)] = torch.zeros(filters, dtype=dtype, device=device)
            if self.batchnorm:
                self.params["bn_" + str(idx + 1) + "_gamma"] = torch.full((filters,), 1, dtype=dtype, device=device)
                self.params["bn_" + str(idx + 1) + "_beta"] = torch.full((filters,), 0, dtype=dtype, device=device)
            #update input shape for next conv layer
            in_channels = filters
            if idx in self.max_pools:
                H = H // 2
                W = W // 2

        self.params["fc"] = kaiming_initializer(Din=in_channels * H * W, Dout=num_classes, device=device, dtype=dtype) if weight_scale == "kaiming" \
            else torch.randn(in_channels * H * W, num_classes, dtype=dtype, device=device) * weight_scale
        self.params["fc_b"] = torch.zeros(num_classes, dtype=dtype, device=device)

        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        # Replace "pass" statement with your code
        caches = []
        for idx in range(self.num_layers - 1):
            if idx == 0:
                out = X
                
            if idx not in self.max_pools and self.batchnorm == False:
                out, cache = Conv_ReLU.forward(out, self.params["W" + str(idx + 1)], self.params["b" + str(idx + 1)], conv_param)
            elif idx not in self.max_pools and self.batchnorm == True:
                out, cache = Conv_BatchNorm_ReLU.forward(out, self.params["W" + str(idx + 1)], self.params["b" + str(idx + 1)],
                                                self.params["bn_" + str(idx + 1) + "_gamma"], self.params["bn_" + str(idx + 1) + "_beta"], conv_param, self.bn_params[idx])
            elif idx in self.max_pools and self.batchnorm == False:
                out, cache = Conv_ReLU_Pool.forward(out, self.params["W" + str(idx + 1)], self.params["b" + str(idx + 1)], conv_param, pool_param)
            elif idx in self.max_pools and self.batchnorm == True:
                out, cache = Conv_BatchNorm_ReLU_Pool.forward(out, self.params["W" + str(idx + 1)], self.params["b" + str(idx + 1)],
                                                    self.params["bn_" + str(idx + 1) + "_gamma"], self.params["bn_" + str(idx + 1) + "_beta"], conv_param, self.bn_params[idx], pool_param)
            else:
                raise NotImplementedError("Invalid ConvNet configuration")
            
            caches.append(cache)

        logits, cache_fc = Linear.forward(out, self.params["fc"], self.params["fc_b"])
        scores = logits
        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################
        # Replace "pass" statement with your code
        loss, dout = softmax_loss(scores, y)
        dout, grads["fc"], grads["fc_b"] = Linear.backward(dout, cache_fc)

        #l2 regularization
        loss += 0.5 * self.reg * torch.sum(self.params["fc"] ** 2)
        grads["fc"] += self.reg * self.params["fc"]
        
        for idx in range(self.num_layers - 2, -1, -1):
            if idx not in self.max_pools and self.batchnorm == False:
                #weight indices start from 1
                dout, grads["W" + str(idx + 1)], grads["b" + str(idx + 1)] = Conv_ReLU.backward(dout, caches[idx])

            elif idx not in self.max_pools and self.batchnorm == True:
                dout, grads["W" + str(idx + 1)], grads["b" + str(idx + 1)], grads["bn_" + str(idx + 1) + "_gamma"], grads["bn_" + str(idx + 1) + "_beta"] = \
                    Conv_BatchNorm_ReLU.backward(dout, caches[idx])
                
            elif idx in self.max_pools and self.batchnorm == False:
                dout, grads["W" + str(idx + 1)], grads["b" + str(idx + 1)] = Conv_ReLU_Pool.backward(dout, caches[idx])

            elif idx in self.max_pools and self.batchnorm == True:
                dout, grads["W" + str(idx + 1)], grads["b" + str(idx + 1)], grads["bn_" + str(idx + 1) + "_gamma"], grads["bn_" + str(idx + 1) + "_beta"] = \
                    Conv_BatchNorm_ReLU_Pool.backward(dout, caches[idx]) 
            else:
                raise NotImplementedError("Invalid ConvNet configuration")
        
        for name in grads.keys():
            #l2 regularization for conv and linear weights
            if name[0] == "W":
                loss += 0.5 * self.reg * torch.sum(self.params[name] ** 2)
                grads[name] += self.reg * self.params[name]
        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    weight_scale = 8e-2
    learning_rate = 1.8e-3
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # TODO: Train the best DeepConvNet that you can on      #
    # CIFAR-10 within 60 seconds.                           #
    #########################################################
    # Replace "pass" statement with your code

    #VGG-19-like model
    from fully_connected_networks import sgd_momentum
    model = DeepConvNet(weight_scale="kaiming", reg=1e-4, 
                        num_filters= [32] * 2 + [64] * 2 + [128] * 2,
                        max_pools=[1, 3], batchnorm=False, dtype=dtype, device=device,)
    solver = Solver(model, data_dict, batch_size=128, update_rule=adam, num_train_samples=40000, device=device,
                    optim_config={"learning_rate": 1e-3}, num_epochs=20)

    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code

        weight = torch.randn(Din, Dout, device=device, dtype=dtype) * (gain / Din) ** 0.5

        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code

        weight = torch.randn(Dout, Din, K, K, device=device, dtype=dtype) * (gain / (Din * K * K)) ** 0.5

        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" statement with your code
            
            sample_mean = torch.mean(x, axis=0)
            sample_var = torch.var(x, axis=0, correction=0)
            x_hat = (x - sample_mean) / torch.sqrt(sample_var + eps)
            
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
            out = gamma * x_hat + beta  
            cache = (x, sample_mean, sample_var, gamma, eps, x_hat)
            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################
            # Replace "pass" statement with your code

            out = (x - running_mean) / torch.sqrt(running_var + eps)
            out = gamma * out + beta

            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        # Replace "pass" statement with your code
        
        x, sample_mean, sample_var, gamma, eps, x_hat = cache
        dgamma = torch.sum(dout * x_hat, axis=0)
        dbeta = torch.sum(dout, axis=0)
        dx_hat = dout * gamma

        dvar = torch.sum(dx_hat * (x - sample_mean) * (-0.5) * torch.pow(sample_var + eps, -1.5), axis=0)
        dmean = torch.sum(dx_hat * (-1) / torch.sqrt(sample_var + eps), axis=0) 
        dx = dx_hat / torch.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean) / x.shape[0] + dmean / x.shape[0]

        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta



class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        # Replace "pass" statement with your code

        out = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        cache = []
        for i in range(x.shape[1]):
            #(N, C, H, W) -> (N, H, W, C) -> (N*H*W, C) -> (1, N*H*W)
            out_temp, cache_temp = BatchNorm.forward(x.permute(0, 2, 3, 1).flatten(end_dim=-2)[:, i].unsqueeze(1), gamma[i], beta[i], bn_param)
            cache.append(cache_temp)
            out[:, i, :, :] = out_temp.T.reshape(x.shape[0], x.shape[2], x.shape[3])
        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        # Replace "pass" statement with your code

        dx = torch.zeros(dout.shape, device=dout.device, dtype=dout.dtype)
        dgamma = torch.zeros(dout.shape[1], device=dout.device, dtype=dout.dtype)
        dbeta = torch.zeros(dout.shape[1], device=dout.device, dtype=dout.dtype)
        for i in range(len(cache)):
            dx_temp, dgamma[i], dbeta[i] = BatchNorm.backward(dout[:, i, :, :].flatten().unsqueeze(1), cache[i])
            dx[:, i, :, :] = dx_temp.T.reshape(dout.shape[0], dout.shape[2], dout.shape[3])
            
        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
