from pynq import Xlnk
from pynq import Overlay
import numpy as np
import pynq.lib.dma

overlay = Overlay('Bitstream.bit')
dma = overlay.cnn.axi_dma_0

xlnk = Xlnk()

class Convolution2D:
    def __init__(self, input_planes, output_planes, conv_h, conv_w, stride_h, stride_w, padding, relu, weight, bias=np.any([Flase]), precision=10000):
        self.input_planes = input_planes
        self.output_planes = output_planes
        self.conv_h = conv_h
        self.conv_w = conv_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding = padding
        self.relu = relu
        assert len(weight.shape) == 4, 'Input tensor deminision should be (N, D, X, Y)'
        self.weight = np.multiply(weight.flatten(), precision)
        if bias.any():
            self.bias = np.multiply(bias, precision)
        else:
            self.bias = np.any([False])
        self.precision = precision

    def forward(self, data, dma, load_input=1, load_weight=1):
        assert len(data.shape) == 3, 'Input Tensor deminision should be (X,Y,Z) '
        in_buffer = xlnk.cma_array(shape=(3000,), dtype=np.int32)
        data = np.multiply(data, self.precision)
        x = data.flatten()
        size = data.shape

        configuration = xlnk.cma_array(shape=(17,), dtype=np.int32)
        configuration[0] = 0
        configuration[1] = x.shape[0]
        configuration[2] = size[0]  # image deminsion d
        configuration[3] = size[1]  # image deminsion h
        configuration[4] = size[2]  # image deminsion w
        configuration[5] = self.out_planes  # Number of filters
        configuration[6] = self.input_planes  # first deminsion of weights
        configuration[7] = self.conv_h
        configuration[8] = self.conv_w
        configuration[9] = self.stride_h
        configuration[10] = self.stride_w
        configuration[11] = self.padding
        if self.bias.any():
            configuration[12] = 1
        else:
            configuration[12] = 0
        configuration[13] = self.relu
        configuration[14] = self.precision
        configuration[15] = load_input
        configuration[16] = load_weight
        dma.sendchannel.transfer(configuration)
        dma.sendchannel.wait()
        dma.recvchannel.transfer(configuration)
        dma.recvchannel.wait()

        # upload image inside FPGA
        for idx in range(int((x.shape[0]) / 3000)):
            np.copyto(in_buffer, x[(idx * 3000):(idx * 3000) + 3000], casting='unsafe')
            dma.sendchannel.transfer(in_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer)
            dma.recvchannel.wait()
        if ((x.shape[0]) % 3000 != 0):
            in_buffer_remind = xlnk.cma_array(shape=(((x.shape[0]) % 3000),), dtype=np.int32)
            np.copyto(in_buffer_remind, x[int((x.shape[0]) / 3000) * 3000:x.shape[0]], casting='unsafe')
            dma.sendchannel.transfer(in_buffer_remind)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer_remind)
            dma.recvchannel.wait()

        # upload bias inside FPGA

        if (self.bias.any()):

            for idx in range(int((self.bias.shape[0]) / 3000)):
                np.copyto(in_buffer, self.bias[(idx * 3000):(idx * 3000) + 3000], casting='unsafe')
                dma.sendchannel.transfer(in_buffer)
                dma.sendchannel.wait()
                dma.recvchannel.transfer(in_buffer)
                dma.recvchannel.wait()
            if ((self.bias.shape[0]) % 3000 != 0):
                in_buffer_remind = xlnk.cma_array(shape=(((self.bias.shape[0]) % 3000),), dtype=np.int32)
                np.copyto(in_buffer_remind, self.bias[int((self.bias.shape[0]) / 3000) * 3000:self.bias.shape[0]], casting='unsafe')
                dma.sendchannel.transfer(in_buffer_remind)
                dma.sendchannel.wait()
                dma.recvchannel.transfer(in_buffer_remind)
                dma.recvchannel.wait()

        # upload weights inside FPGA


        for idx in range(int((self.weight.shape[0]) / 3000)):
            np.copyto(in_buffer, self.weight[(idx * 3000):(idx * 3000) + 3000], casting='unsafe')
            dma.sendchannel.transfer(in_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer)
            dma.recvchannel.wait()
        if ((self.weight.shape[0]) % 3000 != 0):
            in_buffer_remind = xlnk.cma_array(shape=(((self.weight.shape[0]) % 3000),), dtype=np.int32)
            np.copyto(in_buffer_remind, self.weight[int((self.weight.shape[0]) / 3000) * 3000:self.weight.shape[0]], casting='unsafe')
            dma.sendchannel.transfer(in_buffer_remind)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer_remind)
            dma.recvchannel.wait()

        # Get final Result size
        in_buffer_remind = xlnk.cma_array(shape=(3,), dtype=np.int32)
        dma.sendchannel.transfer(in_buffer_remind)
        dma.sendchannel.wait()
        dma.recvchannel.transfer(in_buffer_remind)
        dma.recvchannel.wait()

        total_result = in_buffer_remind[0]
        out_h = in_buffer_remind[1]
        out_w = in_buffer_remind[2]
        output = np.zeros((total_result,))
        # Return Final size from FPGA
        for idx in range(int((total_result) / 3000)):
            dma.sendchannel.transfer(in_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer)
            dma.recvchannel.wait()
            np.copyto(output[(idx * 3000):((idx * 3000) + 3000)], in_buffer, casting='unsafe')
        if ((total_result) % 3000 != 0):
            in_buffer_remind = xlnk.cma_array(shape=(((total_result) % 3000),), dtype=np.int32)
            dma.sendchannel.transfer(in_buffer_remind)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer_remind)
            dma.recvchannel.wait()
            np.copyto(output[(int((total_result) / 3000) * 3000):total_result], in_buffer_remind, casting='unsafe')

        # Return result to float format and reshape
        output = np.divide(output, self.precision).reshape(self.out_planes, out_h, out_w)
        return output



class Pool:
    def __init__(self, win_h, win_w, stride_h, stride_w, pool_type, padding, relu, precision=10000):
        self.win_h = win_h
        self.win_w = win_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding = padding
        self.relu = relu
        if (pool_type == 'Max'):
            self.pool_type = 0
        else:
            self.pool_type = 1

        self.precision = precision  # precision of weights and bias in our calculation inside FPGA

    # Parameters In POOLING{ 0-Module selection,1-Input size, 2-Input D, 3-Input H, 4-Input W, 5-Pooling window H,6-
    #  Pooling window W,7- stride H,8- Stride W,9- Pooling Type {0:max , 1: Average},10- padding,11- relu_Activation, 12-precision, 13-load_input }

    def forward(self, data, dma, load_input=1):

        assert len(data.shape) == 3, 'Input Tensor deminision should be (X,Y,Z) '
        in_buffer = xlnk.cma_array(shape=(3000,), dtype=np.int32)
        data = np.multiply(data, self.precision)
        x = data.flatten()
        size = data.shape
        configuration = xlnk.cma_array(shape=(17,), dtype=np.int32)
        configuration[0] = 1
        configuration[1] = x.shape[0]
        configuration[2] = size[0]  # image deminsion d
        configuration[3] = size[1]  # image deminsion h
        configuration[4] = size[2]  # image deminsion w
        configuration[5] = self.win_h
        configuration[6] = self.win_w
        configuration[7] = self.stride_h
        configuration[8] = self.stride_w
        configuration[9] = self.pool_type
        configuration[10] = self.padding
        configuration[11] = self.relu
        configuration[12] = self.precision
        configuration[13] = load_input
        configuration[14] = 0
        configuration[15] = 0
        configuration[16] = 0
        dma.sendchannel.transfer(configuration)
        dma.sendchannel.wait()
        dma.recvchannel.transfer(configuration)
        dma.recvchannel.wait()

        # upload image inside FPGA
        for idx in range(int((x.shape[0]) / 3000)):
            temproray = x[(idx * 3000):(idx * 3000) + 3000]

            np.copyto(in_buffer, temproray, casting='unsafe')
            dma.sendchannel.transfer(in_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer)
            dma.recvchannel.wait()
        if ((x.shape[0]) % 3000 != 0):
            in_buffer_remind = xlnk.cma_array(shape=(((x.shape[0]) % 3000),), dtype=np.int32)
            temproray = x[int((x.shape[0]) / 3000) * 3000:x.shape[0]]
            np.copyto(in_buffer_remind, temproray, casting='unsafe')
            dma.sendchannel.transfer(in_buffer_remind)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer_remind)
            dma.recvchannel.wait()

        # Get final Result size
        in_buffer_remind = xlnk.cma_array(shape=(3,), dtype=np.int32)
        dma.sendchannel.transfer(in_buffer_remind)
        dma.sendchannel.wait()
        dma.recvchannel.transfer(in_buffer_remind)
        dma.recvchannel.wait()

        total_result = in_buffer_remind[0]
        out_h = in_buffer_remind[1]
        out_w = in_buffer_remind[2]
        output = np.zeros((total_result,))

        # Return Final size from FPGA
        for idx in range(int((total_result) / 3000)):
            dma.sendchannel.transfer(in_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer)
            dma.recvchannel.wait()
            np.copyto(output[(idx * 3000):((idx * 3000) + 3000)], in_buffer, casting='unsafe')
        if ((total_result) % 3000 != 0):
            in_buffer_remind = xlnk.cma_array(shape=(((total_result) % 3000),), dtype=np.int32)
            dma.sendchannel.transfer(in_buffer_remind)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer_remind)
            dma.recvchannel.wait()
            np.copyto(output[(int((total_result) / 3000) * 3000):total_result], in_buffer_remind, casting='unsafe')

        # Return result to float format and reshape
        output = np.divide(output, self.precision).reshape(size[0], out_h, out_w)
        return output



class FC:
    # { 0-Module selection,1-Input size, 2-output size 3- relu_Activation, 4-precision, 5-load_input, 6- bias Activation }
    def __init__(self, input_size, output_size, relu, weight, bias=np.any([False]), precision=10000):
        self.input_planes = input_size
        self.output = output_size
        self.relu = relu
        assert len(weight.shape) == 2, 'weight Tensor deminision should be (D,X) '
        assert weight.shape[1] == input_size, 'weight Tensor deminision should be same az Input size '
        assert weight.shape[0] == output_size, 'weight Tensor deminision should be same az Input size '
        self.weight = np.multiply(weight, precision)
        if (bias.any()):
            self.bias = np.multiply(bias, precision)
        else:
            self.bias = np.any([False])
        self.precision = precision  # precision of weights and bias in our calculation inside FPGA

    def forward(self, data, dma, load_input=1):

        in_buffer = xlnk.cma_array(shape=(3000,), dtype=np.int32)
        data = np.multiply(data, self.precision)
        x = data.flatten()
        size = data.shape
        assert x.shape[0] == self.input_planes, 'Input Tensor deminision should be Same Input size '
        configuration = xlnk.cma_array(shape=(17,), dtype=np.int32)
        configuration[0] = 2
        configuration[1] = self.input_planes
        configuration[2] = self.output
        configuration[3] = self.relu
        configuration[4] = self.precision
        configuration[5] = load_input
        if (self.bias.any()):
            configuration[6] = 1
        else:
            configuration[6] = 0
        configuration[7] = 0
        configuration[8] = 0
        configuration[9] = 0
        configuration[10] = 0
        configuration[11] = 0

        configuration[13] = 0
        configuration[14] = 0
        configuration[15] = 0
        configuration[16] = 0
        dma.sendchannel.transfer(configuration)
        dma.sendchannel.wait()
        dma.recvchannel.transfer(configuration)
        dma.recvchannel.wait()

        # upload image inside FPGA
        for idx in range(int((x.shape[0]) / 3000)):
            temproray = x[(idx * 3000):(idx * 3000) + 3000]

            np.copyto(in_buffer, temproray, casting='unsafe')
            dma.sendchannel.transfer(in_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer)
            dma.recvchannel.wait()
            print('image1', in_buffer)
        if ((x.shape[0]) % 3000 != 0):
            in_buffer_remind = xlnk.cma_array(shape=(((x.shape[0]) % 3000),), dtype=np.int32)
            temproray = x[int((x.shape[0]) / 3000) * 3000:x.shape[0]]
            np.copyto(in_buffer_remind, temproray, casting='unsafe')
            dma.sendchannel.transfer(in_buffer_remind)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer_remind)
            dma.recvchannel.wait()

        # upload bias inside FPGA

        if (self.bias.any()):

            for idx in range(int((self.bias.shape[0]) / 3000)):
                temproray = self.bias[(idx * 3000):(idx * 3000) + 3000]
                np.copyto(in_buffer, temproray, casting='unsafe')
                dma.sendchannel.transfer(in_buffer)
                dma.sendchannel.wait()
                dma.recvchannel.transfer(in_buffer)
                dma.recvchannel.wait()
            if ((self.bias.shape[0]) % 3000 != 0):
                in_buffer_remind = xlnk.cma_array(shape=(((self.bias.shape[0]) % 3000),), dtype=np.int32)
                temproray = self.bias[int((self.bias.shape[0]) / 3000) * 3000:self.bias.shape[0]]
                np.copyto(in_buffer_remind, temproray, casting='unsafe')
                dma.sendchannel.transfer(in_buffer_remind)
                dma.sendchannel.wait()
                dma.recvchannel.transfer(in_buffer_remind)
                dma.recvchannel.wait()

        # upload weights inside FPGA
        for i in range(self.output):
            Tmp_w = self.weight[i]
            size = (Tmp_w.shape[0])
            for idx in range(int(size / 3000)):
                temproray = Tmp_w[(idx * 3000):(idx * 3000) + 3000]
                np.copyto(in_buffer, temproray, casting='unsafe')
                dma.sendchannel.transfer(in_buffer)
                dma.sendchannel.wait()
                dma.recvchannel.transfer(in_buffer)
                dma.recvchannel.wait()
            if ((size) % 3000 != 0):
                in_buffer_remind = xlnk.cma_array(shape=(((size) % 3000),), dtype=np.int32)
                temproray = Tmp_w[int((size) / 3000) * 3000:size]
                np.copyto(in_buffer_remind, temproray, casting='unsafe')
                dma.sendchannel.transfer(in_buffer_remind)
                dma.sendchannel.wait()
                dma.recvchannel.transfer(in_buffer_remind)
                dma.recvchannel.wait()
                # Return result to float format and reshape
        total_result = self.output
        output = np.zeros((total_result,))
        for idx in range(int((total_result) / 3000)):
            dma.sendchannel.transfer(in_buffer)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer)
            dma.recvchannel.wait()
            np.copyto(output[(idx * 3000):((idx * 3000) + 3000)], in_buffer, casting='unsafe')
        if ((total_result) % 3000 != 0):
            in_buffer_remind = xlnk.cma_array(shape=(((total_result) % 3000),), dtype=np.int32)
            dma.sendchannel.transfer(in_buffer_remind)
            dma.sendchannel.wait()
            dma.recvchannel.transfer(in_buffer_remind)
            dma.recvchannel.wait()
            np.copyto(output[(int((total_result) / 3000) * 3000):total_result], in_buffer_remind, casting='unsafe')
        output = np.divide(output, self.precision)
        return output
