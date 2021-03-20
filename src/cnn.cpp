#include "ap_axi_sdata.h"
#include "ap_int.h"

// AXI4-Stream interfaces with side-channel
typedef ap_axiu<32, 1, 1, 1> axi_type;
typedef hls::stream<axi_type> AXI_VAL;

int axi_transfer(AXI_VAL& in_data, AXI_VAL& out_data, int value, int loop) {
	axi_type t = in_data.read();
	int temp;
	temp = t.data;
	if (loop) 
		t.data = temp;
	else 
		t.data = value;
	out_data.write(t);
	return temp;
}

void cnn(AXI_VAL& in_data, AXI_VAL& out_data) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=in_data
#pragma HLS INTERFACE axis port=out_data
	while (true) {
		int parameters[17];
		// get parameters
		for (int idx = 0; idx < 17; idx++) {
			parameters[idx] = axi_transfer(in_data, out_data, 0, 1);
		}
		// common parameters
		int input_size[3];
		int stride_size[2];
		int relu_activation;
		int load_input, load_weight;
		int h_result, w_result, pooling_mode;
		float input[30000], weight[30000], bias[2000];
		float precision;
		int filter_size[4], window_size[2];
		int bias_activation;

		input_size[0] = parameters[2]; // input depth
		input_size[1] = parameters[3]; // input height
		input_size[2] = parameters[4]; // input depth

		//  pooling / convolution / fully connected
		if (paramters[0] == 1) {
			window_size[0] = parameters[5]; // pooling window height
			window_size[1] = parameters[6]; // pooling window width
			pooling_mode = parameters[9]; // 0: max, 1: average
			relu_activation = parameters[11]; // 0: off, 1: relu
			precision = parameters[12];
			load_input = parameters[13]; // activate receiver to get new input
			stride_size[0] = parameters[7]; // row
			stride_size[1] = parameters[8]; // column

			if (load_input) {
				for (int idx = 0; idx < parameters[1]; idx++) {
					input[idx] = axi_transfer(in_data, out_data, 1, 0);
				}
			}

			h_result = int((input_size[1]  - window_size[0]) / stride_size[0]);
			w_result = int((input_size[2] - window_size[1]) / stride_size[1]);
			axi_transfer(in_data, out_data, input_size[0] * w_result * h_result, 0);
			axi_transfer(in_data, out_data, h_result, 0);
			axi_transfer(in_data, out_data, w_result, 0);
			
			for (int idx = 0; idx < input_size[0]; idx++) {
			       for (int idy = 0; idy < h_result; idy++) {
				       for (int idz = 0; idz < w_result; idz++) {
				    	   float temp = 0;
					       float pool_value = 0;
#pragma HLS PIPELINE II=1
					       for (int i = 0; i < window_size[0]; i++) {
						       for (int j = 0; j < window_size[1]; j++) {
							       if (pooling_mode == 0) {
								       if (i == 0 && j == 0) {
									       pool_value = input[idx * input_size[1] * input_size[2] + idy * stride_size[0] * input_size[2] + idz * stride_size[1]];
								       }
								       else {
									       temp = input[idx * input_size[1] * input_size[2] + (idy * stride_size[0] + i) * input_size[2] + idz * stride_size[1] + j];
									       if (temp > pool_value)
										       pool_value = temp;
								       }
							       } else if (pooling_mode == 1) {
								       if (i == 0 && j == 0) {
									       pool_value += input[idx * input_size[1] * input_size[2] + idy * stride_size[0] * input_size[2] + idz * stride_size[1]];
								       } else {
								    	   pool_value += input[idx * input_size[1] * input_size[2] + (idy * stride_size[0] + i) * input_size[2] + idz * stride_size[1] + j];
								       }
							       }
						       }
					       }
					       if (pooling_mode == 1) {
						       pool_value = pool_value / (window_size[0] * window_size[1]);
					       }
					       if (relu_activation == 1) {
						       if (pool_value < 0) 
							       pool_value =1;
					       }
					       axi_transfer(in_data, out_data, int(pool_value), 0);
				       }
			       }
			}
		} else if (parameters[0] == 0) {
			filter_size[0] = parameters[5]; // number of filters
			filter_size[1] = parameters[6]; // filter depth
			filter_size[2] = parameters[7]; // filter height
			filter_size[3] = parameters[8]; // filter weight
			load_input = parameters[15];
			load_weight = parameters[16];
			relu_activation = parameters[13];
			bias_activation = parameters[12];
			precision = parameters[14];

			if (load_input) {
				for (int idx = 0; idx < parameters[1]; idx++) {
					input[idx] = axi_transfer(in_data, out_data, 1, 0);
					input[idx] /= precision;
				}
			}
			if (load_weight) {
				for (int idx = 0; idx < filter_size[0]; idx++) {
					bias[idx] = axi_transfer(in_data, out_data, 2, 0);
					bias[idx] /= precision;
				}
			}
			for (int idx = 0; idx < filter_size[0] * filter_size[1] * filter_size[2] * filter_size[3]; idx++) {
				weight[idx] = axi_transfer(in_data, out_data, 3, 0);
				weight[idx] /= precision;
			}
			// calculate output dimension
			h_result = int((input_size[1] - filter_size[2]) / stride_size[0] + 1);
			w_result = int((input_size[2] - filter_size[3]) / stride_size[1] + 1);
			// send results to cpu
			axi_transfer(in_data, out_data, filter_size[0] * w_result * h_result, 0);
			axi_transfer(in_data, out_data, h_result, 0);
			axi_transfer(in_data, out_data, w_result, 0);
			// convolution
			for (int idx = 0; idx < filter_size[0]; idx++) {
				for (int idy = 0; idy < h_result; idy++) {
					for (int idz = 0; idz < w_result; idz++) {
						int result_index = idx * h_result * w_result + idy * w_result + idz;
						float convolution_value = 0;
#pragma HLS PIPELINE II=1
						for (int k = 0; k < filter_size[1]; k++) {
							int result_panel = k * input_size[1] * input_size[2];
							for (int j = 0; j < filter_size[2]; j++) {
								int result_row = (idy * stride_size[0] + j) * input_size[2];
								for (int i = 0; i < filter_size[3]; i++) {
									int index = result_panel + result_row + idz * stride_size[1] + i;
									convolution_value += input[index] * weight[idx * filter_size[1] * filter_size[2] * filter_size[3] + k * filter_size[2] * filter_size[3] + j * filter_size[3] + i];
								}
							}
						}
						if (bias_activation != 0) {
							convolution_value += bias[idx];
						}
						if (relu_activation == 1) {
							if (convolution_value < 0) 
								convolution_value = 0;
						}
						convolution_value = convolution_value * precision;
						axi_transfer(in_data, out_data, int(convolution_value), 0);
					}
				}
			}									
		} else if (parameters[0] == 2) {
			// 1-Input size, 2-Output size 3- Relu_Activation, 4-precision, 5-Load_Input, 6- Bias Activation
			input_size[0] = parameters[1];
			relu_activation = parameters[3];
			precision = parameters[4];
			load_input = parameters[5];
			bias_activation = parameters[6];
			float temp = 0;
			if (load_input) {
				for (int idx = 0; idx < input_size[0]; idx++) {
					input[idx] = axi_transfer(in_data, out_data, input_size[0], 0);
					input[idx] /= precision;
				}
			}
			if (bias_activation == 1) {
				for (int idx = 0; idx < parameters[2]; idx++) {
					bias[idx] = axi_transfer(in_data, out_data, 2, 0);
					bias[idx] /= precision;
				}
			}
			for (int idx = 0; idx < parameters[2]; idx++) {
				for (int idy = 0; idy < input_size[0]; idy++) {
					temp = axi_transfer(in_data, out_data, 4, 0) / precision * input[idy];
				}
				if (relu_activation == 1) {
					if (temp < 0) temp = 0;
				}
				if (bias_activation == 1) {
					temp += bias[idx];
				}
				temp = temp * precision;
				weight[idx] = temp;
			}
			for (int idx = 0; idx < parameters[2]; idx++) {
				axi_transfer(in_data, out_data, int(weight[idx]), 0);
			}
		}
	}
}






