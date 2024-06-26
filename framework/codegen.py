from collections import defaultdict
from functools import reduce
import os
import onnx_graphsurgeon as gs
from pathlib import Path
import re


class CodeGenerator():
    def __init__(self, execution_order, selected_kernels, kernel_output_lifetimes, onnx_graph, input_path=None, output_path=os.path.join(os.getcwd(), "code_output", "forward_pass.cu")):
        if not input_path:
            print("CodeGen input path is not specified")
            exit(1)
        self.execution_order = execution_order
        self.selected_kernels = selected_kernels
        self.kernel_output_lifetimes = kernel_output_lifetimes
        self.output_path = output_path
        self.f = None
        self.input_path = input_path
        self.onnx_graph = onnx_graph

        self.indents = 0
    
    def get_kernel_launch_config(self, kernel_id):
        gridDim_x = None
        blockDim_x = None
        with open(os.path.join(self.input_path, f'{kernel_id}.CUDAlaunch.config')) as config_file:
            lines = config_file.readlines()
            for line in lines:
                if 'blockIdx.x' in line and not gridDim_x:
                    gridDim_x = line.split('=')[1]
                if 'threadIdx.x' in line and not blockDim_x:
                    blockDim_x = line.split('=')[1]
        # remove i64 type hint
        to_remove = ['i64', ';', '{']
        for char in to_remove:
            gridDim_x = gridDim_x.replace(char, '').strip()
            blockDim_x = blockDim_x.replace(char, '').strip()
        gridDim_x = int(gridDim_x)
        blockDim_x = int(blockDim_x)
        return gridDim_x, blockDim_x

    def get_kernel_parameter_shapes(self, kernel_id):
        parameters_shape_dict = dict()
        parameter_lines = []
        with open(os.path.join(self.input_path, f'{kernel_id}.CUDAlaunch.config')) as config_file:
            lines = config_file.readlines()
            to_include = False
            for line in lines:
                if 'buffers' in line:
                    parameter_lines.append(line)
                    to_include = True
                elif 'buffer_map' in line:
                    break
                else:
                    if to_include:
                        parameter_lines.append(line)
        # print(parameter_lines)
        for line in parameter_lines:
            param_info = line.split(', ')
            shape = param_info[2:][:-1]
            to_remove = ['i64', '[', ']']
            for char in to_remove:
                shape = [s.replace(char, '').strip() for s in shape]
            shape = [int(s) for s in shape]
            name = param_info[0].split(': ')[1][len("Buffer("):-2]
            parameters_shape_dict[name] = shape
        
        # print(parameters_shape_dict)
        
        parameter_names = []
        with open(os.path.join(self.input_path, f'{kernel_id}.cu')) as kernel_file:
            func_signature = ""
            kernel_code = kernel_file.readlines()
            for line in kernel_code:
                if 'extern "C"' in line:
                    func_signature = line
                    break
            # Regular expression to find parameter names
            # This regex matches `float* __restrict__` followed by one or more word characters (parameter names)
            pattern = r"float\*\s+__restrict__\s+(\w+)"

            # Find all matches in the function signature
            parameter_names = re.findall(pattern, func_signature)

        shape_in_order = [tuple(parameters_shape_dict[name]) for name in parameter_names]
        return list(zip(parameter_names, shape_in_order))


    def write_line(self, line):
        try:
            self.f.write(self.indents*"    " + line + "\n")
        except IOError:
            print(f"Error: can't write to file {self.output_path}, line was: {line}")
            exit(1)

    def write_kernel(self, kernel_id):
        try:
            with open(os.path.join(self.input_path, f"{kernel_id}.cu"), 'r') as kernel_file:
                kernel_code = kernel_file.readlines()
                self.f.writelines(kernel_code)
        except IOError:
            print(f"Error: can't open file {self.input_path}/{kernel_id}.cu") 
            exit(1)

    
    def generate_code(self):
        # try:
        output_dir_parent = Path(self.output_path).parent
        output_dir_parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.output_path, 'w+')
        self.write_line('// This file is generated by the code generator. See codegen.py')
        
        #? Preprcessor directives
        preprocessor_directives = ['#include <cudnn.h>', 
                                   '#include <iostream>',
                                   '#include <fstream>', 
                                   '#include <limits>', 
                                   '#define CONV_MODE CUDNN_CROSS_CORRELATION'] # todo: add header files that are needed by tensor RT operators
        for directive in preprocessor_directives:
            self.write_line(directive)
        #? Global Variables
        self.write_line('const cudnnDataType_t DATA_TYPE = CUDNN_DATA_FLOAT;')
        self.write_line('cudnnMathType_t MATH_TYPE = CUDNN_DEFAULT_MATH;')
        self.write_line('cudnnTensorFormat_t INPUT_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;')
        self.write_line('cudnnTensorFormat_t OUTPUT_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;')
        self.write_line('cudnnTensorFormat_t KERNEL_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;')
        self.write_line('size_t workspace_size = (size_t) 1024 * 1024 * 1024;')
        self.write_line('cudnnHandle_t cudnn;')
        self.write_line('cudnnTensorDescriptor_t input_descriptor, output_descriptor, bias_descriptor;')
        self.write_line('cudnnFilterDescriptor_t kernel_descriptor;')
        self.write_line('cudnnConvolutionDescriptor_t convolution_descriptor;')
        self.write_line('cudnnActivationDescriptor_t activation_descriptor;')
        
        #? Constants
        constant_tensors = []
        tensors = self.onnx_graph.tensors()
        for tensor_name in tensors:
            tensor = tensors[tensor_name]
            if isinstance(tensor, gs.Constant):
                host_tensor_name = 'host_' + tensor.name
                device_tensor_name = 'device_' + tensor.name
                tensor_value = tensor.values
                tensor_size = reduce(lambda x,y: x*y, list(tensor_value.shape))
                flattened_array = tensor_value.flatten()
                array_string = ', '.join(map(str, flattened_array))
                self.write_line(f'const float {host_tensor_name}[] = {"{" + array_string + "}"};')
                # self.write_line(f'__constant__ float {device_tensor_name}[{tensor_size}];')
                constant_tensors.append((tensor, tensor_size))   
           
        #? Each candidate kernel in execution order
        for kernel_id in self.execution_order:
            _, params, kernal_name = self.selected_kernels[kernel_id]
            # Only generate code for memory-bound kernels, we call vendor APIs for compute-bound kernels
            if not params:
                self.write_kernel(kernal_name)
            else:
                print(f"Kernel {kernal_name} is compute-bound, we call vendor APIs for it.")
        #? Main entrance: 
        self.write_line('int main(int argc, char* argv[]) {')
        self.indents = 1    
        self.write_line('if (argc != 3) {')
        self.indents = 2
        self.write_line('std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;')
        self.write_line('return 1;')
        self.indents = 1
        self.write_line('}')
        self.write_line('std::ifstream input_file(argv[1]);')
        self.write_line('if (!input_file) {')
        self.indents = 2
        self.write_line('std::cerr << "Error: cannot open file " << argv[1] << std::endl;')
        self.write_line('return 1;')
        self.indents = 1
        self.write_line('}')
        self.write_line('int n;')
        self.write_line('input_file >> n;')

        # CUDNN initialization
        self.write_line('cudnnCreate(&cudnn);')
        self.write_line('cudnnCreateTensorDescriptor(&input_descriptor);')
        self.write_line('cudnnCreateTensorDescriptor(&output_descriptor);')
        self.write_line('cudnnCreateFilterDescriptor(&kernel_descriptor);')
        self.write_line('cudnnCreateConvolutionDescriptor(&convolution_descriptor);')
        self.write_line('cudnnCreateActivationDescriptor(&activation_descriptor);')
        self.write_line('cudnnCreateTensorDescriptor(&bias_descriptor);')
        self.write_line('cudnnSetConvolutionMathType(convolution_descriptor, MATH_TYPE);')
        self.write_line('float *workspace;')
        self.write_line('cudaMalloc(&workspace, workspace_size);')
        self.write_line('const float alpha = 1, beta = 0;')

        variable_names = set()
        def declare_variable(var_name):
            if var_name not in variable_names:
                self.write_line(f'float* {var_name};')
                variable_names.add(var_name)

        # Input allocation
        first_kernel, param, _ = self.selected_kernels[self.execution_order[0]]
        input_name = first_kernel.inputs[0].name # !Assume there is only one input tensor for the entire computation graph
        input_shape = first_kernel.inputs[0].shape
        host_input_name = 'host_' + input_name
        device_input_name = 'device_' + input_name
        declare_variable(host_input_name)
        declare_variable(device_input_name)
        self.write_line(f'cudaHostAlloc((void **)&{host_input_name}, {input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]} * sizeof(float), cudaHostAllocDefault);')
        # Fill the input tensor from input text file
        self.write_line(f'// Fill the input tensor {host_input_name}')

        self.write_line('for (int i = 0; i < n; i++) {')
        self.indents = 2
        self.write_line(f'input_file >> {host_input_name}[i];')
        self.indents = 1
        self.write_line('}')
        self.write_line('input_file.close();')
        
        self.write_line(f'cudaMalloc((void **)&{device_input_name}, {input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]} * sizeof(float));')
        self.write_line(f'cudaMemcpy({device_input_name}, {host_input_name}, {input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]} * sizeof(float), cudaMemcpyHostToDevice);')

        for i in range(len(self.execution_order)):
            kernel_id = self.execution_order[i]
            kernel, params, kernel_name = self.selected_kernels[kernel_id]
            input_list = kernel.inputs
            constant_inputs = []
            for node in kernel.nodes:
                for tensor in node.inputs:
                    if isinstance(tensor, gs.Constant):
                        constant_inputs.append(tensor)
                        input_list.append(tensor)

            # Set up constant input tensors of current kernel
            for c_input in constant_inputs:
                host_tensor_name = 'host_' + c_input.name
                device_tensor_name = 'device_' + c_input.name
                tensor_size = reduce(lambda x,y: x*y, c_input.shape)
                declare_variable(device_tensor_name)
                self.write_line(f'cudaMalloc((void **)&{device_tensor_name}, {tensor_size} * sizeof(float));')
                self.write_line(f'cudaMemcpy({device_tensor_name}, {host_tensor_name}, {tensor_size} * sizeof(float), cudaMemcpyHostToDevice);')

            kernel_output_name = kernel.outputs[0].name
            device_kernel_output_name = 'device_' + kernel_output_name
            kernel_output_size = reduce(lambda x,y: x*y, kernel.outputs[0].shape)
            declare_variable(device_kernel_output_name)
            self.write_line(f'cudaMalloc((void **)&{device_kernel_output_name}, {kernel_output_size} * sizeof(float));')

            if not params:
                # Generate kernel launch code for memory-bound kernels
                gridDim_x, blockDim_x = self.get_kernel_launch_config(kernel_name)
                output_formal_param = device_kernel_output_name
                # ! Order of input is to be confirmed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                kernel_param_placeholders_names_and_shapes = self.get_kernel_parameter_shapes(kernel_name)

                computation_graph_shape2tensor = defaultdict(list)
                for j in range(len(input_list)):
                    tensor_name, tensor_shape = input_list[j].name, tuple(input_list[j].shape)
                    computation_graph_shape2tensor[tensor_shape].append(tensor_name)
                
                kernel_launch_param_names_in_order = []
                # print("Kernel name: ", kernel_name)
                for k in range(len(kernel_param_placeholders_names_and_shapes)):
                    placeholder_name, param_shape = kernel_param_placeholders_names_and_shapes[k]
                    if 'T_' in placeholder_name:
                        # if contains 'T_', this placeholder is the output tensor    
                        kernel_launch_param_names_in_order.append(device_kernel_output_name)
                    else:
                        # if not, this placeholder is an input tensor
                        potential_tensor_names = computation_graph_shape2tensor[param_shape]
                        # print("computation_graph_shape2tensor: ", computation_graph_shape2tensor)
                        # print("param_shape:", param_shape)
                        # print("potential_tensor_names:", potential_tensor_names)

                        # if multiple input tensor has corresponding shape, we choose based on the order they appear in input_list
                        
                        longest_tensor_name_index = -1
                        max_length = 0
                        for p in range(len(potential_tensor_names)):
                            if len(potential_tensor_names[p]) > max_length:
                                max_length = len(potential_tensor_names[p])
                                longest_tensor_name_index = p
                        

                        chosen_tensor_name = potential_tensor_names.pop(longest_tensor_name_index)

                        # chosen_tensor_name = potential_tensor_names.pop(0)
                        kernel_launch_param_names_in_order.append(f'device_{chosen_tensor_name}')

                formal_params_str = ', '.join(kernel_launch_param_names_in_order)
                kernel_name = f'kernel_{kernel_name}'
                # print(f"{kernel_name}'s parameter placeholders and shapes are: {kernel_param_placeholders_names_and_shapes}")
                self.write_line(f'{kernel_name}<<<{gridDim_x}, {blockDim_x}>>>({formal_params_str});')

            else:
                # Call vendor APIs for compute-bound kernels
                if params['type'] == 'conv':
                    device_kernel_input_name = "device_" + kernel.inputs[0].name
                    input_channels_per_group = params['in_channels'] // params['groups']
                    out_height = (params['in_height'] + 2 * params['padding'] - params['dilation'] * (params['kernel_size'] - 1) - 1) // params['stride'] + 1
                    out_width = (params['in_width'] + 2 * params['padding'] - params['dilation'] * (params['kernel_size'] - 1) - 1) // params['stride'] + 1
                    self.write_line(f'cudnnSetTensor4dDescriptor(input_descriptor, INPUT_TENSOR_FORMAT, DATA_TYPE, {params["batch_size"]}, {params["in_channels"]}, {params["in_height"]}, {params["in_width"]});')
                    self.write_line(f'cudnnSetTensor4dDescriptor(output_descriptor, OUTPUT_TENSOR_FORMAT, DATA_TYPE, {params["batch_size"]}, {params["out_channels"]}, {out_height}, {out_width});')
                    self.write_line(f'cudnnSetFilter4dDescriptor(kernel_descriptor, DATA_TYPE, KERNEL_TENSOR_FORMAT, {params["out_channels"]}, {input_channels_per_group}, {params["kernel_size"]}, {params["kernel_size"]});')
                    self.write_line(f'cudnnSetConvolution2dDescriptor(convolution_descriptor, {params["padding"]}, {params["padding"]}, {params["stride"]}, {params["stride"]}, {params["dilation"]}, {params["dilation"]}, CONV_MODE, DATA_TYPE);')
                    weight_name, bias_name = '', ''
                    for c_input in constant_inputs:
                        if len(c_input.shape) == 4:
                            weight_name = "device_" + c_input.name
                        else:
                            bias_name = "device_" + c_input.name
                    if params['groups'] > 1:
                        self.write_line(f'cudnnSetConvolutionGroupCount(convolution_descriptor, {params["groups"]});')
                        self.write_line(f'cudnnConvolutionForward(cudnn, &alpha, input_descriptor, {device_kernel_input_name}, kernel_descriptor, {weight_name}, convolution_descriptor, (cudnnConvolutionFwdAlgo_t){params["algo"]}, workspace, workspace_size, &beta, output_descriptor, {device_kernel_output_name});')
                    if params['mode'] != 0:
                        self.write_line(f'cudnnSetTensor4dDescriptor(bias_descriptor, OUTPUT_TENSOR_FORMAT, DATA_TYPE, 1, {params["out_channels"]}, 1, 1);')
                        self.write_line(f'cudnnSetActivationDescriptor(activation_descriptor, {"CUDNN_ACTIVATION_RELU" if params["mode"] == 2 else "CUDNN_ACTIVATION_IDENTITY"}, CUDNN_NOT_PROPAGATE_NAN, std::numeric_limits<float>::infinity());')
                        self.write_line(f'cudnnConvolutionBiasActivationForward(cudnn, &alpha, input_descriptor, {device_kernel_input_name}, kernel_descriptor, {weight_name}, convolution_descriptor, (cudnnConvolutionFwdAlgo_t){params["algo"]}, workspace, workspace_size, &beta, output_descriptor, {device_kernel_output_name}, bias_descriptor, {bias_name}, activation_descriptor, output_descriptor, {device_kernel_output_name});') 
                else:
                    print("Unsupported compute-bound kernel type")
                    exit(1)
            
            # Free tensors that are out of lifetimes
            for expired_kernel_id in self.kernel_output_lifetimes[i]:
                expired_kernel = self.selected_kernels[expired_kernel_id][0]
                device_expired_kernel_output_name = 'device_' + expired_kernel.outputs[0].name
                self.write_line(f'cudaFree({device_expired_kernel_output_name});')
            # Free constant input tensors of current kernel
            for c_input in constant_inputs:
                device_tensor_name = 'device_' + c_input.name
                self.write_line(f'cudaFree({device_tensor_name});')
            
            
            
            self.write_line('')
        
        # Free input tensor
        self.write_line(f'cudaFree({device_input_name});')
        self.write_line(f'cudaFreeHost({host_input_name});')
        
        # Transfer back output from device to host
        last_kernel, _, _ = self.selected_kernels[self.execution_order[-1]]
        last_kernel_output_name = last_kernel.outputs[0].name
        last_kernel_host_output_name = 'host_' + last_kernel_output_name
        last_kernel_device_output_name = 'device_' + last_kernel_output_name
        last_kernel_output_size = reduce(lambda x,y: x*y, last_kernel.outputs[0].shape)
        self.write_line(f'float* host_{last_kernel_output_name} = (float*) malloc({last_kernel_output_size} * sizeof(float));')
        self.write_line(f'cudaMemcpy({last_kernel_host_output_name}, {last_kernel_device_output_name}, {last_kernel_output_size} * sizeof(float), cudaMemcpyDeviceToHost);')
        self.write_line('std::ofstream output_file(argv[2]);')
        self.write_line('if (!output_file) {')
        self.indents = 2
        self.write_line('std::cerr << "Error: cannot open file " << argv[2] << std::endl;')
        self.write_line('return 1;')
        self.indents = 1
        self.write_line('}')
        # self.write_line('output_file << n << std::endl;')
        self.write_line(f'for (int i = 0; i < {last_kernel_output_size}; i++) {"{"}')
        self.indents = 2
        self.write_line(f'output_file << {last_kernel_host_output_name}[i] << " ";')
        self.indents = 1
        self.write_line('}')
        self.write_line('output_file.close();')


        # Free output tensor
        self.write_line(f'cudaFree({last_kernel_device_output_name});')

        # CUDNN destruction
        self.write_line('cudaFree(workspace);')
        self.write_line('cudnnDestroyTensorDescriptor(input_descriptor);')
        self.write_line('cudnnDestroyTensorDescriptor(output_descriptor);')
        self.write_line('cudnnDestroyFilterDescriptor(kernel_descriptor);')
        self.write_line('cudnnDestroyTensorDescriptor(bias_descriptor);')
        self.write_line('cudnnDestroyActivationDescriptor(activation_descriptor);')
        self.write_line('cudnnDestroyConvolutionDescriptor(convolution_descriptor);')
        self.write_line('cudnnDestroy(cudnn);')
    
        self.indents = 0
        self.write_line('}')
        self.f.close()
        # except IOError:
            # print(f"Error: can't open file {self.output_path}")
