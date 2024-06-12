import tvm
import os
from tvm import rpc
import toml
import argparse
from pathlib import Path

class Target:
    def __init__(self, target_device_name: str):
        self.target_device_name = target_device_name
        self.device, self.target, self.rpc_config = configure_target(target_device_name)

    def get_target_device_name(self):
        return self.target_device_name

    def get_device(self):
        return self.device

    def get_target(self):
        return self.target

    def get_rpc_config(self):
        return self.rpc_config

class GraphConfig:
    def __init__(self, path=None):
        if path != None and os.path.exists(path):
            with open(path, "r") as f:
                config_dict = toml.load(f)
                self.graph_partition = config_dict["graph_partition"]
                self.cut_points = self.graph_partition["cut_points"]
                self.full_graph = self.graph_partition["full_graph"]
                self.specified_subgraph = getattr(self.graph_partition, "specified_subgraph", None)
        else:
            self.graph_partition = None
            self.cut_points = []
            self.full_graph = []
            self.specified_subgraph = None


def configure_target(TARGET_DEVICE):
    # Configure target device for kernel profiling
    # Max shared memory per block: see tuning guide on https://docs.nvidia.com/cuda/
    if TARGET_DEVICE == 'v100':
        DEVICE = tvm.cuda()
        TARGET = tvm.target.cuda(arch='sm_70' , 
                                options="-max_threads_per_block=1024 \
                                -max_shared_memory_per_block=96000")
        RPC_CONFIG = []
        
        return DEVICE, TARGET, RPC_CONFIG
            
    elif TARGET_DEVICE == 'a100':
        DEVICE = tvm.cuda()
        TARGET = tvm.target.cuda(arch='sm_80' , 
                                options="-max_threads_per_block=1024 \
                                -max_shared_memory_per_block=163000")
        RPC_CONFIG = []
        
        return DEVICE, TARGET, RPC_CONFIG
            
    elif TARGET_DEVICE == 'a5000':
        DEVICE = tvm.cuda()
        TARGET = tvm.target.cuda(arch='sm_86' , 
                                options="-max_threads_per_block=1024 \
                                -max_shared_memory_per_block=99000")
        RPC_CONFIG = []
        
        return DEVICE, TARGET, RPC_CONFIG
    elif TARGET_DEVICE == 'android':
        # TODO: experimental
        # DEVICE = tvm.opencl()

        # by default on CPU target will execute.
        # select 'cpu', 'opencl' and 'opencl -device=adreno'
        # TEST_TARGET = "opencl -device=adreno"

        # Change target configuration.
        # Run `adb shell cat /proc/cpuinfo` to find the arch.
        # arch = "aarch64"
        # TARGET = tvm.target.Target("llvm -mtriple=%s-linux-android" % arch)
        
        arch = "arm64"
        target = "llvm -num-cores 1 -mtriple=%s-linux-android" % arch
        TARGET = tvm.target.Target(target)
        # TARGET = tvm.target.Target("llvm -num-cores 1 -device=arm_cpu -mtriple=aarch64-linux-gnu")
        
        # if TEST_TARGET.find("opencl"):
        #     TARGET = tvm.target.Target(TEST_TARGET, host=TARGET)
        
        # Get RPC related settings.
        rpc_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
        rpc_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
        rpc_key = "android"
        
        RPC_CONFIG = [rpc_host, rpc_port, rpc_key]

        tracker = rpc.connect_tracker(rpc_host, rpc_port)
        remote = tracker.request(rpc_key, priority=0, session_timeout=60)
        DEVICE = remote.cpu(0)
        
        return DEVICE, TARGET, RPC_CONFIG
        
    else:
        assert(f'TARGET_DEVICE {TARGET_DEVICE} not supported')


def sanitize_prepare_args(args: argparse.Namespace):
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file {args.model} not found")
        
    if not args.model.endswith(".onnx"):
        raise ValueError("Model file must be in ONNX format")
    
    if args.config is not None:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file {args.config} not found")
    config = GraphConfig(args.config)

    model_name = Path(args.model).stem
    if args.database_dir is None:
        # TODO: make use of OS's temp folder
        args.database_dir = f"./temp/{model_name}/"
    if not os.path.exists(args.database_dir):
        os.makedirs(args.database_dir)
    
    if args.code_output_dir is not None:
        if not os.path.exists(args.code_output_dir):
            os.makedirs(args.code_output_dir)
    
    target_info = Target(args.device)
    
    return args, config, target_info



