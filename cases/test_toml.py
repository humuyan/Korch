import toml

class Config:
    def __init__(self, cut_points, full_graph):
        self.cut_points = cut_points
        self.full_graph = full_graph
    

def write_config(config, file_path, model_name):
    config_dict = {
        "graph_partition": {
            "cut_points": config.cut_points, 
            "full_graph": config.full_graph
        }
    } 
    
    
    with open(file_path, "w") as f:
        f.writelines(f"# Configuration for {model_name}\n",)
        toml.dump(config_dict, f)
    

def read_config(file_path):
    with open(file_path, "r") as f:
        config_dict = toml.load(f)
        graph_partition = config_dict.get("graph_partition", {})
        cut_points = graph_partition.get("cut_points", [])
        full_graph = graph_partition.get("full_graph", [])
        return Config(cut_points, full_graph)

if __name__ == "__main__":
    # For segformer
    config = Config(
        cut_points=[(["input.1"], ["/model/segformer/encoder/Transpose_output_0"]),
            (["/model/segformer/encoder/Transpose_output_0"], ["/model/segformer/encoder/Transpose_1_output_0"]),
            (["/model/segformer/encoder/Transpose_1_output_0"], ["/model/segformer/encoder/Transpose_2_output_0"]),
            (["/model/segformer/encoder/Transpose_2_output_0"], ["/model/decode_head/linear_c.3/proj/MatMul_output_0"]),
            (["/model/segformer/encoder/Transpose_output_0"], ["/model/decode_head/linear_c.0/proj/MatMul_output_0"]),
            (["/model/segformer/encoder/Transpose_1_output_0"], ["/model/decode_head/linear_c.1/proj/MatMul_output_0"]),
            (["/model/segformer/encoder/Transpose_2_output_0"], ["/model/decode_head/linear_c.2/proj/MatMul_output_0"]),
            (["/model/decode_head/linear_c.0/proj/MatMul_output_0", "/model/decode_head/linear_c.1/proj/MatMul_output_0", "/model/decode_head/linear_c.2/proj/MatMul_output_0", "/model/decode_head/linear_c.3/proj/MatMul_output_0"], ["/model/decode_head/Concat_4_output_0"]),
            (["/model/decode_head/Concat_4_output_0"], ["1558"])], 
        full_graph=[7]
    )
    write_config(config, "segformer.toml", "segformer")

    # c = read_config("segformer.toml")
    # print(c.cut_points)
    # print(c.full_graph)

    # For yolox
    config = Config(
        cut_points=[(["onnx::Slice_0"], ["/backbone/backbone/dark3/dark3.1/conv3/act/Mul_output_0"]),
        (["/backbone/backbone/dark3/dark3.1/conv3/act/Mul_output_0"], ["/backbone/backbone/dark4/dark4.1/conv3/act/Mul_output_0"]),
        (["/backbone/backbone/dark4/dark4.1/conv3/act/Mul_output_0"], ["/backbone/lateral_conv0/act/Mul_output_0"]),
        (["/backbone/lateral_conv0/act/Mul_output_0", "/backbone/backbone/dark4/dark4.1/conv3/act/Mul_output_0"], ["/backbone/reduce_conv1/act/Mul_output_0"]),
        (["/backbone/reduce_conv1/act/Mul_output_0", "/backbone/backbone/dark3/dark3.1/conv3/act/Mul_output_0"], ["/backbone/C3_p3/conv3/act/Mul_output_0"]),
        (["/backbone/C3_p3/conv3/act/Mul_output_0", "/backbone/reduce_conv1/act/Mul_output_0"], ["/backbone/C3_n3/conv3/act/Mul_output_0"]),
        (["/backbone/C3_n3/conv3/act/Mul_output_0", "/backbone/lateral_conv0/act/Mul_output_0"], [f"/head/{i}_preds.2/Conv_output_0" for i in ["reg", "obj", "cls"]]),
        (["/backbone/C3_n3/conv3/act/Mul_output_0"], [f"/head/{i}_preds.1/Conv_output_0" for i in ["reg", "obj", "cls"]]),
        (["/backbone/C3_p3/conv3/act/Mul_output_0"], [f"/head/{i}_preds.0/Conv_output_0" for i in ["reg", "obj", "cls"]]),
        ([f"/head/{j}_preds.{i}/Conv_output_0" for i in range(3) for j in ["reg", "obj", "cls"]], ["1171"])], 
        full_graph=[9]
    )
    write_config(config, "yolox.toml", "yolox")


    # For yolov4
    config = Config(
        cut_points=[(["input.1"], ["/down3/conv5/conv.2/Mul_output_0"]),
        (["/down3/conv5/conv.2/Mul_output_0"], ["/down4/conv5/conv.2/Mul_output_0"]),
        (["/down4/conv5/conv.2/Mul_output_0"], ["/neck/conv6/conv.2/LeakyRelu_output_0"]),
        (["/down4/conv5/conv.2/Mul_output_0", "/neck/conv6/conv.2/LeakyRelu_output_0"], ["/neck/conv13/conv.2/LeakyRelu_output_0"]),
        (["/neck/conv13/conv.2/LeakyRelu_output_0", "/down3/conv5/conv.2/Mul_output_0"], ["/neck/conv20/conv.2/LeakyRelu_output_0"]),
        (["/neck/conv20/conv.2/LeakyRelu_output_0"], ["/head/conv2/conv.0/Conv_output_0"]),
        (["/neck/conv13/conv.2/LeakyRelu_output_0", "/neck/conv20/conv.2/LeakyRelu_output_0", "/neck/conv6/conv.2/LeakyRelu_output_0"], ["/head/conv10/conv.0/Conv_output_0", "/head/conv18/conv.0/Conv_output_0"]),
        (["/head/conv2/conv.0/Conv_output_0", "/head/conv10/conv.0/Conv_output_0", "/head/conv18/conv.0/Conv_output_0"], ["1979"])], 
        full_graph=[7]
    )
    write_config(config, "yolov4.toml", "yolov4")

