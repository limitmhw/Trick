import importlib
import inspect
import pkgutil
import transformers
from types import ModuleType
import torch.nn as nn
import ast

#=======================================================================================================================
def get_member_assignments(code: str, class_name: str):
    tree = ast.parse(code)
    members = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for stmt in item.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if (isinstance(target, ast.Attribute)
                                        and isinstance(target.value, ast.Name)
                                        and target.value.id == "self"):
                                    var_name = target.attr
                                    value = stmt.value
                                    members[var_name] = infer_type(value)
    return members

def infer_type(value):
    """推断右侧表达式的类型"""
    if isinstance(value, ast.Call):
        # 函数调用
        if isinstance(value.func, ast.Attribute):
            if isinstance(value.func.value, ast.Name):
                return f"{value.func.value.id}.{value.func.attr}"
            elif isinstance(value.func.value, ast.Attribute):
                # 支持 torch.nn.Linear 这种链式调用
                return get_full_attr_name(value.func)
        elif isinstance(value.func, ast.Name):
            return value.func.id
        return "Call"
    elif isinstance(value, ast.Constant):
        return "Constant"
    elif isinstance(value, ast.Name):
        return f"Reference({value.id})"
    elif isinstance(value, ast.BinOp):
        return "Expression(BinOp)"
    elif isinstance(value, ast.Attribute):
        return get_full_attr_name(value)
    elif isinstance(value, ast.IfExp):
        return "Expression(IfExp)"
    return type(value).__name__

def get_full_attr_name(node):
    """递归获取 Attribute 的完整名称"""
    if isinstance(node, ast.Attribute):
        return get_full_attr_name(node.value) + "." + node.attr
    elif isinstance(node, ast.Name):
        return node.id
    return "Unknown"

def get_op_in_class_init(code, class_name, filter_str_list):
    members = get_member_assignments(code, class_name)
    result = {filter_str: [] for filter_str in filter_str_list}
    for k, v in members.items():
        for filter_str in filter_str_list:
            if filter_str.lower() in v.lower():
                result[filter_str].append((k, v))
    return result

#=======================================================================================================================
def iter_submodules(package: ModuleType):
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            mod = importlib.import_module(name)
            yield mod
        except Exception:
            pass  

def get_decoder_layers():
    transformers_decoders = []
    import transformers.models as models
    for submodule in iter_submodules(models):
        modeling_items = [item for item in dir(submodule) if item.startswith('modeling')]
        for modeling_item in modeling_items:
            modeling = getattr(submodule, modeling_item)
            decoders = [
                item for item in dir(modeling)
                if 'decoder' in item.lower()
                and inspect.isclass(getattr(modeling, item))
                and issubclass(getattr(modeling, item), nn.Module)
            ]
            attentions = [
                item for item in dir(modeling)
                if 'attention' in item.lower()
                and inspect.isclass(getattr(modeling, item))
                and issubclass(getattr(modeling, item), nn.Module)
            ]
            transformers_decoders.append([f"{submodule.__name__}.{modeling_item}", decoders, attentions])
    return transformers_decoders

def analysis_all_decoder_layers():
    transformers_decoders = get_decoder_layers()
    count = 0
    awq_config = {}

    for decoder_info in transformers_decoders:
        module_name, decoder_names, attention_names = decoder_info
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            print(f"\033[91mFailed to import module: {module_name}")
            continue

        for decoder_class_name in decoder_names:
            print(f"\n\n\nProcessing: {module_name} {decoder_class_name},{attention_names}")
            decoder_class = getattr(module, decoder_class_name)
            source = inspect.getsource(decoder_class)

            # decoder layer
            decoder_res = get_op_in_class_init(source, decoder_class_name, ["norm", "Attention", "mlp"])
            if len(decoder_res["norm"]) >= 1 and len(decoder_res["Attention"]) == 1:
                print(f"\033[94mdecoder_class {decoder_res}")
                prev_op = decoder_res["norm"][0][0]

                class_name = decoder_res["Attention"][0][1]
                layers_prefix = decoder_res["Attention"][0][0]
                print(f"attention_class {class_name}")
                try:
                    attention_class = getattr(module, class_name)
                except:
                    print(f"\033[91mSkip {decoder_class_name}: attention class {class_name} not found in {module_name}")
                    continue
                source = inspect.getsource(attention_class)
                attention_res = get_op_in_class_init(source, class_name, ["linear"])
                print("attention_res:", attention_res)
                if len(attention_res["linear"]) < 2:
                    print(f"\033[91mSkip {decoder_class_name}: attention linear layers < 2 -> {attention_res}")
                    continue

                #-----------------------------------------------------
                layers = [layers_prefix + "." + k[0] for k in attention_res["linear"][:-1]]
                print("-------------------------------------------------------")
                print("prev_op:", prev_op)
                print("layers:", layers)
                print("inp:", layers[0])
                print("module2inspect:", layers_prefix)
                awq_config[decoder_class_name] = []
                awq_config[decoder_class_name].append(
                    {"prev_op": prev_op, "inp": layers[0], "module2inspect": layers_prefix, "layers": layers}
                )
                prev_op = layers_prefix + "." + attention_res["linear"][-2][0]
                layers = [layers_prefix + "." + attention_res["linear"][-1][0]]
                print("-------------------------------------------------------")
                print("prev_op:", prev_op)
                print("layers:", layers)
                print("inp:", layers[0])
                print("module2inspect:", layers[0])
                awq_config[decoder_class_name].append(
                    {"prev_op": prev_op, "inp": layers[0], "module2inspect": layers[0], "layers": layers}
                )
                #-------------------------------------------------------------------------
                if len(decoder_res["mlp"]) != 1:
                    print(f"\033[91mSkip {decoder_class_name}: expected 1 mlp, got {len(decoder_res['mlp'])} -> {decoder_res}")
                    continue
                class_name = decoder_res["mlp"][0][1]
                try:
                    mlp_class = getattr(module, class_name)
                except:
                    print(f"\033[91mSkip {decoder_class_name}: mlp class {class_name} not found in {module_name}")
                    continue

                source = inspect.getsource(mlp_class)
                mlp_res = get_op_in_class_init(source, class_name, ["linear"])
                if len(mlp_res["linear"]) not in [2, 3]:
                    print(f"\033[91mSkip {decoder_class_name}: mlp linear layers not in [2,3] -> {mlp_res}")
                    continue
                print("-------------------------------------------------------")
                if len(mlp_res["linear"]) == 2:
                    prev_op = [layers_prefix + "." + mlp_res["linear"][0][0]]
                    layers = [layers_prefix + "." + mlp_res["linear"][-1][0]]
                else:
                    prev_op = [layers_prefix + "." + mlp_res["linear"][1][0]]
                    layers = [layers_prefix + "." + mlp_res["linear"][-1][0]]

                print("prev_op:", prev_op)
                print("layers:", layers)
                print("inp:", layers[0])
                print("module2inspect:", layers[0])
                awq_config[decoder_class_name].append(
                    {"prev_op": prev_op, "inp": layers[0], "module2inspect": layers[0], "layers": layers}
                )

                if len(decoder_res["norm"]) != 2:
                    print(f"\033[91mSkip {decoder_class_name}: expected 2 norm, got {len(decoder_res['norm'])} -> {decoder_res}")
                    continue

                prev_op = decoder_res["norm"][1][0]
                layers_prefix = decoder_res["mlp"][0][0]
                if len(mlp_res["linear"]) == 2:
                    layers = [layers_prefix + "." + mlp_res["linear"][0][0]]
                else:
                    layers = [
                        layers_prefix + "." + mlp_res["linear"][0][0],
                        layers_prefix + "." + mlp_res["linear"][1][0]
                    ]

                print("prev_op:", prev_op)
                print("layers:", layers)
                print("inp:", layers[0])
                print("module2inspect:", layers_prefix)
                awq_config[decoder_class_name].append(
                    {"prev_op": prev_op, "inp": layers[0], "module2inspect": layers_prefix, "layers": layers}
                )
  
                print("-------------------------------------------------------")
                count = count + 1
            else:
                print(f"\033[91mSkip {decoder_class_name}: norm/attention structure not matched -> {decoder_res}")

    print(f"\n\n\n\nall support transformers decoders: {100 * count / len(transformers_decoders):.2f}%")
    print(f"support transformers decoders: {100 * count / len(awq_config):.2f}%")
    return awq_config


if __name__ == "__main__":
    awq_config = analysis_all_decoder_layers()
