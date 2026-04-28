"""一次性 inspect ONNX 元数据：opset、输入名/形状/dtype、输出形状、节点总数。

读取 ``fusion/full_net_v2_no_road.onnx`` 与 ``fusion/full_net_v2_with_road.onnx``
（路径相对仓库根硬编码），不存在的文件自动跳过。

用法（在 new_plan/ 下）::

    cd new_plan
    $env:PYTHONPATH = "$PWD"
    python -m fusion.code._inspect_onnx

前置：先跑 ``python -m fusion.code.export_onnx`` 把两份 ONNX 导到 fusion/ 目录。
"""
import os
import sys
from pathlib import Path

import onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
files = [
    REPO_ROOT / "fusion" / "full_net_v2_no_road.onnx",
    REPO_ROOT / "fusion" / "full_net_v2_with_road.onnx",
]

for fn in files:
    if not fn.exists():
        print(f"[Skip] {fn} 不存在")
        continue
    m = onnx.load(fn.as_posix())
    print("=" * 60)
    print(fn.name)
    opsets = [(imp.domain or "ai.onnx", imp.version) for imp in m.opset_import]
    print(f"  opset_import: {opsets}")
    print(f"  inputs ({len(m.graph.input)}):")
    for i in m.graph.input:
        shape = [
            d.dim_param if d.dim_param else d.dim_value
            for d in i.type.tensor_type.shape.dim
        ]
        elem_t = onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type)
        print(f"    {i.name:<14} dtype={elem_t:<8} shape={shape}")
    print(f"  outputs ({len(m.graph.output)}):")
    for o in m.graph.output:
        shape = [
            d.dim_param if d.dim_param else d.dim_value
            for d in o.type.tensor_type.shape.dim
        ]
        elem_t = onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type)
        print(f"    {o.name:<14} dtype={elem_t:<8} shape={shape}")
    print(f"  total nodes: {len(m.graph.node)}")
    print(f"  file size: {os.path.getsize(fn) / 1024:.1f} KB")
