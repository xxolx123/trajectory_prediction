common/
=======

所有子网络可复用的公共组件。

文件：
  - scaler.py           StandardScaler 归一化工具（fit / transform / save / load）
  - context_schema.py   ContextBatch + build_dummy_context（供 gnn1 / gnn2 /
                        constraint_optimizer / fusion 的外部上下文输入使用）
  - outlier_filter.py   异常值剔除（当前 pass；fusion 运行时会调用）

使用方式：
  子网络（lstm1 / gnn1 / ... / fusion）把 new_plan/ 加到 PYTHONPATH 之后，
  就可以 `from common.scaler import StandardScaler` 等。
