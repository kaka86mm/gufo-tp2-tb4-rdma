# gufo-tp2-tb4-rdma

**双 AMD Strix Halo（Ryzen AI MAX+ 395）通过 USB4 / 雷电 4 RDMA write striping 跑 gufo TP2（张量并行）大模型推理的一站式工具集。**

[English README](README.md)

两台 128G 统一内存的 Strix Halo 合成一个 250G 推理池：每层 transformer 切分到两台机器，部分和以 RDMA WRITE 跨线缆交换，write striping 把单个 QP 摊到两根 USB4 线的全部 rail 上（[thunderbolt-ibverbs](https://github.com/hellas-ai/thunderbolt-ibverbs) 栈，最高 4 rail × 20 Gb/s × 2 lane），模型经 gufo 的 TP2 传输（[neuhaus/gufo](https://github.com/neuhaus/gufo) `rdma` 分支）对外服务。

## 实测成绩（真实数据，可用 `run/bench/` 复现）

硬件：2× FAEX1（Ryzen AI MAX+ 395，128G 统一内存）、2 根 USB4 40G 线、Ubuntu 24.04 内核 7.0.0-34、模型 `Qwen3.8-Flash-Next UD-Q4_K_XL`（abliterated 版）+ MTP 投机解码（d=7）、上下文 262,144、sessions 12。

| 负载 | 单机 | 旧 TP2 栈（单 rail） | **本仓库 TP2（write striping）** |
|---|---:|---:|---:|
| 单流 decode（重复文本，MTP） | 59.4 t/s | 35 t/s | **66–70 t/s** |
| 8 路并发聚合 | 141 t/s | — | **123 t/s** |
| 12 路并发聚合 | — | — | **136 t/s** |
| Prefill（冷启，16–26K token） | ~1,630 t/s | 137 t/s | **~1,610 t/s** |
| 258K 深度 prefill | — | — | **1,159 t/s** |
| 258K 深度 decode | — | — | **53 t/s** |
| 深上下文追问（258K，磁盘缓存命中） | — | — | **5.3 秒** |

prefill 随深度几乎不衰减、全程高于同传输的公开参考值；decode 一路到满 262K 上下文保持在 53–70 t/s。

## 仓库内容

```
kernel/   build-ws-core.sh     整合内核构建：westeri v7.2-rc1 树 + local.nix 补丁系列
                              + 配套重编 thunderbolt_ibverbs（逐主机本机编译，无符号链接农场）
deploy/   deploy-ws.sh         快路径：stock 内核 core + ibverbs + dummy 网卡 tbv0
                              + systemd 单元（双机）
          deploy-ws-core.sh    整合 core 路径：开机加载 patched core（失败自动回退
                              stock）、blacklist、单元接线
          ws-roce-boot.sh      开机 RDMA 拉起脚本（参数已固化）
          *.service            systemd 单元
run/      start-tp2.sh         TP2 协同启动器（rank0 本机，rank1 走 ssh）
          bench/               decode / 聚合 / 深上下文 / 缓存命中基准
docs/     BENCHMARKS.md        完整数据表（含失败的配置）
          PITFALLS.md          我们踩过的每一个坑，让你不用再踩
```

## 快速开始

**快路径 —— stock 内核（Ubuntu 7.0+，6.14+ 大概率可用）**

1. **在每台主机上**用它自己的内核头文件编译 `thunderbolt_ibverbs`（版本号相同 ≠ 构建相同，见 PITFALLS #1），并带上上游 README 的 TP2 示例里省略的激活参数：
   `profile=linux_perf bind_services=1 allocate_rings=1 start_rings=1 negotiate_native=1 enable_tunnels=1 tbnet=prefer_rdma lanes=2 register_verbs=1 roce_netdev=tbv0 native_write_striping=1`
2. 双机执行部署：blacklist `thunderbolt_net`（它会偷 DMA rail）、建 dummy `tbv0` 网卡承载 RoCE GID（10.77.0.1/.2）、装开机单元。
3. 两根 USB4 线连好，确认两条链路都训练到 `20.0 Gb/s x2`：
   `cat /sys/bus/thunderbolt/devices/*-*/rx_speed` —— 掉速或卡死就拔插线（**温重启不会给 TB PHY 断电复位**，见 PITFALLS #5）。
4. 启动：`./run/start-tp2.sh`（改好顶部的 IP/路径/模型），等 `rdma_ready`。

**整合内核路径** —— 若你的 stock 内核 core 拒绝服务绑定（`bind_services=1` 报 EINVAL，见 PITFALLS #2），用 `kernel/build-ws-core.sh` 在每台主机构建匹配 core 并执行 `deploy/deploy-ws-core.sh`。可获得 source-aware XDomain 控制路径，且单端就能注册 rail。

## 调优结论（实测过，别再折腾）

- `zcopy_min_bytes=4096`：decode 尺寸消息下**更差**（dma-buf 映射开销只有大写才划算）
- `native_write_stripe_min_bytes=16384` / `native_tx_max_inflight=128`：伤单流 decode，默认值最优
- 聚合最优 `sessions=12`；任何并发下 MTP 都保持**开启**（关闭 = 聚合 -45%；verify 批处理能摊销跨机交换）
- MTP `--draft-tokens` 上限 7（Flash-Next 边车硬限制）
- 磁盘缓存：`--cache-disk-staging-bytes` 必须 ≥ 最大快照（262K 会话约 4G），否则深上下文追问会静默重新 prefill；`--cache-disk-bytes` 对齐 GPU KV 余量

## 致谢与许可

- 传输层：[hellas-ai/thunderbolt-ibverbs](https://github.com/hellas-ai/thunderbolt-ibverbs)（含 write-striping 分支）— GPL-2.0
- 引擎：[gufo-org/gufo](https://github.com/gufo-org/gufo) + TP2 RDMA fork [neuhaus/gufo](https://github.com/neuhaus/gufo)（上游评审中）— MIT
- 本仓库脚本：MIT。所引用的内核补丁（不随仓库再分发）位于上游仓库。
