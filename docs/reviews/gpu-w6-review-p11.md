# gpu W6 adversarial review — Part 11（系统集成与生产）

- **Evaluator**: fresh-context adversarial（不继承写作上下文，未派 subagents），read-only（仅写本报告）。
- **Date**: 2026-11-26。
- **Scope**: `gpu/11-system-integration/01-server-forms-mig.md`（11.1）、`02-profiling-tuning.md`（11.2）、`03-faults-reliability.md`（11.3）。
- **Authority**: spec `docs/superpowers/specs/2026-11-24-gpu-dimension-design.md` §5 P11（L435-453）/ §4.3 符号表 / §4.6 R2 / §6.3 强制重推导；T1 `docs/research/gpu-w6-sources.md`；`gpu/references.md` P11；recap 源 models 2.4 §6、gpu 2.1 §4 / 2.4 / 1.1 / 1.4 / 3.3 / 3.4 / 4.1 / 4.2 / 4.3 / 5.4 / 7.1 / 7.2 / 9.3 / 10.2、inference 8.5 / 3.4 / 9.1、training 7.3 / 7.4；primary caches `/tmp/gpu-w6-research/`（mig/xid/ecc/ncu/nsys/dcgm/dgx/smi/vgpu/bw-micro/inside-bw/hopper-wp）。
- **Method**: 全部关键数值独立 Python（`-I`）重算（**207 项 check 全 PASS**，脚本 `/tmp/gpu-w6-p11-recompute.py`）；MIG 几何/profile 表、XID 码语义、ECC/RMA/RAS、nvidia-smi/Nsight/DCGM 逐字逐一对 primary cache（~85 锚）；recap 边界逐句对兄弟源；`lint-chapter.sh` + `lint-math-boundary.py` 机械门 + display 9 禁形/prose \lt/bare <=/process-leak/CJK/保留符号/白名单独立扫描。
- **Diff package**: `.superpowers/sdd/2026-11-26-gpu-w6-system-integration/review-28499af..9520fde.diff`（3 章 + references.md + research；含 T3 fix round 1：11.2 hook gap/wall 口径 + 点标 C→D）。

## 结论速览

| 项 | 结果 |
| --- | --- |
| S1（fatal） | **0** |
| S2（important） | **2**（11.3 NVLink5 计数器版本 v550→应 v560 ×3 处；11.2 §6 点标口径注跨章坐标恒等误述） |
| S3（minor） | **9**（3 新 + 6 与前任务 reviewer 重合、独立确认） |
| 强制重推导（11.1 MIG 分区 / 11.3 ECC-thermal-link / 11.2 bottleneck-roofline） | **全 PASS（207/207）** |
| Recap 边界（11.2←models 2.4/2.4；11.3←2.1 §4 + inference 8.5 + training 7.4/7.3；11.1←7.1/7.2） | **PASS（不重推；ECC 锚 2.1 §4 非 1.2）** |
| MIG 几何 vs MIG User Guide | **PASS（H100 Table 13 / B200 Table 8 / A30 Table 16 逐字；L2 不对称坑注正确）** |
| XID 码语义（31/43/48/63/64/74/79/94/95/149/154；74 Blackwell→149；SXid Hopper-only） | **PASS（Release 620 catalog + archive 七列逐字）** |
| 新 seam（K8S device plugin/调度/编排 = cluster 指针不展开；MIG/vGPU/SR-IOV 硬件 = gpu） | **PASS（三章零展开）** |
| 11.2 R=ridge vs R_th=registers/thread 混场声明 | **PASS（§1 L19）** |
| 11.2 hook gap/wall 33% 口径（非 wall/busy） | **PASS（hook/§1/§2/总结/练习一致 1974/5934=33.3%）** |
| 事实密集章 proof-exercise 定量（11.1/11.3） | **PASS（11.1 练 3 证命题 2 实例上界；11.3 练 3 证命题 3 replay goodput）** |
| lint（chapter + math-boundary） | **3/3 OK** |
| CJK | 11.1=5820（≥2500）/ 11.2=6499 / 11.3=6546（≥3500）**PASS** |
| display 9 禁形 / prose \lt / bare <= / process-leak | **0 / 0 / 0 / 0** |
| 数学白名单 / 保留符号 §4.3 / 总结回指 / 练习分级 | **PASS（白名单外命令 0；总结回指缺失 0；练习 [10][20][20/30][30] 非递减、≥1 证命题、校验锚齐）** |

**总体**：三章数值脊柱全部独立重推闭合（MIG 切片/实例上界/机架功率/N+N/散热/TCO、ECC remap/thermal/replay/链路降级、三段账/四因子分解/带宽饱和加速/MFU/busbw 无一 FAIL）；MIG 几何与 XID 语义对官方一手逐字忠实，L2 跨代不对称与 74→149 代际漂移两个高危坑均正确钉住；recap 边界干净、新 seam 零展开。**2 项 S2 均为局部事实/版本误述（非数值脊柱），T7 必修**；9 项 S3 多为锚偏移/措辞/回写，其中 6 项与前任务 reviewer 的 T7 清单重合（本审独立确认）。

---

## 1. 强制重推导 verification log（独立 Python，`python3 -I`）

脚本 `/tmp/gpu-w6-p11-recompute.py`，**207 项 check，每项 = 独立重算值 vs 章内值，容差 1%（另注者除外）**。不信任章内数字，全部从口径重新推。下表按章列关键项（全量 207 项脚本输出均 PASS）。

### 11.1 MIG 分区资源账 / 机架功率 / TCO（全 PASS）

| 项 | 独立重算 | 章内值 | 判定 |
| --- | --- | --- | --- |
| SM slice 1/7：H100 132/7 | 18.857 | ≈18.9（roughly） | PASS |
| 1g.10gb 容量 80/8 / BW 3350/8 / L2 50/8 | 10 GB / 418.75 / 6.25 MB | 同 | PASS |
| 3g.40gb SM 3/7×132 / BW 4/8×3350 / L2 4/8×50 | 56.57 / 1675 / 25 | ≈56.6 / 1675 / 25 | PASS |
| B200 1g.23gb SM 148/7 / 容量 180/8 / BW 7700/8 | 21.14 / 22.5 / 962.5 | ≈21.1 / 22.5 / 962.5 | PASS |
| profile 名舍入 22.5→23（half-up）/ 186/8=23.25→23 / A30 24/4=6 | 23 / 23 / 6 | 同 | PASS |
| 实例上界 min(7,8)=7；A30 min(4,4)=4 | 7 / 4 | 同 | PASS |
| 7×1g：SM 7/7 用满、显存 7/8、空闲 1/8=10 GB | 1 / 0.875 / 10 | 同 | PASS |
| 4×1g.20gb：显存 8/8 满、SM 4/7、空闲 3/7≈56.6 SM | 1 / 4/7 / 56.57 | 同 | PASS |
| 4g.40gb+3g.40gb：SM 7/7 + 显存 8/8 双满 | 1 / 1 | 同 | PASS |
| H100 1g.20gb L2=1/8=6.25（非按 slice 反推 12.5） | 6.25 / 12.5 | 6.25 / 坑 12.5 | PASS |
| B200 SM 2×74=148 / 物理 2×80=160 / FP32 148×256×1.98G | 148 / 160 / 74.99 TF | 148 / 160 / 75 TF | PASS |
| 148/132=1.12（论文 1.09× 不自洽残留注） | 1.1212 | 1.12 | PASS |
| GPU-only 8×700/8×1000/72×1200 → kW | 5.6 / 8 / 86.4 | 同 | PASS |
| GPU 占比 5.6/10.2 / 8/14.3 / 86.4/120 | 54.9 / 55.9 / 72% | 同 | PASS |
| 机架-GPU-only 差 120−86.4 / 120/10.2 | 33.6 kW / 11.76 | 33.6 / ≈11.8 | PASS |
| 每 GPU 系统功率 10.2/8 / 14.3/8 / 120/72 | 1.275 / 1.7875 / 1.667 | 同 | PASS |
| 每 U 10.2/8 / 14.3/10 | 1.275 / 1.43 | 同 | PASS |
| N+N 0.5×8×6×5.5=132≥120；每 shelf 6×5.5=33 | 132 / 33 | 同 | PASS |
| DGX 4×3.3=13.2≥10.2 / 5×3.3=16.5≥14.3 / boot 3×3.3=9.9<10.2 | 13.2 / 16.5 / 9.9 | 同 | PASS |
| ζ_water/ζ_air=4.18e6/1.2e3 | 3483.3 | ≈3483 | PASS |
| Q_air H100 10200/(1200×15) m³/s → CFM | 0.5667 / 1200.7 | 0.567 / ≈1200 | PASS |
| 1105 CFM→0.5215 m³/s 反推 Δθ / 1550 CFM 反推 | 16.30 / 16.29 K | ≈16.3 / ≈16.3 | PASS |
| NVL72 风 6.667 m³/s≈14126 CFM / 水 1.914e-3 m³/s=1.91 L/s | 6.667 / 14126 / 1.91 | 6.67 / ≈14100 / 1.91 | PASS |
| NVL72/DGX H100 官方风量比 6.667/0.5215 | 12.78 | ≈12.8 | PASS |
| 风/水流量比 = ζ 比 | 3483 | ≈3500 | PASS |
| GPU-only 低估 (10.2−5.6)/10.2 / (120−86.4)/120 | 45.1 / 28% | 45 / 28% | PASS |
| TCO 10.2×1.3×8760×0.1 / 14.3×1.3×… / 120×1.15×… | 11615.8 / 16285.3 / 120888 | 11616 / 16285 / 120888 | PASS |
| 每 GPU 年电费 /8 /8 /72 | 1452 / 2035.7 / 1679 | 1452 / 2036 / 1679 | PASS |
| 设施侧每 GPU 1.667×1.15 / 1.7875×1.3 | 1.917 / 2.324 | 同 | PASS |
| 租金 2×8760=17520；电费占比 1452/1679/2036 ÷17520 | 8.29 / 9.58 / 11.62% | 8.3 / 9.6 / 11.6% | PASS |
| 域聚合 8×900/8×1.8/72×1.8 → TB/s | 7.2 / 14.4 / 129.6 | 同 | PASS |
| switch chip 72×100 GB/s=7.2 TB/s；57.6 Tbps÷8=7.2 | 7.2 / 7.2 | 同 | PASS |
| tray 2×7.2=14.4；9×14.4=129.6；坑 9×7.2=64.8 | 14.4 / 129.6 / 64.8 | 同 | PASS |
| Hopper 4×64=256 端口、占用 8×18=144；B200 2×72=144 | 256 / 144 / 144 | 同 | PASS |
| NVL72 端口 9×2×72=1296=72×18 | 1296 | 同 | PASS |
| 450/50=9× / 900/50=18× / 节点出口 8×50=400 | 9 / 18 / 400 | 同 | PASS |
| min(64,40)=40；40/64=62.5%；C2C 900/128≈7 | 40 / 62.5 / 7.03 | 同 | PASS |
| NVL72 tray 18×4=72 GPU / 18×2=36 Grace | 72 / 36 | 同 | PASS |

### 11.2 bottleneck / roofline / 通信读数（全 PASS）

| 项 | 独立重算 | 章内值 | 判定 |
| --- | --- | --- | --- |
| t_busy 330×12 / t_wall 329×18+12 | 3960 / 5934 μs | 同 | PASS |
| duty 3960/5934 / gap 1974 / gap÷wall | 0.6673 / 1974 / 33.27% | 0.667 / 1974 / 33.3% | PASS |
| gap÷busy（非 hook 口径，对照） | 49.85% | —（hook 用 gap/wall） | PASS |
| graph wall 3960+5 / 加速 5934/3965 / duty after | 3965 / 1.4966 / 0.9987 | 3965 / 1.50 / 0.999 | PASS |
| 命题 1 BW_e2e=3352×0.85×1×0.667 | 1900.9（56.71%） | 1901（56.7%） | PASS |
| graph 后 BW_e2e（duty 0.9987） | 2846.3（84.9%） | 2846（84.9%） | PASS |
| 命题 1 恒等 W/t_wall == BW·u_B·η·duty | 闭合 | 同 | PASS |
| u_B 0.85→1 加速 1/0.85 | 1.176 | 1.18 | PASS |
| strideCopy 有用 16777216×4×2=128 MiB | 128 | 同 | PASS |
| η=1/8 → V 1024 MiB / t 320.33 μs；s=1 → 128 MiB / 40.04 μs | 1024 / 320.33 / 40.04 | 同 | PASS |
| 加速=流量比 8；有用 BW 419→3352（12.5%→100%） | 8 / 419 / 3352 | 同 | PASS |
| u_B=0.9 → 44.49 μs / 加速 7.20 | 44.49 / 7.20 | 同 | PASS |
| Sectors/Req stride16=32 / excessive 32−4=28 | 32 / 28 | 同 | PASS |
| latency C=16×1×4=64 / L_req 0.434×478.8≈208 | 64 / 207.7 | 64 / 208 | PASS |
| F_ach=2.5×3352×64/208 / loss 1−64/208 / u_B 64/208 | 2.578 / 69.23% / 30.77% | 2.58 / 69.2% / 30.8% | PASS |
| ILP4 C=256≥208 → F_ach 8.38 / 加速 3.25 / occ 0.81→52 warp | 8.38 / 3.248 / 51.84 | 8.38 / 3.25 / 52 | PASS |
| roof I=2.5 → 8.375 TFLOPS | 8.375 | 同 | PASS |
| MFU 总 FLOPs 30×7.125e8+300×7.616e5 | 2.160e10 | 2.16e10 | PASS |
| 总 DRAM 字节（I_elem=1/12） | 11.29 GB | 11.29 | PASS |
| 时间占比 3000/3960 / 960/3960 | 75.76 / 24.24% | 75.8 / 24.2% | PASS |
| 忙碌加权 u_F / MFU_step=duty×u_F | 0.551% / 0.368% | 同 | PASS |
| ridge 989.5e12/3352e9 / A 点 u_F 700/989.5 | 295.2 / 70.7% | ≈295 / 70.7% | PASS |
| nccl algbw 1e9/4.75e-3 / busbw ×1.75 | 210.53 / 368.42 | 210.5 / 368.4 | PASS |
| busbw vs 370.8 差 / 精确 t 4.7475 ms → 368.6 差 | 0.64% / 0.59% | ≈0.6% / ≈0.6% | PASS |
| 带宽项 4.72 ms / 延迟项 0.028 ms / n64 因子 1.96875 / 370.8=18×20.6 | 4.72 / 0.028 / 1.96875 / 370.8 | 同 | PASS |
| fusion decode 3.15MB→0.94 μs +15=15.94 / launch÷IO≈16 / prefill 201.33MB→60.06 | 0.94 / 15.94 / 15.96 / 60.06 | 同 | PASS |
| BP transpose 140.2/12.8=10.95 / 199.4/140.2=1.42 | 10.95 / 1.42 | 同 | PASS |
| decode 理想步时 4.79/2 | 2.395 | 2.40 | PASS |
| 命题 2 恒等 V_before/V_after == t_before/t_after | 闭合 | 同 | PASS |

### 11.3 ECC / thermal / replay / 链路降级（全 PASS）

| 项 | 独立重算 | 章内值 | 判定 |
| --- | --- | --- | --- |
| row remap 512/64 | 8 | 8× | PASS |
| FP32 66.9/2 / 频率 1980/2 | 33.45 TF / 990 MHz | 同 | PASS |
| 80% 峰值 66.9×0.8 / throttle 后 ÷2 | 53.52 / 26.76 | 53.5 / 26.8 | PASS |
| 式(1) φ=0.10→1−0.05 / φ=0.50→1−0.25 | 0.95 / 0.75 | 95% / 75% | PASS |
| E[K]=Σ j·p^(j−1)(1−p) 数值（p=1e-6/0.01/0.1/0.5） | 1/(1−p) 逐点 | 同 | PASS |
| goodput=BW/E[K]=BW(1−p)；损失 p=1e-6/0.01/0.1 | 0.0001 / 1 / 10% | 同 | PASS |
| 断 1 link 17×25=425 / 425÷450 / 断 2 16×25=400 / ÷450 | 425 / 94.44% / 400 / 88.89% | 425 / 94.4% / 400 / 88.9% | PASS |
| 双向 850/900 / 单 link 损失 1/18 | 94.44% / 5.56% | 94.4% / ≈5.6% | PASS |
| straggler recap n16 M1GB：每 rank 1.875 GB / 50→37.5 / 25→75 / max | 1.875 / 37.5 / 75 / 75 | 同 | PASS |
| DCGM r4 2.25h=135 min / r4÷r2 135/10.5 | 135 / 12.86 | 135 / ≈13 | PASS |
| MTBF recap 54×86400/466 | 10012 s=2.78 h | ≈1.0e4 s / 2.8 h | PASS |
| f_max≥1−0.8 | 0.2 | 0.2 | PASS |
| RMA 条件②可在总 remap=1<8 触发（析取） | True | 同 | PASS |

**重推导结论：207/207 PASS，无 FAIL。三章数值脊柱全部独立闭合。**

---

## 2. Recap 边界判定（逐源核对，PASS）

| 章 | recap 源 | 边界核对 | 判定 |
| --- | --- | --- | --- |
| 11.2 | models 2.4 §6（I/ridge/吞吐上限，经 gpu 2.4） | §1 边界声明「I/吞吐上限/ridge 推导归 models 2.4 §6（经 2.4 recap，不重推）」；§4 三区判定 recap 2.4 §5 式(6)(7)，增量=把 NCU 读数接进判定；命题 1 四因子分解 + 命题 2 带宽饱和加速 + MFU_step 式(6) 为本章新增对账（非重推已交付公式） | PASS |
| 11.2 | gpu 1.4 / 3.4 / 4.1-4.3 / 5.4 / 6.1 / 7.2 / 10.2 | stall 四分类 recap 1.4 §7、occupancy recap 3.4、strideCopy recap 4.1 §6、fusion recap 5.4 §2、busbw recap 10.2 §8、370.8 recap 7.2 式(2)——均引结论 + 数值锚，不重推机制 | PASS |
| 11.3 | gpu 2.1 §4（ECC SECDED/Sideband） | §3 开篇「Hamming 界与教学实例化 12.5% 存储开销归 2.1 §4 命题 2，本章不重推」；**锚 2.1 §4（L124-147）非 1.2**（1.2 无 ECC 节，已核）；增量=生产计数/remap/RMA | PASS |
| 11.3 | inference 8.5 / training 7.4 / 7.3 | §2「本章只 recap 结论 + 指引」：MTBF 式(2)、最优间隔式(4)、u_max=0.8→f_max≥0.2 命题 4、rerouting 命题 5、EP failover doc L265、herd 419/58.7%（含表文差异口径注）——均 recap 不重推 | PASS |
| 11.3 | gpu 9.3 §6-7（IB CRC/retry/降级/straggler） | §6 开篇「IB 侧机制与带宽后果已 9.3 §6 交付…本章 recap 不重推；增量 NVLink 计数器语义 + replay 开销 + link 降级」；straggler 37.5→75 ms 引 9.3 §7 算例 | PASS |
| 11.3 | gpu 1.1 §3（峰值频率账）/ 2.4（roofline） | 命题 2 用 1.1 乘积式 recap + roofline 两支讨论（增量=thermal 诊断）；**注：引「1.1 式(2)」应为式(1)，见 S3-2** | PASS（锚偏移 S3） |
| 11.1 | gpu 7.1 / 7.2（NVLink 域 / PCIe/NUMA） | §1 边界 + §3「三个数都是 7.1 式(4) 域聚合恒等式直接代入，recap 不重推」；PCIe min 账 recap 7.2 式(4)；端口账为 W6 新披露 + 算术（非重推 7.1 乘法链） | PASS |
| 11.1 | inference 9.1（成本账） | TCO 只计电力成分，完整 USD/M tokens 模型「归 inference 9.1 §7，指引不重推」；租金 2 USD/GPU-hour 引该章教学假设口径 | PASS |

**越界检查**：三章无 serving 调度/引擎编排展开（归 inference）、无 K8S/容器编排展开（归 cluster）；硬件机制/命令片段/transport 级计数器不触发。PASS。

---

## 3. MIG 几何 / XID 码 / 新 seam / R-R_th 专项判定

### 3.1 MIG 几何 vs MIG User Guide（primary cache，PASS）

- **切片定义逐字**（mig-concepts.txt L47/49/56/61）：memory slice「roughly one eighth…including both capacity and bandwidth」+「including the corresponding memory controllers and cache」；SM slice「roughly one seventh…when configured in MIG mode」；GI「provides memory QoS」；CI「The CIs share memory and engines」。章 §6 引文逐字对上。
- **H100 Table 13**（mig-profiles.txt L756+）：1g.10gb(1/8,1/7,L2 1/8,CE1,7) / 1g.20gb(**1/4 mem,1/7,L2 1/8**,CE1,4) / 2g.20gb(2/8,2/7,L2 2/8,CE2,3) / 3g.40gb(4/8,3/7,L2 4/8,CE3,2) / 4g.40gb(4/8,4/7,L2 4/8,CE4,1) / 7g.80gb(Full,7/7,Full,CE8,1；7 NVDEC/7 JPEG/1 OFA)。章表逐行对上；**1g.20gb L2=1/8 例外**正确。
- **B200 Table 8**（L305+）：1g.23gb(1/8,1/7,L2 1/8,CE2,7) / 1g.45gb(**2/8,1/7,L2 2/8**,CE2,4) / 2g.45gb(2/8,2/7,L2 2/8,CE3,3) / 3g.90gb(4/8,3/7,L2 4/8,CE6,2) / 4g.90gb(4/8,4/7,L2 4/8,CE8,1) / 7g.180gb(Full,Full,CE16,1；1 NVDEC/1 JPEG/1 OFA)。章表逐行对上；**1g.45gb L2=2/8** 与 H100 1g.20gb L2=1/8 的跨代不对称——章坑注「L2 随 memory slice 是方向、具体值以官方表为准、禁跨代外推」**正确钉住**。
- **A30 Table 16**（L1044+）：1g.6gb(1/4,1/4,L2 1/4,CE1,**4**)——A30=4 例外正确。GB200 7g.186gb CE16 正确。
- **Table 3**（streams Unlimited / MPS 48 / MIG 7；SM 隔离 No/百分比/Yes）逐字对上。
- **部署语义**（mig-start.txt L134-145/233）：Hopper+「enabling MIG mode no longer requires a GPU reset」+ 不跨 reboot 持久（无 InfoROM bit、driver reload disable）vs Ampere 需 reset + InfoROM 持久；GI/CI 全代际不跨 reboot；monitoring agent 阻 reset。章坑注逐字对上。P2P R570+（mig-profiles.txt L34）、R510 4+3 限制注（mig-concepts.txt L157）对上。
- **B200 SM 148 [B]**：bw-micro.txt L107 逐字「feature 148 SMs spread across eight GPCs…four L2 cache partitions…eight HBM3e memory stacks」；inside-bw.txt L32「up to 160 SMs…varies by SKU」；论文 1.09× 残留（L527「we have not directly measured this decomposition」）——章口径注（148 标 [B] 非 datasheet、官方不印、1.09× 不自洽）全部忠实。

### 3.2 XID 码语义 vs Release 620 catalog + archive（PASS）

- **xid-key-codes.txt 逐码**：13 GR_EXCEPTION/RESTART_APP；31 page fault/RESTART_APP/WORKFLOW_XID_31（「Typically…application-level bugs, but can also be driver bugs or hardware bugs」）；43 stopped processing/IGNORE/REPORT_ISSUE（「The GPU remains in a healthy state」）；45 preemptive cleanup/WORKFLOW_XID_45；48 DBE/WORKFLOW_XID_DBE（SRAM threshold flag）；63 remapping event/RESET_GPU_DEFERRED（「Only Xid 63, 64, or 177…for a given ECC event」）；64 remapping failure/RESET_GPU；79 fallen off bus/CONTACT_SUPPORT；94 contained/RESTART_APP；95 uncontained/RESET_GPU；149 NETIR/WORKFLOW_NVLINK5_ERR；154 recovery action 五档逐字。章 §4.1 表 + §4.2 三判别 + §4.4 全部对上。
- **74 代际漂移**：catalog Applies「A100:YES H-x00:YES **B-x00:NO GB-x00:NO Rubin:NO**」；Desc「Unused」（章注「Desc 列标 Unused，语义见 Trigger」正确）；同情错误逐字「if a GPU fails, another GPU…may report an Xid 74 simply because the link went down」。149 Applies「A100:NO H-x00:NO B-x00:YES GB-x00:YES Rubin:YES」。章 §4.3 硬注 + 坑注正确。
- **SXid**：xid.txt L78「SXid messages apply only to Hopper and earlier generation GPUs」+ Fabric Manager User Guide。章「SXid 仅 Hopper 及更早、NVL72 走 Fabric Manager」正确。
- **archive 七列归因**（HTML 解析逐列）：31=HW+Driver+UserApp；43=Driver+UserApp（**非 HW**）；48=HW only；63=HW+Driver+FB；64=HW+Driver；74=HW+Driver+Bus；79=HW+Driver+SysMem+Bus+Thermal；94/95=HW+Driver+FB。章表归因列逐行对上。

### 3.3 新 seam（K8S = cluster 指针不展开；MIG/vGPU/SR-IOV 硬件 = gpu，PASS）

- 11.1 §1 边界 + §7 边界指针段 + 总结行：K8S device plugin / GPU 调度策略 / 多机集群编排「归 cluster 维度…纯文字指针、不展开」；MIG/vGPU/SR-IOV 硬件分区与 hypervisor 接入归本章。零展开。
- 11.2 §1 边界：「K8S 编排归 cluster」「serving 调度与引擎编排归 inference」。零展开。
- 11.3 §1 边界 + §2 图 + §8.2 尾段：集群级故障编排（节点 drain / 备机池 / 作业重调度 / K8S device plugin 健康上报）「归 cluster 维度…编排层不展开（新 seam）」；本章止于单卡/子系统诊断与处置。零展开。

### 3.4 11.2 R vs R_th 混场声明（PASS）

§1 L19「**混场声明**（spec §4.3，本章同用 roofline 与 occupancy）：R=ridge，R_th=registers/thread，两者勿混」。R 用于 §4/§6（ridge 295），R_th 用于 §5 表（K2 96→25%）。声明到位、两场不混式。

---

## 4. Anchor 抽查表（primary cache 逐字，~85 锚）

| 锚 | 来源 cache | 章内引文/数值 | 判定 |
| --- | --- | --- | --- |
| MIG memory/SM slice 1/8、1/7 + GI QoS + CI share | mig-concepts.txt L47/49/56/61 | 11.1 §6 逐字 | PASS |
| H100 Table 13 / B200 Table 8 / A30 Table 16 全表 | mig-profiles.txt | 11.1 §6 两表 + A30=4 | PASS |
| H100 1g.20gb L2=1/8 vs B200 1g.45gb L2=2/8 | mig-profiles.txt L756/305 | 11.1 §6 坑注 | PASS |
| Hopper+ enable 免 reset 不持久 / Ampere reset+InfoROM / GI-CI 不跨 reboot | mig-start.txt L134-145/233 | 11.1 §6 坑注 | PASS |
| MIG CUDA 透明「programming model remains unchanged」 | mig-concepts.txt L164 | 11.1 §6 | PASS |
| XID 31/43/48/63/64/74/79/94/95/149/154 + 74 Applies NO + SXid | xid-key-codes.txt / xid.txt L78 | 11.3 §4 | PASS |
| archive 七列归因（43 非 HW、48 HW only） | xid-archive.html 解析 | 11.3 §4.1 表 | PASS |
| row remap 512 vs 64 / GPU reset 生效 / 终身持久 / 无软件空洞 / CE 可被 UE 替换 | ecc-row-remapping.txt L37-58 | 11.3 §3.3 | PASS |
| RMA 三条件析取 + 条件②「less than eight」+ SRAM parity>4/SECDED>2 | ecc-rma-policy…txt L38-47 | 11.3 §3.4 命题 1 | PASS |
| RAS Repair「After two row re-mappings…next…trigger repair」+ XID 160 + reboot | ecc-hbm-channel-repair.txt L37-38 | 11.3 §3.3 | PASS |
| RAS fallback「Otherwise…continue remapping until…flag」 | ecc-rma-policy…txt L37 | 11.3 §3.4「repair 失败才继续 remap」 | PASS（源有披露，前 reviewer 疑点销） |
| containment「limited to the applications that encounter the error」+ 多数 UE 免 reset | ecc-error-containment.txt L37-39 | 11.3 §3.2 | PASS |
| HW Thermal「factor of 2 or more」/ SW Thermal / SW Power Cap / HW Power Brake / Sync Boost「slowest GPU」 | smi.txt L944-967 | 11.3 §5.2/5.3 逐字 | PASS |
| 温度三档 Max Operating/Slowdown/Shutdown | smi.txt L1279-1296 | 11.3 §5.2 | PASS |
| Drain and Reset「Asserted only for UCE row remaps」+「regain its full capacity」 | smi.txt L844-846 | 11.3 §7.3 | PASS |
| GPU Recovery Action 8 值（含 Recover IMEX Domain） | smi.txt L802-803 | 11.3 §7.3 | PASS |
| ECC volatile/aggregate + SBE 纠正无 corruption/DBE 检不纠 | smi.txt L1114-1127 | 11.3 §3.1 | PASS |
| **NVLink5 计数器版本** | smi.txt L3880-3889（v560 section） | 11.3「v550+」 | **FAIL→S2-1（应 v560+）** |
| DGX H100 10.2kW/8U/130.45kg/6×3.3kW 4+2/≥3 PSU boot/1105 CFM@80% | dgx-h100-ug.txt L72-143 | 11.1 §2/§4 | PASS |
| DGX B200 14.3kW/10U/142.4kg/5+1/1550 CFM/2×NVLink5 14.4TB/s/8570 | dgx-b200.txt L45-94/167 | 11.1 §2/§4 | PASS |
| GB200 rack≈120kW/8 shelves×6×5.5kW N+N/33kW shelf/busbar 50-51V DC/cold plate+leak | dgx-gb200-hw.txt L106-117 | 11.1 §2/§4 | PASS |
| switch tray「2x NVLink NVSwitches…72 NVLink ports」+ 57.6 Tbps full duplex | dgx-gb200-hw.txt L94-96 | 11.1 §3 | PASS |
| compute tray 18×1RU(2 Grace+4 GPU)/9 switch/2 TOR/4×CX7+2×BF3/4×3.84TB E1.S RAID0 | dgx-gb200-hw.txt L37-75 | 11.1 §2 | PASS |
| TDP GB200 1200W/HGX B200 1000W（blackwell-ds）/ GB300 1400W/B300 1100W（bw-ds-new） | blackwell-ds.txt L527 / bw-ds-new.txt L281 | 11.1 §4 | PASS |
| Hopper wp 18 links 900GB/s / 64 ports / PCIe 128(64/方向) / FP32 66.9@1980 / ECC p.38 SECDED+Sideband+spare rows | hopper-wp.txt L1574/1597/386/694/1311/1269-1282 | 11.1 §5/§6、11.3 §3/§5.2/§6.3 | PASS |
| Nsight SOL「achieved percentage…theoretical maximum」/ roofline 距离 / baseline / 层级 L1-L2-DRAM / Hierarchical{Double,Half,Single,Tensor} | ncu-pg.txt L135/1781/1782/1785/136-148 | 11.2 §3/§6 | PASS |
| NCU 锁频/flush cache/串行化 + nsys 三「does not」+ gpu__time_duration semaphore | ncu-pg.txt L852-870 | 11.2 §7 表 | PASS |
| 「generally > 20 µs」/ 异步引擎干扰 / thermal override / MIG clock-control 失败 / host-timer 禁 / software-patch 不采时长 | ncu-pg.txt L845/843/1635/1638/867/165 | 11.2 §7 | PASS |
| 「Keep capture settings identical across runs」 | ncu-pg.txt L1614 | 11.2 §7 | PASS |
| NVTX「projects these ranges onto the GPU timeline」/ --cuda-trace-all-apis overhead / graph node overhead / --gpu-metrics-frequency 10000 / 「CUDA data buffer saves…overhead」 | nsys.txt L66/378/362/552/338 | 11.2 §2/§7 | PASS |
| DCGM diag r1-r4 时长 + 成员矩阵（nvbandwidth/nccl_tests r3、memtest/pulse r4）+ 不替代 Field Diag/不服务 RMA | dcgm-dcgm-diagnostics.txt L106-160 | 11.2 §9 / 11.3 §7.1 | PASS |
| DCGM PROF 1001-1012 语义 + 设备 100/140/150/156 + GPM 94 + BOARD_POWER_WATTS 名称漂移 | dcgm-fields.txt L896-925/163-203 | 11.2 §9 | PASS |
| DCGM ECC 310-313/345/346 + remap 392-396 + NVLink CRC 400-405/409、replay 420-425/429、recovery 430-435/439 + PCIe 202 | dcgm-fields.txt L318-473/256 | 11.3 §6.1/§7.2 | PASS |
| DCGM 三能力 Field Value Watches/Active Health Checks/Policy Notification | dcgm-feature-overview.txt L65/75/95 | 11.2 §9 / 11.3 §7.2 | PASS |
| nccl-tests busbw 因子 AR 2(n-1)/n、RS/AG/A2A (n-1)/n、Bcast/Reduce 1 + 「hardware bottleneck」 | gpu 10.2 §8（W5 已锚 PERFORMANCE.md） | 11.2 §8 | PASS |
| vGPU time-sliced 共享 engines / FRL B45-Q-A60 / MIG-backed 仅 AI Enterprise / sriov-manage 唯一 / 每 VF 一 mdev / virtfn0→41:00.4 | vgpu-ug.txt L220/229/218/1130/1140/1158 | 11.1 §7 | PASS |
| B200 microbench 148 SMs 逐字 + 1.09× 未实测残留 | bw-micro.txt L107/527 | 11.1 §6 | PASS |
| Inside Blackwell「up to 160 SMs…varies by SKU」+ RAS engine「AI-powered…thousands of parameters」 | inside-bw.txt L32/98 | 11.1 §6 / 11.3 §8.2 | PASS |
| Summit 24576 GPU 180× / 731× 理论比 | gpu 10.2 §8（W5 已锚 NCCL 2.4 blog） | 11.2 §8 | PASS |
| BP V100 transpose 12.8/140.2/199.4 + tiling 119.9/144.4/195.5 | gpu 4.2 §3 / 4.3 §5（W2 已锚 BP） | 11.2 §5 表 | PASS |

---

## 5. Findings

### S2（必修）

**S2-1 · 11.3 NVLink5 错误计数器版本误述 v550→应 v560（3 处 + refs/research 同源）**
- 位置：`03-faults-reliability.md:185`（§6.1「nvidia-smi nvlink -e 可查错误计数器（v550+ 含 NVLink5 计数器）」）、`:240`（§7.2 表「nvlink -e（v550+ 含 NVLink5）」）、`:331`（参考「nvlink -e（v550+ NVLink5）」）；同源 `gpu/references.md:199`（nvidia-smi-docs「nvlink -e（v550+ NVLink5）」）、`docs/research/gpu-w6-sources.md` §4.3。
- 断言：章称 NVLink5 错误计数器自 v550+ 可用。
- 证据：nvidia-smi docs changelog（smi.txt）「Changes between nvidia-smi **v560** Update and v555」节内逐字「Added support in 'nvidia-smi nvlink -e' to display NVLink5 error counters」；v570 节再「Added new counters for Nvlink5」；**v550 节（v550 vs v545）无任何 NVLink 条目**。故 NVLink5 计数器显示支持自 **v560** 起，非 v550。
- 影响：版本漂移硬注是本章核心纪律；v550/v555 部署者会被误导以为有 NVLink5 计数器。局部事实错误。
- 建议修正：三处「v550+」→「v560+」（可注「v570 增补计数器」）；同步 references.md nvidia-smi-docs 条与 research §4.3。

**S2-2 · 11.2 §6 点标口径注跨章坐标恒等误述（T3 fix 引入）**
- 位置：`02-profiling-tuning.md:231`（§6「点标口径注：latency-bound 点在 2.4 §7 标 C，本章改标 D，避与并发符号 C（§1 声明）冲突；**坐标与损失读数不变**」）。
- 断言：注称本章 D 点即 2.4 §7 的 C 点、仅改标、坐标与损失不变。
- 证据：gpu 2.4 §7 的 C 点 = **(I=2.5, F=4.19 TFLOPS)、u=50%、loss 50%**（104-sector 并发减半案例，2.4 §5 表第二行）；本章 D 点 = **(2.5, 2.58)、loss 69.2%**（64-sector 案例，2.4 §5 表第三行，§4 实例与总结均正确引「2.4 §5」）。两点坐标/损失**不同**，D 并非 2.4 §7 C 的改标。
- 影响：recap 忠实性注对源章数值误述；读者按注回查 2.4 §7 会见到 (2.5,4.19)/50% 与本章 (2.5,2.58)/69.2% 不符。本章自身数值全对（207/207 PASS），仅 provenance 注错。注：若按「改标不改本章数字」的善意读法可通，但句面跨章恒等读法不成立——按局部事实错误判 S2，writer 可附证据走反驳通道。
- 建议修正：注改为「本章 latency-bound 点取 2.4 §5 表 64-sector 案例（O 0.25/ILP 1），标 D 避与并发符号 C 冲突；2.4 §7 的 C 点是 104-sector 半并发案例 (2.5,4.19)/loss 50%，两者不同」。

### S3（minor，T7 累计清零；★=本审新发现，余与前任务 reviewer T7 清单重合、独立确认）

- **S3-1★ · 11.2:7 kernel 数引文「inference 3.4 / 3.3 §5」**：330-kernel 口径实在 inference 3.4 §3（launch 账）；inference 3.3 §5 是 Fused MoE，无 kernel 数。建议改「recap inference 3.4 §3」（若意引 gpu 3.3 §5 须显式注 gpu）。本章 L63/练习 1 已正确单引 inference 3.4。
- **S3-2 · 11.3:19/153「1.1 式(2)」应为式(1)**：乘积形 F_peak=n_SM·c_prec·f_prec 是 gpu 1.1 **式(1)**（命题 1 峰值乘积式，L65）；式(2) 是反解 c_prec 形。§5.2 命题 2 证明「1.1 式(2) 重排为乘积形」与 §1 符号声明「recap 1.1 式(2)」均宜直引式(1)。（Task 4 reviewer 已列 T7。）
- **S3-3★ · 11.2「四面板/四组面板」vs 5 项**：hook「四面板（SpeedOfLight / roofline / stall reason / memory chart / occupancy）」括注 5 项；§3「四组面板」后 5 个 bullet；总结「NCU 四面板交叉读」列 5 角色。计数不一致。建议改「五面板」或显式把 SOL+roofline 并为一组（SpeedOfLight 族）。（全章 4 处「四面板/四组面板」。）
- **S3-4★ · 11.2:19 符号声明措辞**：「裸字母 n/N/B 本章不占用…不用裸 T」与 §8 用 n=8（rank）、§5 用 B=64/T=4096（fusion recap）字面冲突；实际用法均合全书保留语义。建议改「n/N/B/T 仅按全书保留语义使用（n=rank §8、B=batch/T=序列长 §5 fusion recap），本章不另作占用」。
- **S3-5 · references.md citedBy 6 条缺 11.2**：hopper-whitepaper / nvidia-datasheet / cuda-best-practices / nccl / nccl-tree-blog / hopper-microbench 的 citedBy 未含 11.2（11.2 参考段已引）。T7 回写。（Task 3 reviewer 已列。）
- **S3-6 · references.md:144 gb200-nvl72「57.6 Tbps full duplex per 1U tray（=7.2 TB/s）」混 tray/chip**：57.6 Tbps=7.2 TB/s 是**单 chip**；tray=2 chip=115.2 Tbps=14.4 TB/s。章内 §3 坑注正确，entry 措辞需紧。（Task 2 reviewer 已列。）
- **S3-7 · 11.1:314 练习 4(d) 校验锚缺 B200 11.6%**：§8 L288 有 8.3/9.6/11.6% 三值，练习锚只列 H100 8.3%/NVL72 9.6%。补 B200 11.6%。（Task 2 reviewer 已列。）
- **S3-8 · 11.3:253-266 ASCII 健康检查框右边框 misaligned**（CJK 双宽未对齐）：视觉项，图子 pass（T6）替换。（Task 4 reviewer 已列。）
- **S3-9 · 11.3:334 inside-blackwell-blog 参考条「TEE-I/O + inline NVLink protection（§8.2）」over-claim**：§8.2 body 仅引 RAS engine 逐字，未用 TEE-I/O；entry 的 §8.2 标签宜收窄到 RAS engine。（Task 4 reviewer 已列。）

---

## 6. 机械门与规范扫描（独立）

| 检查 | 11.1 | 11.2 | 11.3 |
| --- | --- | --- | --- |
| lint-chapter.sh | OK | OK | OK |
| lint-math-boundary.py | 0 违规 | 0 违规 | 0 违规 |
| display 9 禁形 raw grep（\, \! \; \: \{ \} \| \% \#） | 0 | 0 | 0 |
| prose \lt（公式外） | 0 | 0（\lt 均在 $...$ 内，合法） | 0 |
| bare <= / >= | 0 | 0 | 0 |
| process-leak（T1-7/task/writer/evaluator/S1-3/W0-7/research pass/builder/fix轮/波次/sweep/lint） | 0（「brief」命中均为 Blackwell brief 文档名） | 0 | 0 |
| 数学白名单外命令 | 0 | 0 | 0 |
| CJK（正文汉字） | 5820（≥2500） | 6499（≥3500） | 6546（≥3500） |
| 模板（metadata/总结/练习/参考） | 齐 | 齐 | 齐 |
| 练习分级 | [10][20][20][30] 非递减 | [10][20][30][30] | [10][20][30][30] |
| ≥1 [10]/[20] / ≤1 [40] / 校验锚 / ≥1 证命题 | ✓/✓/✓/练3 证命题2 | ✓/✓/✓/练3 证命题1 | ✓/✓/✓/练3 证命题3 |
| 总结回指列缺失行 | 0 | 0 | 0 |
| ASCII 图 ≥1 | ✓（形态/拓扑/MIG 切片） | ✓（timeline/决策树/roofline） | ✓（恢复链/处置树/健康四层） |
| R2-1 节首 crux/设问/数值 | 8/8 节 | 9/9 节 | 8/8 节 |
| R2-2 章脊口径（11.1 多 SKU 对比表） | ✓ | ✓ | ✓ |
| R2-3 公式后 300 字全数字实例 | ✓ | ✓ | ✓ |
| R2-4/5 命题-证明 + 路线句 | 命题1/2 ✓ | 命题1/2 ✓ | 命题1/2/3 ✓ |
| R2-10/11 提点/坑注块 | 3 坑 | 1 提点+2 坑 | 1 提点+3 坑 |
| 证据分级（[C] 官方规格 / [B] microbench / 教学参照） | ✓（148 标 [B] 4 处、datasheet 不印 3 处） | ✓ | ✓ |

**保留符号 §4.3**：11.1 声明 n_SM/n_GPU/s_SM/s_mem/f_SM/f_mem/P_rack/P_IT/n_shelf/n_PSU/P_PSU/PUE/c_elec/h_year/C_elec/ζ/Δθ/Q_air，N/B/T/M/α/β/I/R 不占用（公式扫描确认仅声明行出现）；11.3 n=rank/M=消息/局部 p/φ/k/X，N/B/T 不占用；11.2 混场 R/R_th 声明到位（措辞 S3-4）。GEMM M/N/K 未误用、裸 T 表 tile=0、bytes 不占字母。

---

## 7. 修复清单（T7）

- **S2-1**：11.3 三处 v550+→v560+（+ references.md nvidia-smi-docs + research §4.3 同源）。
- **S2-2**：11.2:231 点标口径注改写（D=2.4 §5 64-sector 案例，区别于 2.4 §7 C=104-sector）。
- **S3-1…S3-9**：逐条修复/接受/驳回（S3-2/5/6/7/8/9 与前任务 T7 清单合并去重）。
- 含 S2 章节（11.2/11.3）修复后 scoped re-review；结论记 gpu/README.md 审查记录区。

**审查结论**：P11 三章数值脊柱与一手事实锚全部独立闭合，recap/seam/符号纪律干净；2 项 S2 为局部版本/provenance 误述（非数值错误），9 项 S3 为锚偏移/措辞/回写。无 S1。S2 必修后即可达 commit gate（图 QA 另由 T6 出）。
