# gpu W1 research pass — P1/P2 硬件一手核实证据记录（2026-11-25）

Task 1：P1（GPU 微架构）+ P2（内存层级）硬件一手来源核实 + `gpu/references.md` P1/P2 扩充 + blelloch URL 修。本文件记录核实通道、逐源证据（URL/引句）、⟨verify⟩ 销项/残留、版本漂移注、网络 fallback。writer（T2-T5）引用白名单 = `gpu/references.md` P1/P2 条目。

## 0. 结果摘要

- **parse-check（pure ESM）**：`groups 11 entries 64`（≥59 地板过；W0 59 → +5）。
- **per-Part**：P1=12 / P2=4 / P3=1 / P4=0 / P5=9 / P6=10 / P7=6 / P8=6 / P9=6 / P10=4 / P11=6 = 64。
- **证据等级分布**：[C]=29 / 教学参照=16 / [A]=10 / [B]=8 / [W]=1（W0 [C]27/[B]5 → 本轮 +2 [C] tuning guides、+3 [B] 微基准）。
- **污染检查全过**：64 数据行均恰 7 字段；无 escaped `\|`；无重复 key；日期均 YYYY-MM-DD；等级单值无混合档。
- **新增 5 条**：`hopper-tuning-guide` / `blackwell-tuning-guide`（[C] 官方规格）、`hopper-microbench`（2402.13499）/ `blackwell-microbench`（2507.10789）/ `b200-power-bench`（2610.41919）（[B] 微基准）。
- **更新 10 条**：hopper-whitepaper / blackwell-whitepaper / nvidia-datasheet / cuda-programming-guide / ptx-isa / cuda-best-practices / nsight-compute / hopper-blog / jedec-hbm3（P1/P2）+ blelloch-scan（P5 URL 修）。核实日期统一 2026-11-25。
- **⟨verify⟩ W1 销项**：Hopper/Blackwell whitepaper + datasheet 精确数值、SM 内部资源数、HBM 栈几何逐项、单发射、MMA 几何、CUDA Programming Guide URL（子页漂移）。**残留**：B200 SM 数官方未披露（W3/W6）、JEDEC 精确 URL（403，同 PCI-SIG W4）。

## 1. 核实通道与网络 fallback（本轮实际使用）

本轮网络高度不稳定（沿 inference W1「NVIDIA 网页反爬」记录，但表现不同）：

- **dam-cdn.nvd.orangelogic.com（NVIDIA 官方 DAM CDN）= 本轮主通道，稳定 200**。`resources.nvidia.com` 落地页（200，HTML 内嵌 AssetLink）→ `https://dam-cdn.nvd.orangelogic.com/AssetLink/<hash>.pdf` 直接可下。**W0「Hopper whitepaper PDF 镜像不可达」残留由此销项**：Hopper whitepaper V1.04 PDF（7.88 MB）一手取得。
- `docs.nvidia.com`：首轮 TLS handshake 挂起（0 字节超时），**重试后恢复 200**（Best Practices / PTX ISA / Programming Guide 子页 / Nsight / tuning guides / MIG 均取得）。网络波动须重试，勿凭单次失败判不可达。
- `developer.nvidia.com`：本轮可达（Hopper blog 454 KB 200）；inference W1 曾记 body 挂起，波动。
- `resources.nvidia.com`：首轮挂起、重试 200（三落地页）。
- **WebFetch 全域名失败**（含 example.com，「Unable to verify if domain is safe」域名验证服务不可达）→ 本轮全部改用 curl。
- `archive.org` / `web.archive.org`：80/443 均连接超时 → Wayback 通道不可用。
- `images.nvidia.com` / `www.nvidia.com`：挂起 → 改用 dam-cdn PDF（datasheet 一手优于产品页）。
- `developer.download.nvidia.com`：301 → `.cn` 镜像；redist JSON 多版本 404 → 未用（docs.nvidia.com 恢复后不需要）。
- **JEDEC / rambus / news.skhynix.com 搜索端点 403 bot 拦截** → 成员官方产品页互证（Micron / Samsung / product.skhynix.com 200；news.skhynix.com WP REST search 200）。
- **arXiv export API 波动**：首轮 200、中途 429/超时 → 单 ID + 重试可过（2402.13499 / 2507.10789 / 2610.41919 / 2501.12084 标题作者全中）。
- **CMU PDF 截断**：Ble90.pdf 首取 221184 B / Content-Length 1349154（截断、xref 损坏）；Ble93.pdf 首取 188416 B / Content-Length 311166（截断）→ `--retry` 重取全量后 pdftotext 首页标题逐字可读。**大 PDF 须核对 Content-Length 防截断**（沿 inference W5 大 tarball 截断坑）。
- PDF 文本：`/opt/homebrew/bin/pdftotext -layout`；框图（无文本层）用 `pdfimages -png` 提取后视觉读图（SM 框图 / GH100 框图）。

## 2. 逐源证据（URL × 引句 × 结论）

### 2.1 Hopper whitepaper V1.04（[C] 一手，销 W0 镜像残留）

URL：`https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf`（落地页 `resources.nvidia.com/en-us-hopper-architecture`）。封面：「Includes final GPU / memory clocks and final TFLOPS performance specs. V1.04」。

- **Table 3（H100 SXM5 / PCIe）**：SMs 132 / 114；TPC 66 / 57；FP32 Cores/SM 128；FP64 Cores/SM 64；INT32/SM 64；Tensor Cores/SM 4（528/GPU）。
  - **逐精度 boost clock**：「GPU Boost Clock for FP8, FP16, BF16, TF32 Tensor Core Ops」= **1830 MHz**（SXM5）/1620（PCIe）；「for FP64 Tensor Core Ops, FP32 and FP64 non-Tensor Core Ops」= **1980 MHz** /1755。→ **频率依精度类而异**，1.1 芯片级 FLOPS 反推须用对应精度 clock（BF16 989.4 = 132×4096×1.83 GHz；FP32 66.9 = 132×128×2×1.98 GHz）。
  - Peak BF16/FP16 TC = **989.4/1978.9**（dense/sparse）；FP8 = 1978.9/3957.8；TF32 = 494.7/989.4；FP64 TC 66.9；FP32 66.9；FP64 non-TC 33.5。**whitepaper 989.4 vs datasheet 989.5 为舍入漂移**（132×4096×1.83 = 989.46）。
  - Memory Interface **5120-bit HBM3**；Memory Size 80 GB；**Memory Data Rate 2619 MHz DDR**；**Memory Bandwidth 3352 GB/sec**；L2 50 MB；Shared/SM up to **228 KB**；Register File/SM **256 KB**（/GPU 33792 KB）；TDP 700 W；80 B 晶体管 / 814 mm² / TSMC 4N。
  - 栈数：「80 GB HBM3, **5 HBM3 stacks**, 10 512-bit Memory Controllers」（SXM5）；full GH100「6 HBM3 or HBM2e stacks, 12 512-bit Memory Controllers」。
- **Table 4（CC 9.0）**：Max Warps/SM **64**；Max Threads/SM 2048；Max Thread Blocks/SM **32**；Max 32-bit Registers/SM **65536**；Max Registers/Thread **255**；Max Thread Block Size 1024；Shared/SM up to 228 KB。
- **Fig 6（GH100 full，视觉读图）**：8 GPC × 9 TPC × 2 SM = **144 SM**；**L2 两分区**（左右各一 L2 Cache 块、中间双向箭头）；6 HBM3 栈 / 12 控制器；18 NVLink links。→ 2.2「H100 50 MB 两分区」销项。
- **Fig 7（SM 框图，视觉读图）**：4 processing block，每 block = L0 I-cache + **1 Warp Scheduler (32 thread/clk)** + **1 Dispatch Unit (32 thread/clk)** + **Register File (16,384 × 32-bit)** + 32 FP32 + 16 INT32 + 16 FP64 + **1 TENSOR CORE 4th GENERATION** + 8 LD/ST + SFU；SM 级 **Tensor Memory Accelerator** + **256 KB L1 Data Cache / Shared Memory** + 4 Tex。
  - **单发射互证**：每 scheduler 恰 1 Dispatch Unit、32 thread/clk = 1 warp 指令/cycle → 4×32×1 = **128 FP32 results/cycle/SM** = Best Practices Table 5 CC 9.0 FP32 128。双发射（Kepler 术语）不成立。RF 16384×4 block = 65536 = 64K×32bit = 256 KB ✓。
- L2 prose：「Using a **partitioned crossbar structure**, the L2 cache localizes and caches data for memory accesses from SMs in GPCs directly connected to the partition.」
- HBM prose：「80 GB (five stacks) … over 3 TB/sec」（prose 未更新，与 Table 3 final 3352 内部不一致 → 漂移注）。
- **ECC Memory Resiliency 节（p.38，TOC 在列；T4 review fix round 1 回补，T1 漏录）**：逐字「The H100 HBM3/2e memory subsystems support Single-Error Correcting Double-Error Detecting (SECDED) Error Correction Code (ECC) to protect data.」+「H100 supports “Sideband ECC” for its HBM3/2e memories, where a small memory region, separate from the main HBM memory, is used for ECC bits (which is in contrast to “Inline ECC”, where a portion of main memory is carved out to store ECC bits).」+「Other key memory structures in H100 are also protected by SECDED ECC including the L2 cache and the L1 caches and register files inside all the SMs.」→ **SECDED 码族 + Sideband 放置为逐字披露**（2.1 §4 据此断言数据带宽不受 ECC 折扣；仅码参数/数据字宽未披露，Hamming d=64 账留教学实例化）。同页 **Memory Row Remapping** 节：每 bank 预留 spare rows，坏行 boot 时以预留好行替换。缓存文本 /tmp/t4-review-ecc/hopper-wp.txt L1267-1283（reviewer 下载同一 dam-cdn V1.04 PDF 提取）。

### 2.2 Blackwell Technical Brief（[C] 一手）

URL：`https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf`（落地页 `resources.nvidia.com/en-us-blackwell-architecture`；含 GB300 更新版）。

- 「208 billion transistors … TSMC's 4NP」；「two GPU dies … largest die possible within … reticle size … connected and unified with a single **10 terabyte-per-second (TB/s)** chip-to-chip NVIDIA High-Bandwidth Interface (**NV-HBI**), providing one fully coherent chip」。
- 「fifth-generation Tensor Core … supports new number formats such as **FP4**, including community-defined microscaling (**OCP**) formats」；Table 1 数据类型 FP64/FP32/TF32/FP16/BF16/FP8/INT8/FP6/FP4。
- 「18 fifth-generation NVLink links to provide 1.8 [TB/s]」。
- B300/B200 对照表：B200「**Up to 192 GB HBM3e | 7.7 TB/s**」、TDP 1000 W、PCIe Gen5；B300 270 GB / 1100 W / PCIe Gen6。GB200「8TB/s of HBM3e」。
- **B200 SM 数未披露**（brief / datasheet / tuning guide 均无）→ ⟨verify⟩ 残留（W3 sm_100 deviceQuery/CUTLASS 或 W6 nvidia-smi/MIG 实测）。spec 值 148 与官方 FLOPS 自洽（见 §4）。

### 2.3 Datasheet 四份（[C] 一手，dense = sparse/2）

- H100：`.../u5hh6fv4r7564i4y673484y3m20083nj.pdf`；H200：`.../b2kt2580e5b6424if22lg65223m6ghjb.pdf`；HGX H100/H200：`.../44wa8u63l12ps4ie2870214xa2exoyra.pdf`；Blackwell：`.../01dyp27s6aj4e0483cs5goo15354m438.pdf`。
- 脚注逐字：「Specifications in sparse. **Dense is one-half of the sparse spec shown.**」
- **H100 SXM**：FP64 34 / FP64 TC 67 / FP32 67 / TF32 989* / BF16 1979* / FP16 1979* / FP8 3958* / INT8 3958* TOPS / 80 GB / **3.35 TB/s** / ≤700W configurable / MIG 7×10GB。**H100 NVL**：94 GB / 3.9 TB/s / 350-400W（勿混）。
- **H200 SXM**：FP64 34 / FP64 TC 67 / FP32 67 / TF32 989 / BF16 1979 / FP8 3958 / **141 GB / 4.8 TB/s** / ≤700W；**NVL ≤600W**（沿 inference W1 fix）。
- **HGX B200**：FP4 18 PF*→**9000 dense** / FP8 9 PF*→4500 / INT8 9 POPS* / BF16 4.5 PF*→**2250** / TF32 2.2 PF*→1100 / FP32 75 / FP64 37 / **180 GB HBM3E / 7.7 TB/s** / ≤1000W / NVLink5 1.8 TB/s / PCIe Gen5 128 GB/s / MIG 7。
- **GB200（NVL72/NVL4 每 GPU）**：FP4 20 PF*→10000 / FP8 10 PF*→5000 / BF16 5 PF*→2500 / TF32 2.5 PF*→1250 / FP32 80 / FP64 40 / **186 GB / 8 TB/s** / ≤1200W。
- **SKU 口径**：HGX B200 180/7.7 ≠ GB200 186/8 ≠ raw 192（brief「Up to 192 GB」= 8×24 GB 栈容量）。2.4 对账以 inference 1.3 同表（HGX B200）为准。

### 2.4 CUDA Programming Guide（[C]，URL 子页漂移）

根 `https://docs.nvidia.com/cuda/cuda-programming-guide/` 200，但 **CUDA 13.x 拆子页**；compute capability 表在 `05-appendices/compute-capabilities.html`（**Table 33/34/35**）。

- Table 33（CC 9.0 / 10.0）：resident blocks/SM **32 / 32**；resident warps/SM **64 / 64**；threads/SM 2048；threads/block 1024；warp 32。
- Table 34：32-bit regs/SM **64 K**（全 CC）；regs/thread **255**；shared/SM **228 KB**（9.0 与 10.0）；shared/block opt-in **227 KB**；banks **32**；local/thread 512 KB；constant **64 KB**。CC 10.7 = 328 KB（B300 系，非 W1 口径）。
- Table 35：CC 9.0/10.0 **Unified Data Cache 256 KB**，SMEM carveout {0,8,16,32,64,100,132,164,196,228}。
- 注：「KB and K units … correspond to 1024 bytes (KiB) and 1024」；「shared memory allocations over 48 KB per block must use dynamic shared memory and require an explicit opt-in」。
- 子页：`02-basics/understanding-memory.html`、`04-special-topics/l2-cache-control.html`（accessPolicyWindow / hitRatio / persisting L2）。

### 2.5 Hopper / Blackwell Tuning Guide（[C] 新增）

- `https://docs.nvidia.com/cuda/hopper-tuning-guide/`：CC 9.0 warps 64、shared 228 KB（+39% vs A100 164）、TMA 1D-5D、DSMEM、clusters。
- `https://docs.nvidia.com/cuda/blackwell-tuning-guide/`：CC 10.0 warps 64 / blocks 32 / RF 64K / shared 228 KB / combined 256 KB / carveout 0-228；**「CUDA reserves 1 KB of shared memory per thread block. Hence, the B200 GPU enables a single thread block to address up to 227 KB」**（1 KB reserved + 227 opt-in 官方逐字）；static 48 KB；**GB200 L2 126 MB**；B200 HBM up to 180 GB；B200 nonportable cluster 16。

### 2.6 Best Practices Table 5 + Nsight（[C]）

- BP `.../cuda-c-best-practices-guide/`：**Table 5**「Throughput of Native Arithmetic Instructions (Number of Results per Clock Cycle per Multiprocessor)」编号维持；列漂移为 7.5/8.0/8.6-8.9/**9.0**/10.0/10.3/12.0-12.1。CC 9.0：FP32 add/mul/fma **128**、FP64 **64**、INT32 add 128、shfl 32、SFU(lg2.approx) 16、DPX 64。CC 10.0 FP32 128 / FP64 64。
- 延迟 prose：「The latency of most arithmetic instructions is **typically 4 cycles** on devices of compute capability 7.0」（FMA ~4 cycle 官方口径，CC 7.0 措辞沿用）；「When accessing uncached local or global memory, there are **hundreds of clock cycles** of memory latency」（官方只给定性，精确 cycles 见微基准）。
- Nsight ProfilingGuide `.../nsight-compute/ProfilingGuide/index.html` §2.4.7 Warp Stall Reasons：`long_scoreboard`（L1TEX local/global/surface/texture 依赖，memory-latency-bound）、`mio_throttle`（shared/MIO 队列）、`wait`、`not_selected`、`barrier`、`membar`、`lg_throttle`、`math_pipe_throttle`、`imc_miss`、`drain`、`dispatch_stall`、`branch_resolving`、`no_instructions`；Warp Scheduler States（eligible→issued 单 warp issue-slot）；memory chart DCC/IDC/IMC 常量缓存。

### 2.7 PTX ISA MMA 几何（[C] 逐字）

`.../cuda/parallel-thread-execution/`：
- `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 d, a, b, c;`（f16 同形）✓。
- `wgmma.mma_async.m64nNk16`（warpgroup，N∈{8..256}，含 m64n256k16）；m64nNk8 tf32、m64nNk32 f8f6f4；A 可 register 或 shared-desc、B shared-desc；异步。
- `tcgen05.mma / .sp / .ws`（5th-gen）；**Tensor Memory sm_100a/f：「512 columns and 128 rows per CTA, with each cell being 32-bits」= 256 KB/CTA**；`.cta_group::1 / ::2`（双 CTA/2-SM 协作）；`kind::mxf4nvf4` block scaling（NVFP4）。

### 2.8 HBM 栈几何（JEDEC 403 → 成员官方互证）

- JEDEC `jedec.org/standards-documents/docs/jesd238a` **403**（JESD238A 真实存在，W0 已确认）→ 精确 URL 残留。
- **Micron**（`micron.com/products/memory/hbm/hbm3e`）：「**HBM3E offers 16 independent channels and 32 pseudo channels.**」+「HBM3E has **1024 IO pins**」+「**24GB 8-high HBM3E is shipping with NVIDIA H200**」+ pin >9.2 Gbps / >1.2 TB/s。→ n_ch=16、w_ch=1024/16=**64 bit**、2 pseudo/ch、H200 24 GB 栈。
- **Samsung**（`semiconductor.samsung.com/dram/hbm/`）：HBM3「6.4 Gb/s per pin and up to **819 GB/s** … per stack … 16 GB and 24 GB … Organization **1024** … 12H, 8H」；HBM3E 9.2 Gbps / 1180 GB/s / 24-36 GB。
- **SK hynix**（`product.skhynix.com/products/dram/hbm.go`）：HBM3「16-24GB … ~6.4Gbps(~819GB/s) … 8,12Hi」；HBM3E 9.6 Gbps / 1.23 TB/s / 24-36GB。
- 819 GB/s = 6.4 Gbps × 1024/8 ✓ 自洽。**HBM3e = vendor speed-bin（9.2/9.6 Gbps），非独立 JEDEC 编号**。

### 2.9 微基准（[B] 新增，延迟/分区/SKU 实测支撑）

- **2402.13499**（Luo et al.，H800）Table IV 延迟 cycles：**L1 40.7 / shared 29.0 / L2 263.0 / global 478.8**；wgmma m64n128k16 dense 128 cycles / sparse SS 144；SM-to-SM DSMEM 180 cycles。→ 1.4/2.2 延迟表实测锚（spec ~20-30 shared / ~200 L2 / ~400-800 HBM 为粗估，实测 shared 29 / L2 263 / HBM 479）。扩展版 2501.12084 同作者。
- **2507.10789**（Jarmusch et al.，RTX 5080 GB203 + H100 PCIe）：「In the GH100, the L2 cache is divided into **two independent partitions** … In contrast, the GB203 employs a **monolithic L2**」→ 2.2 分区互证。
- **2610.41919**（Zand et al.，B200/B300）Table III：nvidia-smi HBM3e **179 GiB / 269 GiB** ≈ raw **192 / 288 GB** decimal（vs datasheet usable 180/270）、TDP 1000 W → 2.1 raw-vs-usable 实测注。

## 3. HBM 带宽逐项推导（BW = n_stack·n_ch·f_pin·w_ch/8）

| SKU | n_stack | n_ch | w_ch | f_pin | BW | 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| H100 SXM5 | 5（wp 逐字） | 16（Micron） | 64 bit | 5.238 Gbps（wp 2619 MHz DDR×2） | 5×16×5.238e9×64/8 = **3352 GB/s** = 3.35 TB/s | wp Table 3 + datasheet |
| H200 SXM | 6（GH100 full 6 栈 / 141≈6×24） | 16 | 64 | 6.25 Gbps（4.8 TB/s 反推） | 6×16×6.25e9×64/8 = **4.8 TB/s** | datasheet + Micron 24GB 8-high |
| B200（GB200 bin） | 8（192/24） | 16 | 64 | ~7.81 Gbps（8 TB/s 反推） | 8×16×7.81e9×64/8 = **8 TB/s** | datasheet GB200 |
| HGX B200 | 8 | 16 | 64 | ~7.52 Gbps（7.7 反推） | **7.7 TB/s** | datasheet HGX |

- **禁把 1024 bit/stack 代入含 n_ch 因子的式**（1024 = n_ch×w_ch = 16×64）；等价每栈口径 BW = n_stack·f_pin·1024/8。H100：5×5.238e9×1024/8 = 3352 GB/s ✓。
- f_pin 除 H100（wp 披露 2619 MHz DDR）外为反推值（官方不披露 per-pin），写作须注「反推」。

## 4. B200 SM 数自洽性（残留佐证，非官方披露）

spec 值 148 SM 与官方 FLOPS + CC 10.0 每 SM 吞吐自洽：FP32 75 TF = 148×128×2×1.98 GHz；FP64 37 TF = 148×64×2×1.95 GHz；BF16 2250 TF = 148×8192×1.856 GHz（Blackwell 每 SM BF16 = 2× Hopper 4096）。clock 比 1.98/1.856 ≈ 1.067 与 Hopper 1830/1980 模式一致。若 208 SM 则 clock ~1.4 GHz（不典型）。**但官方未披露 SM 数与 clock，故仍标 ⟨verify⟩ 残留 W3/W6**，writer 1.1 用 148 须注「spec 口径/反推自洽，官方未披露」。

## 5. ⟨verify⟩ W1 销项 / 残留

| 项 | 状态 | 证据 / 留波 |
| --- | --- | --- |
| Hopper whitepaper 精确数值 + PDF 可达 | **销** | dam-cdn V1.04 一手（Table 3/4 + Fig 6/7） |
| Blackwell whitepaper 精确数值 | **销**（SM 数除外） | dam-cdn Technical Brief 一手 |
| H100/H200/B200 datasheet 精确数值 + SKU | **销** | 四份 dam-cdn PDF 一手 |
| SM 内部资源（RF 64K×32 / warp 64 / blocks 32 / shared 228 / combined 256 / 227 opt-in / 1KB reserved） | **销** | wp Table 4 + PG Table 33/34/35 + tuning guide 逐字 |
| warp scheduler 单发射 | **销** | wp Fig 7（1 scheduler+1 dispatch/block）+ BP Table 5 CC9.0 FP32 128 互证 |
| HBM 栈几何逐项 → 3.35 TB/s | **销** | wp Table 3（5 栈/2619 DDR/3352）+ Micron 16ch/1024 IO |
| MMA 几何 mma.sync/wgmma/tcgen05 | **销** | PTX ISA 逐字 + TMEM 512×128×32bit |
| 指令延迟/吞吐表 | **销**（精确 cycles 挂 [B]） | BP 4-cycle/hundreds prose + Nsight stall + 2402.13499 Table IV |
| CUDA Programming Guide URL | **销**（子页漂移注） | 根 200 + compute-capabilities 子页 |
| JEDEC HBM3 精确出处 | **部分销/残留** | JESD238A 真实 + 几何成员官方互证；原文 403 → 精确 URL 留 W4（同 PCI-SIG） |
| B200 SM 数 | **残留** | 官方未披露；148 反推自洽 → W3/W6 |
| L2 两分区（H100）/ GB200 126 MB | **销** | wp Fig 6 + blackwell tuning guide |

## 6. 版本漂移注

- **Hopper blog（2022-03）3 TB/s + 「Memory Data Rate Not finalized」** vs **whitepaper V1.04 Table 3 final 3352 GB/s** vs datasheet 3.35 TB/s：早期披露→final 漂移；whitepaper prose 仍留「over 3 TB/sec」未更新（内部不一致）。
- **whitepaper BF16 989.4/1978.9** vs **datasheet 989.5/1979**：舍入漂移（989.46）。
- **Programming Guide CUDA 13.x 拆子页**：compute capability 表移 `05-appendices/compute-capabilities.html`（Table 33/34/35）；旧单页锚失效，writer 须用子页 URL。
- **BP Table 5 列集漂移**：旧 CC 5.x/6.x/7.0 列移除，现 7.5…12.x；编号维持 Table 5。
- **Blackwell brief 为含 GB300 更新版**（非 2024-03 初版）；B200 raw 192 与 datasheet usable 180 并存。
- **HBM3e 非独立 JEDEC 编号**（vendor speed-bin）。

## 7. blelloch-scan URL 修（W0 残留）

- 旧 URL `cs.cmu.edu/~guyb/papers/Ble90.pdf` **非 CMU-CS-90-190 报告**（Vector Models 书锚；本机首取截断 221184/1349154、xref 损坏）→ 移除。
- 新 URL：报告页 `https://www.cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html`（200；title「Prefix Sums and Their Applications」+「CMU-CS-90-190, November 1990」+ BibTeX `number=CMU-CS-90-190, year=1990` 逐字）+ 页链报告 PDF `https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf`（`--retry` 全量 311166 B；pdftotext 首页「Prefix Sums and Their Applications / Guy E. Blelloch / Carnegie Mellon」逐字）。
- 等级维持 [B]（老经典无 arXiv），核实日期 2026-11-25，被引 5.1。

## 8. 种入纪律

- 跨 Part 共享锚点收录首引 Part、被引跨组累加；新增 tuning guides/microbench 按首引章归 P1/P2。
- 等级单值：官方规格/官方文档 [C]「官方规格披露·一手权威」；微基准 paper 无官方 repo [B]；无混合档。
- 禁竖线：多 URL 用 ` ; `；单元格无 `|` 含 escaped。
- 被引章节按 spec §5 Sources 回写（hopper-whitepaper +1.4；其余维持 W0 意图 + 新增条目首引章）。