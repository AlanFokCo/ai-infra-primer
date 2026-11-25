# gpu W3 Part 6 对抗审查（T6）— 低精度与注意力 kernel（6.1–6.4）

日期：2026-11-25。审查人：fresh-context adversarial evaluator（不继承写作上下文，未派子代理）。
对象：`gpu/06-low-precision-attention-kernels/01-low-precision-formats.md`（6.1）、`02-quantization-kernels.md`（6.2）、`03-flashattention-kernel.md`（6.3【重】）、`04-attention-variant-kernels.md`（6.4）。
量规：STYLE_GUIDE.md + gpu spec §4（§4.3 符号 / §4.2 recap 边界 / §4.6 R2）+ §5 Part 6 大纲 + §6.3 强制重推导清单 + rubric（A/B/C）+ R2 style sheet（R2-1…R2-12 + 习题四档）。
方法：全部数值账独立 Python 重算（171 项）；代码锚 fresh tarball file:line 一手核（FA b038b497 / Marlin 1f25790b / DeepGEMM 5081f851 / FlashMLA main 395cb3af + ba89a346 双锚 / CUTLASS 4.9.0 / FlashInfer 0.7.6 contents API raw）；一手来源 WebFetch/curl（arXiv export API 8 ID、FA4/FA2/FA3/Marlin 摘要逐字、PTX ISA、OCP MX v1.0 PDF、Hopper whitepaper PDF、HGX H100/H200 + Blackwell datasheet PDF、CUDA Best Practices Table 5、V3/V4.1-Flash config hf-mirror）。

## 结论

**S1 = 0，S2 = 3，S3 = 7。** 强制重推导 6.3（tile 预算 + GPU 精确 HBM 流量）**PASS**；recap 边界 **PASS**；MLA 两代几何 **PASS**；NVFP4 UE4M3 **PASS**。四个 T5 重点订正项（m16n8k16 行=lane/4、seesaw 图对 deep-dive L27-38、FA4 根 README L82、UseSchedulerBarrier L352-355）全部核实正确落地。lint 4/4 OK；数学白名单/rescape/flanking/禁令词/process-leak 零真实违例；CJK 篇幅达标。3 条 S2 均为局部事实/口径错误（非致命公式或虚构），须修复；7 条 S3 下波清零。

## findings 表

| # | severity | 位置 | 断言（原文） | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| F1 | **S2** | 6.4 §5 瓶颈搬家账 + §7 决策表 DSA 行 + 总结表 | 「FP8 528 B 生产口径 10.3 MiB/步…indexer…bf16 口径 1.25 GiB/步 = 主路的 64 倍（MXFP4 indexer 省 3.76 倍后仍 17 倍）」；决策表「indexer 带宽主导（64×主路）」 | 独立重算：1.25 GiB ÷ 20 MiB（bf16 反事实主路）= 64×；但 ÷ 10.3 MiB（FP8 生产主路）= **124×**；MXFP4 indexer（340 MiB）÷ FP8 主路 = **33×**（非 17×）。64×/17× 仅在 inference 3.1 的 bf16-bf16 口径成立。本节自框「代入 V4.1 生产格式」并加粗 FP8 生产主路数，比值却沿用 bf16 口径，决策表/总结无条件复述「64×主路」 | 给比值钉口径：「64×/17× 为 bf16 教学反事实口径（inference 3.1 原账）；FP8 生产主路 10.3 MiB 下 indexer bf16 为 124×、生产 MXFP4 indexer 约 33×」；或决策表改「indexer 带宽主导（生产口径约 33×主路）」 |
| F2 | **S2** | 6.4 §7 决策表「MLA V4.1 decode」行 | bound 列「compute（TC 利用率杠杆）」 | 全章未给 V4.1 强度推导。按本章所引 V4.1-Flash config（hf-mirror 一手核：`num_attention_heads`=64）+ FP8 KV 528 B：I = 2×64×(512+512)/528 = **248 FLOP/byte < B200 BF16 ridge 2250/7.7 = 292** → memory-bound，与「compute」矛盾。bound 对口径敏感（FP4 KV 288 B → I=455>292；H_q=128 → I=496），本章均未钉。违反本章 §7 自述「先算强度再选 kernel」纪律（V3 行有 I=241.8 推导，V4.1 行无） | 补 V4.1 强度算式并声明 H_q/KV 格式/MMA 精度口径；或 bound 列改「口径依赖：FP8 KV+H_q=64 贴 ridge 偏 memory，FP4 KV 或 H_q=128 → compute」，勿无条件断 compute |
| F3 | **S2** | 6.1 §5 逐格式吞吐表 INT8 行 B200 dense 列 | 「支持，峰值未列」 | Blackwell datasheet（本章 B200 列声明来源，一手 PDF 核）「Individual Blackwell GPU Specifications」HGX B200 列明列 **INT8 Tensor Core 9 POPS（sparse）→ 4500 TOPS dense**（layout 提取 L507 逐字）。峰值已披露，「未列」误 | B200 INT8 列改「4500 TOPS」（datasheet 9 POPS sparse ÷ 2），对 BF16 比 2（已对）；「未列」口径仅适用 INT4（Hopper/Blackwell 确未列独立 INT4 峰值） |
| F4 | S3 | 6.3 §4 grid 段 | 「32×1×32 = 1024 个 CTA，对 132 个 SM 铺 7.8 波」 | 1024/132 = 7.76 是 **CTA/SM 比**；§3 已立 2 blocks/SM 驻留，按驻留计波数 ≈ 1024/(132×2) = 3.9 波。「7.8 波」混 CTA/SM 与 wave | 改「每 SM 摊约 7.8 个 CTA（2 驻留，约 3.9 波）」或加口径注 |
| F5 | S3 | 6.3 §3 register 绑定段 | 「这解释了 repo 注释里被注掉的候选：…128×128、256×64 的 8-warp 配置全部注释掉」 | fresh tarball `flash_fwd_launch_template.h` L270-271 实际注掉的是 **128×128 8w 与 128×64 8w**（非 256×64；「256 x 64」仅出现在 L269 注释文字里）。且 8-warp（n_th=256）配置寄存器可行（128×128 8w ≈ 188 regs），被注掉因实测慢 28%（注释自述），非 register 预算否决；「这解释了」把预算排除（4-warp 128×128=348 regs，练习 1c）与性能注掉混为一谈 | 引实际注掉行（128×128、128×64 8-warp），注 repo 注释「256 x 64」措辞与代码不符；澄清 8-warp 候选寄存器可行、因实测慢被注掉，预算排除的是 4-warp 大 tile |
| F6 | S3 | 6.3 §6 naive 对照 | 「加 QKV/O 读写 134.22 MB」 | 134.22 MB = 4×T×H_q×d×2，K/V 按 **H_q=32 展开形**计（非 H_kv=8 存储形 83.89 MB）；未注口径。结论稳健（展开形省 6.62×，广播形省 6.48×） | 加「（K/V 按展开形 H_q 口径计）」一句 |
| F7 | S3 | 6.4 §2 式 (1) | $\sqrt{d_h + d_{\mathrm{rope}}}$ 中 $d_h$ | §1 符号声明列 d_latent/d_rope/d_qk/d_v，未声明 $d_h$（= nope head 维 128，models 记号）；式 (1) 直接用，章内未定义 | 符号声明补 $d_h$ = qk_nope_head_dim = 128（models 4.1 记号），或式 (1) 译为本章已声明符号 |
| F8 | S3 | 6.2 §3 切分与归约段 + CALL_IF 段 | 「global_reduce（L514）…（L511-513 注释『reduce directly in the output buffer to maximize L2 cache utilization』）」；「L799 注释『For compilation speed…』」 | fresh tarball：该引句实在 **L515**（L511-513 是另一段「…serially in L2 cache」注释）；compilation-speed 注释起 **L798**（L798-799 跨行，引句文字在 L798） | 引句锚改 L515；compilation-speed 锚改 L798-799 |
| F9 | S3 | 6.2 §4 lop3 走读 | 「注释逐字（L116-117）『…the compiler does not seem to automatically recognize it』」 | fresh tarball L116-117 原文末为「…automatically recognize it **in all cases**.」；「逐字」引句漏「in all cases」 | 补全引句或标省略号 |
| F10 | S3 | 6.1 §5 表 FP16/BF16 行 V100 列 | 「FP16/BF16 \| ✓（V100）」 | V100（sm_70）有 FP16 Tensor Core（mma.m8n8k4），**无 BF16**（BF16 mma 需 sm_80）；合并行 ✓ 掩盖 V100 不支持 BF16 | 拆 FP16/BF16 两行，或 V100 列注「FP16 ✓ / BF16 —」 |

## verification log（强制）

### A. 强制重推导 6.3（spec §6.3：tile↔shared/register 预算 + GPU 精确 HBM 流量）— **PASS**

独立 Python（`/tmp/gpu-w3-t6-verify/rederive.py`，171 项全 PASS）：

**tile 预算（命题 1，K/V 分计）**
- 式 (4) 单缓冲：(128+2×64)×128×2 = 65536 B = 64 KB ✓（占 227 KB 的 28% ✓）；K/V 系数 2 对 `kernel_traits.h` L108 `kSmemKVSize = size(SmemLayoutKV)*2*sizeof(Element)` 逐字 ✓。
- 式 (5) multi-stage：(128+2×2×128)×128×2 = 163840 B = 160 KB ✓（FA3 n_stage=2，`flash_fwd_launch_template.h` L47 kStages=2 ✓）。
- 式 (6) register：acc_o 128×128/128=128、acc_s 128×64/128=64、P 128×64/(2×128)=32、+R_aux 28 = **252** ✓（距 255 余 3 ✓）。
- 式 (7) 上界：shared 项 232448/256−128 = **780** ✓；register 项 (255−28)×128/(128+96) = **129.7** ✓ → T_M=128（64 倍数）✓；下一档 192 → 364 regs（超限 43% ✓）；T_N=128 → 348 regs（练习 1c ✓）。
- occupancy：252×32=8064→8192（256 粒度 ✓）、block 4×8192=32768、⌊65536/32768⌋=2 ✓；shared ⌊233472/66560⌋=3 ✓；warp 64/4=16 ✓；min=2 blocks/SM、O=8/64=0.125 ✓。
- FA3 register 再分配：每 WG acc_o 64+acc_s 64+P 32=160 ✓；block 128×24+256×240=**64512**≤65536 ✓（`flash_fwd_kernel_sm90.h` L82-83 Load=24/Mma=240 对 2WG+TMA ✓）；1 block/SM ✓。
- MLA 预告：64×512 fp32 = 32768 = R_file/2 ✓。

**GPU 精确 HBM 流量（命题 2，causal/split-KV/GQA）**
- n_load = Σ_{i=0..31} 2(i+1) = **1056** ✓；每列 KV 1056×2×64×128×2 = 34.60 MB ✓。
- σ=1：KV 1107.30 + Q/O/LSE 67.63 = **1174.93 MB** ✓，I=117（memory）✓；σ=4：276.82+67.63 = **344.46 MB** ✓，I=399（compute）✓。
- FLOPs 4×(T²/2)×d×H_q = **137.44 GFLOP** ✓；算力地板 138.9 μs ✓；带宽 102.8/350.5 μs ✓。
- naive：S/P 2147.48 + QKV/O 134.22 = **2281.70 MB** ✓，I=60 ✓，省 6.62× ✓（QKV/O 口径见 F6）。
- IO 定理对账：单头非 causal KV 2T²d/T_M = 3.36e7 ✓；定理 T²d²/M_SRAM = 2.37e6 ✓；比 **14.2** ✓ = 2×908/128（K/V 两份 × tile 比）✓；M_SRAM/d=908、908/128=7.1 ✓。
- split-KV η：M_red 2×14×32×516 = 451.5 KiB ✓；KV 16 MiB ✓；η=**2.76%** ✓；对 inference 3.1 满窗 0.086% 大 32× ✓（η∝1/T）。
- PackGQA causal prefill：Σ_{i=0..127}⌈(i+1)/2⌉ = **4160** ✓ 对 4×1056=4224 ✓，比 0.985 ✓。
- SFU/TC：16384 exp2 / 16 = 1024 cycles = 559.6 ns @1.83 GHz ✓；TC 2×4.19 MFLOP/7.496 TFLOPS = 1119 ns ✓；比 50% ✓；串行天花板 66.7% ✓ < 报告 74.8% ✓；B200 峰比 2.274× → 113.7% ✓。
- 硬件常数一手核：Hopper WP Table 3 BF16 989.4/FP8 1978.9/TF32 494.7/FP64TC 66.9/INT8 1978.9 TOPS ✓；**TC boost 1830 MHz**（WP 逐字，1.83 GHz 来源 ✓）；132 SM/50 MB L2/228 KB shared/65536 regs/255 per-thread/64 warps ✓；Best Practices Table 5 CC9.0 FP32 128 / SFU 16 results/cycle/SM ✓。

### B. recap 边界（spec §4.2 强制清单）— **PASS**

- 6.3 §2：online softmax 式 (1)(2) = models 4.1 式 (15)(16) 逐字 recap，「归纳证明见 models 4.1 §5.2，本章不复述」✓；IO 定理式 (3) recap「不重推」✓；§6 对账是「精确账代入定理假设恢复阶」= spec  mandated 增量（实际 traffic 对账理论），非重证定理证明 ✓。FA2/FA3 策略 recap inference 3.1 §3 ✓。
- 6.2 §1：GPTQ/AWQ/SmoothQuant 算法侧 recap inference 3.2 §3-5「不重推」✓；W4A16 理想加速/临界 batch recap inference 3.3 式 (1)/命题 1 ✓；grouped padding 账 recap inference 3.3 命题 2「不重推」✓。增量（命题 1 融合流量、命题 2 mixed-input 强度、命题 3 magic-number、指令账、调度器）均 GPU 侧新内容 ✓。
- 6.4 §1：MLA absorption recap models 4.1 §4 式 (11)-(14)「等价性证明…本章不复述」✓；split-KV/cascade 正确性 recap inference 3.1 命题 1/2 ✓；式 (4) base-2 形是 inference 3.1 式 (3) 翻译「正确性证明归 inference 3.1 命题 1，本章不重证」✓；DSA 选择链/期望界 recap inference 3.1 §7 ✓；块表 recap inference 2.1 式 (3) ✓。
- 6.1：值公式 recap training 1.3 式 (1) ✓；MX/NVFP4 格式事实+字节账 recap inference 3.2 §2/式 (3) ✓；增量（范围/ULP 三约定闭式、累加界、scale 编码结构、粒度-精度权衡、吞吐对账）新内容 ✓。
- **未发现重推导已交付证明（无 S2 越界）**；serving 调度/引擎编排内容均接口级 recap + 指针，无越界 ✓。

### C. MLA 两代几何（spec §5 6.4 / inference W4 errata）— **PASS**

- V3：576 = 512 latent + 64 rope ✓（ba89a34 `config.h` L10-11 HEAD_DIM_K=576/HEAD_DIM_V=512 一手 ✓；V3 config hf-mirror：kv_lora_rank=512/qk_rope_head_dim=64/qk_nope=128/v_head_dim=128/128 heads/61 层 ✓）。
- V4.1：512 = 448 latent + 64 rope ✓（V4.1-Flash config hf-mirror 一手：text_config.head_dim=512、qk_rope_head_dim=64、**无 kv_lora_rank 字段**、40 层 ✓；main README L140-143 head_dim=512 + 「64 RoPE dimensions are quantized as well, so there is no bfloat16 part」逐字 ✓）。
- 两代区分无混：6.4 §2 表 + 坑注「把 512+64=576 写成 V4.1 是两代混淆的典型错」✓；§7 表/总结均「512=448+64」✓；6.3 §7 用 64×512（V3 HEAD_DIM_V）正确 ✓。
- KV 格式账：FP8 528=512+16 ✓、FP4 288=256+32 ✓（README L142-143 逐字 ✓）；528 对 1152 省 2.18×、288 省 4× ✓。
- seesaw 图对 ba89a34 deep-dive L27-38 步骤 0-11 **逐行一致** ✓（含 O_R 累积因子 scale0·scale1、跨 WG p0 换基准、rescale 方向译注）；L21「32,768/65,536…one output matrix per SM」逐字 ✓；L42 seesaw 逐字 ✓；L54 9 TMA 逐字 ✓；traits.h L82-83 barriers_K0[9] ✓、splitkv L47 EVICT_FIRST ✓、get_AorC_row_idx L20-23 公式 ✓。

### D. NVFP4 scale 类型 — **PASS**

- 6.1 §4：NVFP4 = E2M1 元素 / block16 / **UE4M3** scale（7-bit 无符号、8-bit 容器 MSB 补零）+ FP32 global ✓；坑注「把 UE4M3 读成 E4M3」✓。
- PTX ISA 一手：ue4m3「7-bit unsigned floating-point…4 bits exponent…3 bits mantissa…NaN limited to 0x7f…MSB bit padded with zero」逐字 ✓；kind::mxf4nvf4 表「.e2m1 / .ue4m3 / .scale_vec::4X」✓；block16 = scale_vec::4X 别名 ✓。
- CUTLASS `exmy_base.h` L918：UE4M3 = FpBitRepresentation<uint8_t,8,4,3,CANONICAL_ONLY,**false**>（末参 IsSigned=false ✓，L398 模板签名核）；E2M1 NONE（L938）✓；NanInfEncoding IEEE_754/CANONICAL_ONLY/NONE（L64-73）✓；exponent_bias=2^(E-1)−1 ✓。
- 6.4 §2 区分 FlashMLA per-16 scale 用 e4m3（README 口径）与 NVFP4 规范 UE4M3 ✓（不混）。
- OCP MX v1.0 PDF 一手：MXFP8/MXFP6/MXFP4/MXINT8 均 block 32 + E8M0 ✓；E8M0 bias 127、范围 −127..127、NaN=11111111₂=0xFF、无 Inf ✓；MXFP4 元素 E2M1 ✓；NVFP4 非 OCP MX ✓。

### E. 代码锚 spot-check（fresh tarball file:line）— 实质全对，2 处行号偏移（F8/F9）

**四个 T5 重点项全核实正确：**
1. **m16n8k16 行=lane/4**：CUTLASS `mma_traits_sm80.hpp` L53 `SM80_16x8_Row = ((4,8),(2,2)):((32,1),(16,8))` ✓；col-major 解码 row=lane/4(+8v1)、col=2(lane%4)(+v0) ✓；6.3 §4「行号=lane/4…lane%4 相同者共享列不是行」正确 ✓；quad Allreduce<4> 两条 shfl_xor offset 1/2（utils.h L111-128）✓。
2. **seesaw 图 vs deep-dive L27-38**：逐行一致（见 C）✓。
3. **FA4 根 README L82**：根 `README.md` L82「written in CuTeDSL and optimized for Hopper and Blackwell GPUs (e.g. H100, B200)」逐字 ✓；L86 pip install / L96 flash_attn_func ✓；`flash_attn/cute/README.md` 33 行、L3 同义 ✓；6.3 §7「勿把根 README 引句归到它」正确 ✓。
4. **UseSchedulerBarrier L352-355**：`mainloop_fwd_sm90_tma_gmma_ws.hpp` L351 注释「tuned for speed…don't affect correctness」+ L352-355 计算式逐字 ✓；**IntraWGOverlap 是 bool 模板参数 L33**、tile_size.h L9-54 派生、launch L40/L45 std::get<3> ✓；6.3 §5「这是 UseSchedulerBarrier 的条件，不是 IntraWGOverlap 的定义」正确 ✓。

**其余锚（抽样全核）**：FA2 flash_fwd_kernel.h L57/L92-96/L140-142/L157/L175/L307-311/L338/L352-353/L374/L376/L387/L442/L446/L597-598 ✓；softmax.h L24/L39/L67/L79-81/L86/L137/L157/L158/L160/L163-164/L170/L172/L179-183 ✓；kernel_traits.h L15/L40-41/L63-64/L74-77/L93-95/L107-109/L127 ✓；FA3 roles L220-222 / regs L82-83 / PipelineTmaAsync L283 / EVICT_LAST L753 / named_barrier L50-58 / barrier 环 L921-937 / mma_init L955 / fwd_step L1199 / warpgroup_wait<1> L1214 / permute L1185/L1189 / tile_size L31-33/L35/L51 / tile_scheduler L108/L261/L605 ✓；FA4 blackwell_helpers.py L8/L14-30 ✓。Marlin L98/L111/L118-126/L131-152/L136-137/L138/L143/L147/L155-159/L443/L447/L463-465/L466/L514/L590-591/L61/L75-76/L705-707/L755-761/L801 ✓（822 行 ✓）；README L7-8/L13-14/L19/L22/L23/L41-43/L65 ✓。DeepGEMM sm90 L56/L251/L254/L272/L286-289/L300-302/L313/L329-345/L340/L349-354 ✓；sm100 L44-45/L51-52/L68/L77/L84-85/L94-95 ✓；jit.hpp L12/L14-22/L26-29 ✓；scheduler L152/L195/L300-302 ✓；README L3/L67-68/L78/L84-86/L172 ✓；scaling-factor-format L29/L33/L89-91 ✓；mqa_logits kernel 存在 ✓。FlashMLA main README L3/L7/L25/L41/L79/L138/L140-143 ✓、csrc 仅 sm100/smxx ✓；ba89a34 README L81/params.h L6-9/config.h L7-11/traits.h L24/L39/L44/L82-83 ✓。FlashInfer cascade.cuh L42/L45/L53-72/L56-57/L90/L125/L147/L224/L378/L583 ✓；scheduler.cuh L74-98/L86/L96/L101-123 ✓；decode.py L1338/L2179/L3237 ✓；cascade.py L42/L226/L401/L537 ✓；jit-core.py L186-202 ✓。CUTLASS sm100_pipeline.hpp L538 PipelineTmaUmmaAsync ✓、version 4.9.0 ✓。
**偏移**：global_reduce 引句 L515（章引 L511-513，F8）；compilation-speed 注释 L798（章引 L799，F8）；lop3 引句漏「in all cases」（F9）。

### F. 一手来源（WebFetch/curl）

- arXiv export API 8 ID 批量 200（totalResults=8）：2205.14135/2307.08691/2407.08608/**2603.05451**/2408.11743/2501.01005/2412.19437/2309.06180 ID↔标题↔第一作者全对 ✓（FA4 = Ted Zadouri 官方，非第三方 2609.04105 ✓）。
- FA4 摘要逐字：「tensor core throughput doubles while other functional units (shared memory bandwidth, exponential units) scale more slowly or remain unchanged」✓、「software-emulated exponential」✓、fully asynchronous MMA + larger tile ✓、CuTe-DSL ✓。
- FA2 摘要：25-40% FA1 / ~2× / A100 50-73% / 225 TFLOPs/s 72% MFU ✓。FA3 摘要：FA2 H100 35% / 1.5-2.0× / FP16 740（75%）/ FP8 ~1.2 PFLOPs/s / 误差低 2.6× ✓。Marlin 摘要：16-32 近 4× / 64-128 仍显著 / vLLM e2e 2.8× ✓。
- PTX ISA：wgmma m64nNk16(f16/bf16)/m64nNk32(e4m3/e5m2)/m64nNk8(tf32) ✓；ue4m3/ue8m0/e2m1 格式节 ✓；kind::mxf4nvf4 ✓。
- Hopper WP PDF / HGX H100-H200 datasheet PDF / Blackwell datasheet PDF：H100 SXM BF16 989.5/FP8 1979/TF32 494.5 dense（HGX 8-GPU 32/16/8 PF sparse ÷8÷2 ✓）；HGX B200 BF16 2250/FP8 4500/FP4 9000/TF32 1100/FP64 37/FP32 75 dense、180 GB/7.7 TB/s ✓、**INT8 9 POPS sparse 已列**（F3）；dense= sparse÷2 脚注逐字 ✓。
- V3/V4.1-Flash config（hf-mirror）：见 C ✓。

### G. 规范（STYLE_GUIDE + R2 + rubric + 模板 + 练习 + 数学白名单 + rescape + process-leak）

- **lint**：4/4 LINT OK ✓。
- **数学白名单/rescape/flanking**：Python 精检（剔除 code span 与 `$`...`$` 保护形）4 章 **0 真实违例** ✓；无 operatorname/align/boxed/裸 `<`字母/裸 `*` display ✓；`` $`...`$ `` 保护形用于含 `_`/`*` 内联（6.2 bytes_weight 等）合规 ✓。
- **rubric A 禁令词 / AI 味**：0 命中（「完美共享」是定义性技术口径 σ=g，非营销形容词，不违规）✓。
- **process-leak**：grep 命中均为 INT4/INT8 子串误匹配 + 「fresh tarball 核实」版本锚约定（全书先例，inference 同用），无 writer/T1-T5/research pass/evaluator 泄漏 ✓。
- **R2-1…R2-12**：节首设问/数值开场 ✓；章脊口径锁定（6.4 多 SKU 表）✓；公式后 300 字内全数字实例 ✓；命题-证明骨架（6.1×3/6.2×3/6.3×2/6.4×1，留作练习者练习均存在）✓；推导路线句 ✓；定义配例+边界反例 ✓；口径注 ✓；术语判断句 ✓；提点/坑块引用 ✓；总结表回指列 ✓。
- **练习四档**：6.1 [10]/[20]/[20]/[30]、6.2 [10]/[20]/[30]/[30]、6.3 [10]/[20]/[30]/[30]、6.4 [10]/[20]/[30]/[30]，难度非递减 ✓、≥1 题 [10]/[20] ✓、无裸 [40] ✓、每计算/代码题带校验锚 ✓、≥1 题证正文命题 ✓、代码题先预测后验证 ✓。6.2 §4 教学 Python 独立运行通过（16 nibble 精确 + 打包路径对整数路径）✓。
- **符号 §4.3**：M/N/K GEMM 维章首声明（6.2）✓；T_M/T_N/T_K 禁裸 T ✓；M_SRAM 章首声明=S_sh/bytes ✓；d_head/d_latent/d_rope 翻译 ✓；R/R_th 混场声明（6.3）✓；bytes 不占字母 ✓；K_top 翻译声明（6.4）✓。例外：6.4 式 (1) d_h 未声明（F7）。
- **模板**：标题/metadata/总结/练习/参考齐全 ✓；每章 ≥1 ASCII 图（6.1 位布局+块布局 / 6.2 物化融合+mainloop / 6.3 五段循环 / 6.4 seesaw）✓。
- **篇幅**（正文 CJK 剔代码，沿 W1/W2 计数口径）：6.1=4224 / 6.2=3809 / 6.3=6200（≥5500 软目标达）/ 6.4=4102，标准章均 ≥3500 ✓。
- **证据分级**：[A]/[C] 官方规格披露·一手权威/[W]/教学参照+官方实现 标注齐全 ✓；跨维引用纯文字、活链接仅 ../references.md ✓；引用 key 均在 gpu/references.md ✓。

## 汇总

- **S1 = 0**（无虚构数值/错误公式/不存在 repo/张冠李戴；171 项数值重算全 PASS，arXiv 8 ID 全对，代码锚实质全对）。
- **S2 = 3**（F1 DSA 比值口径混用、F2 V4.1 decode bound 无推导且与所引 config 矛盾、F3 B200 INT8 峰值误标未列）——须修复。
- **S3 = 7**（F4 波数口径、F5 注掉候选锚+解释混、F6 naive K/V 口径、F7 d_h 未定义、F8 行号偏移×2、F9 引句漏字、F10 V100 BF16 合并行）——下波清零。
- 强制重推导 6.3 **PASS**；recap 边界 **PASS**；MLA 两代几何 **PASS**；NVFP4 UE4M3 **PASS**。
- 处置建议：F1/F2/F3 修复后 6.1/6.4 复审；F4-F10 随下波 sweep 清零。无 S1，不触发单章强制复审门（S2 修复即可）。
