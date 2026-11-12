# W4 审查报告：Part 7 Diffusion & Image Generation（7.1–7.7）

审查日期：2026-11-12 | 审查人：独立对抗 evaluator（fresh context，未继承写作上下文） | 对象：`models/07-diffusion-image-generation/01`–`07` 共 7 章 + `models/references.md` Part 7 条目
方法：全文逐行阅读；一手来源自行 fetch（arXiv API https、ar5iv 全文、GitHub API、hf-mirror、CrossRef）；三个强制推导链全部独立重推；数值主张用自有脚本重算。

## 一句话总结

数学骨架（三条强制推导链 + 全部抽查公式）零错误，约定一致性零违规；但存在 2 处 S1 级引用/归因错误（rescaled CFG 论文作者张冠李戴；iDDPM cosine schedule 动机分辨率方向写反）与 2 处 S2 级局部事实错误（SANA 归一化分母、LDM f 消融边界），修复后可放行。

## Findings

| # | 级别 | 位置 | 断言 | 证据 | 建议修正 |
| --- | --- | --- | --- | --- | --- |
| F1 | S1 | 7.4 §4.4 正文 + 7.4 参考列表 + references.md `common-diffusion` 行 | rescaled CFG 出处标为「[Hang et al., 2023]」 | arXiv API（2305.08891，2026-11-12 取）：作者为 **Shanchuan Lin, Bingchen Liu, Jiashi Li, Xiao Yang**，无 Hang | 全部改为「[Lin et al., 2023]」（正文 1 处、参考列表 1 处、references.md 作者字段 1 处） |
| F2 | S1 | 7.2 §2.3 末段（line 206） | 「iDDPM 的动机正是此：linear schedule 对**高分辨率**图像过早、过快地销毁信息」 | ar5iv 2102.09672 §3.2 原文："while the linear noise schedule ... worked well for **high resolution** images, it was sub-optimal for images of resolution **64×64 and 32×32** ... the end of the forward noising process is too noisy"——方向写反；且与本章自身前文的 32×32 量化台阶分析（同段「对 32×32 图像几乎没有可恢复信息」）自相矛盾 | 改为「linear schedule 对低分辨率（64×64/32×32）图像过早、过快地销毁信息」；后半句「cosine 改善 log-likelihood」经核实成立（iDDPM Table 2：linear L_simple NLL 3.77 → cosine 3.68），可保留 |
| F3 | S2 | 7.6 §8.2 式 (15) 及符号注记 | 「（SANA 论文口径：归一化分母一并省去）」 | ar5iv 2410.10629 Eq. (1)：SANA 的 ReLU linear attention **保留归一化分母**——显式计算共享项 $(\sum_j \mathrm{ReLU}(K_j)^{\top}) \in \mathbb{R}^{d \times 1}$ 作分母，全文无「省去分母」表述 | 删去该括注，式 (15) 右式补上分母 $\phi(Q)(\sum_j \phi(K_j)^{\top})$（同为 O(Nd) 共享计算，不影响「省 16×」的账）；或改注「本书省略分母只展示结合律结构」 |
| F4 | S2 | 7.6 §1（f 消融段） | 「$f$ 太小（f4）时……收敛质量反而不升；$f$ 太大（f16/f32）时重建误差成天花板——f8 是甜点」 | ar5iv 2112.10752 Fig. 6 及正文：「LDM-{**4-16**} strike a good balance」，被点名的坏档只有 LDM-1（同预算 FID 差 38）与 LDM-32（重建限质量）；f4 不在坏侧（LDM-4 正是论文旗舰、本章 §6.3 自引其 FID 3.60，两处口径互相打架），f16 也在论文的平衡区间内 | 改为「f=1 同预算训练太慢（FID 差 38）；f=32 重建误差成天花板；f∈{4–16} 为论文平衡区间，工程落地（SD 1.x）取 f8」 |
| F5 | S3 | 7.6 §6.1 patchify 表 | DiT-XL/2 Gflops「≈ 118.7」 | ar5iv 2212.09748 Table 1：118.64（正文引作 118.6） | 改 ≈118.6 |
| F6 | S3 | 7.5 参考列表末段 | score-SDE 标「[Song et al., 2020]」，7.3/7.4 同一 key 标「[Song et al., 2021]」 | 三章同一 key（score-sde, 2011.13456）年份不一致 | 全书统一为 2021（ICLR 口径，与 7.3/7.4 一致） |
| F7 | S3 | 7.4 全章 | 「[Song et al., 2021]」同章内同时指 ddim 与 score-sde 两文 | 正文均带 arXiv ID 可区分，但 §5.6 现状表/总结行裸引用同名标签 | 现状表/总结行补 arXiv ID 或 key（ddim）/（score-sde）已部分有，补全即可 |
| F8 | S3 | 7.1 §3、§7 | 「VAE 在 **2024 年以后**的主用形态……」「**2022 年以后**的主力图像系统……」 | STYLE_GUIDE §2 禁时间线叙事；这两句是带年份锚点的现状判断，非历史故事，属边界 | 去年份锚点（「当前主用形态」「现役主力图像系统」），论点不变 |
| F9 | S3 | 7.1 §6 | 「网络评估次数的 NFE 记号 7.2 已引入」——7.2 在 7.1 之后，前向引用「已引入」 | 7.1 正文先用「每步至少一次网络前向」口语说法，术语定义跨章后指 | 就地一句定义 NFE，或改为「7.2 引入」并保留口语说法 |
| F10 | S3 | 7.4 §4.5 | 「是当前 T2I 部署的默认做法」无出处无对冲 | 对比 7.6 §7.3 同类工程惯例声明有「（无单一出处）」对冲 | 加「（工程惯例，无单一出处）」或引 guidance-distillation 原始出处 |
| F11 | S3 | 7.6 §1 | 「LDM 论文对 $f \in \{4, 8, 16, 32\}$ 的系统消融」 | 论文 Fig. 6 消融集含 f=1（LDM-1）——集合枚举漏了一项（下一句自己又在讨论 LDM-1） | 枚举改 {1, 4, 8, 16, 32} 或写「含 f=1 对照组」 |
| F12 | S3 | 7.5 文件末尾 | 6 行连续空行 | 版式残渣 | 删除 |

## Verification log

### A. 强制独立重推导（spec §6.3 清单 W4 子集）

| 推导链 | 关键中间步（本人独立复推） | verdict |
| --- | --- | --- |
| 7.2 DDPM 全链 | 前向闭式：归纳法合成方差 $\bar\alpha_t^{(s)}\beta_{t-s} + (1-\bar\alpha_t^{(s)}) = 1-\bar\alpha_t^{(s+1)}$ ✓；后验配方：精度 $\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}$ 通分关键化简 $\alpha_t(1-\bar\alpha_{t-1})+\beta_t = 1-\bar\alpha_t$ ✓、均值两系数与 DDPM 原文一致 ✓；ε-参数化：$x_t$ 系数 $=\frac{1}{\sqrt{\alpha_t}}$、ε 系数 $=-\frac{\beta_t}{\sqrt{\alpha_t(1-\bar\alpha_t)}}$ ✓；ELBO 三项：Bayes 改写 + 伸缩相消 + 塔性质收 KL ✓；高斯 KL 式 (22) ✓；SNR 权重 $\frac{\beta_t}{1-\bar\alpha_t} = 1-\frac{\mathrm{SNR}_t}{\mathrm{SNR}_{t-1}}$ 恒等 ✓；x₀/v 换算与「v-loss = ε-loss + x₀-loss」精确恒等式 ✓（数值见 B 区） | PASS |
| 7.4 CFG | Bayes 两面 → $\nabla\log p(y\mid x_t) = -(\varepsilon_\theta(c)-\varepsilon_\theta(\varnothing))/\sqrt{1-\bar\alpha_t}$ ✓；代入 classifier guidance ε 形式两因子相消 → 式 (18) ✓；$w=1$ 退化纯条件 ✓；tilt 分布 $q_w \propto p(x)p(y\mid x)^w$ 的 score 与式 (18) 逐项一致 ✓ | PASS |
| 7.4 DDIM = PF-ODE | $\lambda = \frac12\log\mathrm{SNR}$ 换元：$d\lambda/dt = -\beta/(2(1-\bar\alpha))$ ✓；半线性式 (9) ✓；积分因子 $\int(1-\bar\alpha)d\lambda = \log\sqrt{\bar\alpha}$（$1+e^{-2\lambda} = 1/\bar\alpha$ 关键步）✓；精确解式 (10) ✓；左端点冻结积分 $e^{-\lambda_t}-e^{-\lambda_s} = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}-\sqrt{(1-\bar\alpha_s)/\bar\alpha_s}}$ ✓；代回逐项 = DDIM 式 (5) —— 恒等非渐近 ✓；边缘匹配引理（$\varepsilon$ 系数平方 + $\sigma^2 = 1-\bar\alpha_{t-1}$）✓；$\sigma=\sqrt{\tilde\beta_t}$ 端点代数收回 7.2 式 (15) ✓ | PASS |
| 7.5 conditional FM 定理 | 边际场 = 条件场后验平均（式 (10)）✓；连续性验证式 (11)（散度-积分换序 + 定义认回）✓；三块展开中 $\|v_\theta\|^2$ 块边缘化相等、靶块无 $\theta$ ✓；关键步（式 (14)）：内积线性 → 后验平均提出 → Fubini → 联合期望 ✓；损失差 = 条件速度方差（练习 2 口径）✓ | PASS |
| 7.5 rectified flow + 映射 | $\alpha=1{-}t,\sigma=t$ 代入式 (17) → $u=(x-x_0)/t=\varepsilon-x_0$ ✓；式 (25) 两方向换算（$x_t = x_0+tv$ 联立）✓；SNR 匹配 $\bar\alpha=(1-t)^2/((1-t)^2+t^2)$ ✓；式 (19) 边际场在 VP 特例化简为 $-\frac{\beta}{2}(x+s_t)$ 与 7.3 PF-ODE 式 (21) 精确一致 ✓；v-prediction = $\omega_t v$（VP 约束消 $x$ 项）✓；logit-normal 密度式 (28) 与 log-SNR $\sim\mathcal{N}(-2m,4s^2)$ ✓；shift 式 (30) 赔率形式 ✓ | PASS |

### B. 数值重算（自有脚本，非书中数字）

| 抽查项 | 来源 | 结论 |
| --- | --- | --- |
| 7.2 §2.3 SNR 表全部 17 格（linear/cosine × 8+1 行） | numpy 按式 (2)(7)(8) 重算 | 全对（含 ᾱ_T=4.04e-5、cosine t=500 → −0.1 dB） |
| 量化台阶穿越步数（linear t≥979 共 22 步、cosine t≥995 共 6 步） | 重算 √ᾱ_t < 1/127.5 | 全对 |
| $w_1/w_T \approx 49$、$w_{500}$、$L_T \approx 0.06$ nats、$1/\sqrt{\bar\alpha_T}\approx157$ | 重算 | 全对（49.0、0.062、157.4） |
| 7.6 SANA attention 占比（N=4096→37%、N=1024→13%）与省 16× | 按式 (7) $2mnk$ 口径重算 | 全对（37.2%、12.9%、N/d_h=16） |
| 7.6 DiT block FLOPs $24Nd^2+4N^2d$ | 逐 QKV/proj/MLP/attn 重算 | 对 |
| 7.7 VAR 总 token 680 | $\sum_k h_k^2$ 重算 | 对（680） |
| 7.6 latent 位点表（f4/f8/f16/f32 位点与 attention 相对值） | 重算 | 全对 |
| 7.1 GAN 最优判别器 → $2\mathrm{JSD}-\log4$（练习 1） | 复推 | 对 |

### C. 一手来源核查（fetch 记录）

| 项 | 来源 | 结论 |
| --- | --- | --- |
| 30 篇被引论文 arXiv ID × 标题 × 作者 | arXiv API（https，4 批，2026-11-12） | 30/30 标题匹配；**唯一例外：2305.08891 作者为 Lin et al. 非 Hang et al.（→F1）**；GAN arXiv 标题复数 ✓ |
| DDPM：T=1000、β₁=1e-4→β_T=0.02、"reweighting leads to better sample quality"、离散化 bin ±1/255 | ar5iv 2006.11239 | 全对 |
| iDDPM：cosine 式 (17) 形、s=0.008（√β₀ 略小于 1/127.5）、clip 0.999、log 域插值方差、L_hybrid = L_simple + 0.001 L_vlb | ar5iv 2102.09672 | 公式全对；**动机句分辨率方向与论文相反（→F2）**；「cosine 改善 NLL」有 Table 2 支撑 ✓ |
| score-SDE：λ(t)=g(t)² 权重、Hutchinson 估计器、DDPM ancestral 对应式 | ar5iv 2011.13456 | 对 |
| EDM preconditioning 四系数 + σ_data=0.5 | NVlabs/edm 官方代码 `training/networks.py`（GitHub API） | 逐行一致（ar5iv 该 ID 无渲染版，改用官方代码为准） |
| FM（Lipman）约定 x₁=数据、p₀=噪声先验 | ar5iv 2210.02747 §3 | 对，映射框准确 |
| SD3：T5 移除胜率 50%/46%/38%、CLIP-L+OpenCLIP-G pooled 768+1280=2048、T5-XXL 4.7B、logit-normal 密度、shift Eq (23) 形、α=3.0@1024²、rf/lognorm(0,1) | ar5iv 2403.03206 | 全对 |
| DiT：Gflops 7.39/29.05/118.64、参数 33/130/458/675M、FID 2.27 vs LDM-4-G 3.60、adaLN-Zero 最优且近零开销 | ar5iv 2212.09748 | 全对（书中 118.7 与论文 118.6 差 0.1 →F5） |
| LDM：LDM-1 同预算 FID 差 38、LDM-{4-16} 平衡、LDM-32 限质量 | ar5iv 2112.10752 | **书中 f4/f16 边界划分与论文口径不符（→F4）**；LDM-1 对照组主张本身对 |
| PixArt-α：训练耗时 = SD v1.5 的 12%（753 vs 6250 A100 天）、adaLN 占 27%、Flan-T5-XXL 4.3B、LLaVA caption | ar5iv 2310.00426 | 全对 |
| SANA：0.6B/20× smaller/100+× faster/16GB laptop/<1s@1024²、ReLU linear attention、Mix-FFN 3×3 depthwise、Gemma-2、F32C32P1 | ar5iv 2410.10629 | 全对；**但 Eq. (1) 保留归一化分母（→F3）** |
| VAR：d30 FID 1.92、d30-re 1.73、2.0B、-re=rejection sampling、18M–2B scaling | ar5iv 2404.02905 | 全对 |
| LlamaGen：codebook 16384、f16、256 token、FID 2.18@3.1B(cfg)、vLLM 加速 326–414% | ar5iv 2406.06525 | 全对 |
| Emu3：8B、SBER-MoVQGAN-270M、codebook 32768、空间 8×8、512²=4096 token | ar5iv 2409.18869 | 全对 |
| SDXL：UNet 2.6B、micro-conditioning 三组标量 Fourier 嵌入加时 embedding、双 text encoder 合计 817M、base+refiner | ar5iv 2307.01952 | 全对 |
| MaskGIT：FID 6.18@8 iter、甜点 8–12 轮、更多迭代反而有害、cosine 最优、n=⌈γ(t/T)N⌉ | ar5iv 2202.04200 | 全对 |
| MUSE：Table 3（Muse 0.5s/1.3s TPUv4；Imagen 256² 9.1s 同硬件；LDM-50 512² 3.7s A100） | ar5iv 2301.00704 | 全对（书中硬件/分辨率标注准确） |
| Infinity/LCM/CM/DMD/DMD2/PD/RF/FM 摘要级主张 | arXiv API 摘要 | 全对（DMD 梯度=两 score 差、DMD2 去教师配对回归、LCM 增广 PF-ODE latent 蒸馏、CM 一步+多步、PD 减半+v 参数化、Infinity bitwise 三件套） |
| Vincent 2011 期刊 | CrossRef DOI 10.1162/NECO_a_00142 | Neural Computation 23(7):1661–1674 ✓（spec 勘误正确） |
| 18 个 GitHub repo + 14 个具体文件路径 | GitHub API | 全部 200（diffusion_tf/、cm/losses.py、mmditx.py、models.py、var.py、sgm/modules/… 等） |
| SDXL HF config | hf-mirror model_index.json | StableDiffusionXLPipeline + 双 text encoder ✓ |

### D. 规范与约定

| 项 | 方法 | 结论 |
| --- | --- | --- |
| lint | `scripts/lint-chapter.sh` × 7 | 7/7 LINT OK |
| 篇幅（CJK 字数口径） | 自有计数脚本（去代码/公式块） | 7.1=3997、7.2=5705、7.3=5388、7.4=7536、7.5=5850、7.6=6162、7.7=5640；推导重章（7.2/7.5）≥5500 ✓，其余 ≥3500 ✓ |
| 模板（metadata/总结/练习/参考/ASCII 图） | 逐章核对 | 7/7 齐全 |
| AI 味禁令词、排比、空洞形容词 | grep + 人工通读 | 无命中 |
| LaTeX 白名单（`\boxed`/编号环境/`\text{}` 中文等） | grep | 无违规 |
| 历史叙事 | 人工通读 | 无时间线/人物故事；F8 两处年份锚点现状判断属边界（S3） |
| 证据等级标注 | 逐章核对 | 齐全；[B]（Imagen/MUSE）与 [C]（FLUX）使用符合约束 |
| **约定一致性专项** | x₀/x₁/α/ᾱ/t 方向逐章抽查：7.1 §1 登记、7.2 §1 离散桥接 t_cont=t/T、7.3 §1 连续化、7.4 λ 坐标方向、7.5 §1 + Lipman 映射框（与约定页逐字一致）、7.6 z₀ 桥接、7.7 §1 表 | 7/7 一致：x₀=数据、t=1/T=噪声、采样 t:1→0、rectified flow 目标 ε−x₀；无混写 |

### 覆盖率声明

7/7 章全文逐行阅读；被引 30 篇论文 arXiv ID/标题 100% 核验，其中 14 篇取全文或官方代码深核数值主张；19 个非 arXiv 锚点 100% 核验存在性（18 repo + 14 路径 + 1 HF config）；三条强制推导链全部逐步复推；约 50 处数值主张中 45 处经重算或一手来源比对（其余为带「论文口径」对冲的转述性数字，如 SD3 scaling Figure 口径，未逐格重核——此类已在书中标明出处与日期）。

## 放行结论（初查）

**暂不放行；修复 F1–F4 后放行。** F1/F2 为 S1（引用张冠李戴与归因方向反转，均按 spec §6.3 属必修），F3/F4 为 S2 必修；S3×8 记入累计、下一波复审统一清零。F1/F2 所在章（7.4、7.2）修复后需复审该章。

## 复审记录（2026-11-12，第二轮：S1 修复复审）

**范围**：commit 区间 13231d0..600f3cb，含两笔 S1 修复——600f3cb（7.2，F2）与 b492f84（7.4 + references.md，F1）。

### F2（7.2 §2.3，iDDPM 动机方向）——verdict：PASS，放行

修复后文本（现 02-ddpm.md §2.3 末段）与 ar5iv 2102.09672 原文逐项比对：

| 修复后表述 | 原文证据 | 结论 |
| --- | --- | --- |
| 「linear schedule 在高分辨率图像上表现良好，但在 64×64 与 32×32 上 sub-optimal」 | §3.2："worked well for high resolution images, it was sub-optimal for images of resolution 64×64 and 32×32" | 方向已正 ✓ |
| 「小分辨率下 linear 的末端过于嘈杂」 | 同节："the end of the forward noising process is too noisy" | ✓ |
| 「最后约四分之一的 latent 已几乎是纯噪声（原文 Figure 3 口径）」 | Fig. 3 caption："The latents in the last quarter of the linear schedule are almost purely noise" | ✓ |
| 「cosine 的设计意图（原文）：ᾱ_t 中段近似线性下降、两端变化平缓」 | 原文："designed to have a linear drop-off of ᾱ_t in the middle of the process, while changing very little near the extremes of t=0 and t=T" | ✓ |
| 「其论文实验中 cosine schedule 改善 log-likelihood」 | Table 2：linear L_simple NLL 3.77 → cosine L_simple 3.68（200K iters）；正文 "using L_hybrid and our cosine schedule improves log-likelihood" | ✓ |
| 章内自洽性：「与上面的 32×32 量化账互为印证」 | 本章自算的「linear 尾部 22 步低于量化台阶」与 iDDPM 动机（小分辨率末端浪费）同向，原自相矛盾解除 | ✓ |

附带核验（同区间另一笔）：F1（7.4，common-diffusion 作者）——三处（正文 §4.4、参考列表、references.md `common-diffusion` 行）均已改为 Lin et al. / Shanchuan Lin et al.，与 arXiv API 作者表（Shanchuan Lin, Bingchen Liu, Jiashi Li, Xiao Yang）一致——PASS。

### 复审后放行状态

- **7.2：放行**（本章唯一 S1 已正确修复）。
- **7.4：放行**（本章唯一 S1 已正确修复；S3 累计项留待波末统一清零）。
- 全波放行仍待 F3/F4（7.6，S2）修复。

## 复审记录（2026-11-12，第三轮：7.6 的 S2 修复复审）

**范围**：commit 758eec0（`content(7.6): fix SANA normalization and LDM f ablation`，§1 f 消融段 + §8.2 SANA 段 + 总结表两行）。

### F3（7.6 §8.2，SANA 归一化分母误述）——verdict：PASS

| 修复后表述 | 原文证据（ar5iv 2410.10629） | 结论 |
| --- | --- | --- |
| 式 (15) 改为带分母形式 $O_i = \phi(Q_i)(\sum_j \phi(K_j)^{\top}V_j)\,/\,\phi(Q_i)(\sum_j \phi(K_j)^{\top})$ | 论文 Eq. (1) 逐符号一致（含两个共享项各算一次、全 query 复用） | ✓ |
| 注记「省力的来源是共享与重排，分母并不省」「分母与逐 query 读出为低阶项」 | 论文："we compute shared terms ... only once ... reused for each query"；复杂度阶核算成立（分母 O(Nd_h)，低阶） | ✓ |
| 新增：Table 8/12 消融「直接换 linear attention 掉 FID/CLIP、Mix-FFN 补回」 | Table 12 原文："Directly switching from DiT's self-attention to linear attention will result in FID and Clip Score performance loss, but adding Mix-FFN..."；数据行 FullAttn 18.7/24.9 → +Linear 21.5/23.3 → +MixFFN 18.9/24.8 | ✓ |
| 新增：Mix-FFN = inverted residual + 3×3 depthwise + GLU；NoPE「首个完全去位置编码的 DiT、无质量损失」；零填充 3×3 卷积隐式携带位置信息 | 论文 §2.2 逐项一致（"the first design that entirely omits positional embedding in DiT"；"3×3 convolution with zero padding can implicitly incorporate the position information"） | ✓ |
| 新增：Flow-DPM-Solver、笔记本 GPU 0.88s（FP16，Table 5） | 论文 §3 末 Flow-DPM-Solver（14–20 步收敛 vs Flow-Euler 28–50）✓；Table 5「Sana (FP16) 0.88s」于 laptop GPU/1024px ✓，与摘要「Sana-0.6B <1s @16GB laptop」同口径 | ✓ |
| 总结表 SANA 行同步（「分母保留」） | — | ✓ |

### F4（7.6 §1，LDM f 消融边界）——verdict：PASS

| 修复后表述 | 原文证据（ar5iv 2112.10752） | 结论 |
| --- | --- | --- |
| 消融集 $f\in\{1,4,8,16,32\}$、同预算（ImageNet、2M 步、单卡 A100，Fig. 6/7 口径） | Fig. 6 caption 逐项一致（"over 2M train steps on the ImageNet dataset ... single NVIDIA A100 with the same computational budget"）；同时关闭初查 S3 F11（枚举漏 f=1） | ✓ |
| 「LDM-1 训练时间显著长于一切 latent 档，同预算 FID 明显更差」 | Fig. 6 caption："Pixel-based LDM-1 requires substantially larger train times"；正文 FID gap 38 vs LDM-8 @2M 步 | ✓ |
| 「LDM-{4–16} 质量—吞吐基本拉平」 | "LDM-{4-16} strike a good balance between efficiency and perceptually faithful results" | ✓ |
| 「只有 LDM-32 因感知压缩过度封顶质量」 | "Too much perceptual compression as in LDM-32 limits the overall sample quality" | ✓ |
| 「LDM-4 与 LDM-8 最佳合成条件；复杂数据集（ImageNet）需更保守压缩率」 | 原文："Complex datasets such as ImageNet require reduced compression rates... In summary, LDM-4 and -8 offer the best conditions for achieving high-quality synthesis results" | ✓ |
| 章内自洽性 | §1 的 f 边界与本章 §6.3 自引「LDM-4 FID 3.60」不再打架；总结表 latent 行同步更新 | ✓ |

残留扫描：「f8 是甜点」「归一化分母一并省去」等旧表述已无残留；`lint-chapter.sh` 复跑 LINT OK。

### 复审后放行状态（更新）

- **7.6：放行**（F3/F4 均正确修复）。
- **全波放行**：S1×2（F1/F2）与 S2×2（F3/F4）全部关闭；7 章全部通过复审。S3 余 7 项（F5–F10、F12；F11 随 F4 一并关闭）记入累计，下一波复审统一清零。
