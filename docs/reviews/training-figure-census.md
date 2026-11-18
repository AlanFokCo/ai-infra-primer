# training 维度图普查：bare 围栏块 → 场景图注册清单

普查口径与 `site/scripts/build-content.mjs` 的 `replaceDiagrams` 完全一致：开围栏匹配 `^```(\w*)\s*$`，语言非空或未闭合不计数；块序号章内从 0 起，即注册表 `replacesBlock` 的值。注册键 = `training/<part-dir>/<chapter>#<blockIdx>`；training 图模块须显式 `dim: 'training'`（models 存量缺省 `models`）。格式先例：`site/scripts/lib/figures/CENSUS.md`。

总览：**39 章 / 61 块**（每章 ≥1 块）。优先级：高 25 / 中 29 / 低 7。

优先级判据：
- **高** = 机制/数据流核心图，动效（数据流光点、轮转、填充）增益大；缺它章节失脊柱。
- **中** = 对照/账务/流程图，场景化有益但静态图版亦可读。
- **低** = 伪代码 loop / 模板文本 / 目录树 / 纯表格，维持现有 algo / grid 深色图版即可，不建议注册场景图。

| Part | 章数 | 块数 | 高 | 中 | 低 |
|---|---|---|---|---|---|
| 01-autograd-and-backprop | 5 | 6 | 4 | 2 | 0 |
| 02-optimizers | 5 | 7 | 1 | 5 | 1 |
| 03-loss-functions | 2 | 2 | 0 | 2 | 0 |
| 04-data-parallelism | 4 | 6 | 4 | 2 | 0 |
| 05-model-parallelism | 5 | 12 | 7 | 5 | 0 |
| 06-megatron-and-stacks | 3 | 6 | 1 | 3 | 2 |
| 07-pretraining | 5 | 6 | 2 | 2 | 2 |
| 08-rl-math | 5 | 9 | 3 | 5 | 1 |
| 09-post-training | 5 | 7 | 3 | 3 | 1 |

建议注册模块命名：`site/scripts/lib/figures/training-part01.mjs` … `training-part09.mjs`（对应 Part 分组，与 models 侧 part01–part10 同构）。

---

## Part 1 — 01-autograd-and-backprop（5 章 6 块）

### 01-computation-graph-and-autograd
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 12 | 双层 MLP 计算图：左前向（x→mm(W1)→mm(W2)→add→mean→L，标 shape）右反向（逐节点 vjp 与 adjoint 公式） | 前向/反向双联图：数据流光点沿前向边流动后，adjoint 种子从 L 回流，途经节点逐一点亮并浮现对应 vjp 公式 | 高 |

### 02-operator-backward
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 4 | matmul 前向/反向三式与 FLOPs 账（2mnk×3 = 2×前向） | 双栏对照：前向 A@B=Y 与反向 dA/dB 两式，矩阵形状标注 + FLOPs 计数徽标 | 中 |
| 1 | 11 | fused CE 反向的分块流式：H 按行分块→片上投影→logsumexp→partial loss，p−e_y 就地覆写 Z_c | 分块流水线：H_c 逐块滑入片上 buffer，logsumexp 流式聚合，反向覆写高亮（不落 HBM 的边界用虚线框） | 高 |

### 03-mixed-precision
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 6 | 混合精度环路：fp32 主权重→fp16 副本→前向→×S 损失缩放→unscale+inf 检查→fp32 梯度→step | 双精度环路图：fp32/fp16 两色节点围成闭环，×S/÷S 转换闸门与 inf/NaN 检查哨卡点发光 | 高 |

### 04-gradient-checkpointing
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 7 | 分段 checkpoint：前向 k 段只存边界 x0..xk，反向逐段取边界重算后即弃 | 段链时间线：前向边界方框逐个点亮驻留，反向时段内激活重算亮起即灭，峰值公式在底部累加 | 高 |

### 05-gradient-accumulation
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 7 | naive vs no_sync 对照：4 个 micro-batch 时间线，AR 次数 4 vs 1 | 双泳道时间线：fwd+bwd 格重复排列，AR 通信脉冲标记在末 micro 才出现（no_sync 道 hook 关闭标灰） | 中 |

## Part 2 — 02-optimizers（5 章 7 块）

### 01-sgd-and-momentum
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 4 | heavy ball 与 EMA 两种等价写法的算子流对照（式 8/9，变量替换 m=(1−β)v） | 上下双通道流图：v 累加器道与 m 平滑器道平行，中间变量替换桥接虚线 | 中 |
| 1 | 7 | 二次碗截面几何：GD 在快方向 zigzag，heavy ball 历史速度自相抵消/同号累加 | 2D 等高线碗：GD 折线 zigzag 轨迹 vs momentum 平滑轨迹同屏动画，快/慢两特征方向标曲率 | 高 |

### 02-adam-and-adamw
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 5 | Adam 双通道（m/v 偏差校正）汇合更新；L2 写法（λθ 进两通道）vs AdamW（解耦 wd 直通参数） | m̂/v̂ 双通道汇合图：L2 与 AdamW 两条 λθ 支路分色，一条穿通道被 √v̂ 缩放、一条绕开直达 θ | 中 |

### 03-lr-schedules
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 11 | WSD vs cosine 两条 LR 曲线对照（warmup/稳定段/decay 标注，η_min≈10%） | 单坐标系曲线图：两条 schedule 曲线绘制动画，WSD 稳定段「可任意延长」用虚线延展暗示 | 中 |

### 04-gradient-clipping-and-stability
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 9 | 稳定性手段全景：数据→前向→logits→loss→反向→更新→参数七站各挂防护，底部监控总线 | 横向七站流水线图：每站防护标签挂牌，底部监控总线（grad norm/update ratio/logit max/entropy）脉冲 | 中 |

### 05-muon-and-spectral
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 10 | Newton-Schulz 迭代：X₀ 归一→5 步 A/B 多项式迭代（FLOPs 标注）→O≈US′Vᵀ | 迭代环图：X_k 矩阵绕 5 步循环，每步 FLOPs 徽标，奇异值谱从散落压进 [0.5,1.5] 邻域 | 中 |
| 1 | 5 | 参数全集二分树：隐藏层矩阵归 Muon、其余归 AdamW | 简单树形分叉（表格感强，维持 grid 图版） | 低 |

## Part 3 — 03-loss-functions（2 章 2 块）

### 01-ce-family
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 7 | CE 账流水线：H→LM 头 matmul→logits z（FLOPs/字节标注）→logsumexp→逐 token ℓ→mask→L | 竖向 shape 走读流：张量形状逐级变换，FLOPs 与字节账标签贴边 | 中 |

### 02-preference-and-contrastive-objectives
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 13 | DPO 数据流：偏好对 tokenize→chosen/rejected 半批拼接→π_θ/π_ref 双前向→fused LM head→logratio→loss | 纵向流水线：chosen/rejected 半批双色流动，fused head 处「logits 不物化」虚线框高亮 | 中 |

## Part 4 — 04-data-parallelism（4 章 6 块）

### 01-dp-and-ddp
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 15 | ring all-reduce 两阶段：4 rank 环，scatter-reduce 3 步部分和流转 + all-gather 3 步全和分发 | 四节点环动画：c0..c3 四色块沿环逐步流转，scatter-reduce 阶段逐块收齐全和后 all-gather 扩散 | 高 |
| 1 | 7 | bucket 与反向重叠：L4→L1 反向计算条，bucket 就绪即挂 AR，finalize 收尾 | 反向时间线：计算条推进，bucket 就绪点亮并触发 AR 通信流与后续层的计算重叠 | 高 |

### 02-zero
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 6 | DP vs ZeRO-1 通信阶段对照：归约/更新/同步四段，通信量各 2N 严格相等 | 双泳道阶段条：all-reduce vs RS+AG 分段对齐，元素量标签 2N=N+N 对应点亮 | 中 |
| 1 | 10 | ZeRO-3/FSDP 逐层节奏：前向逆序 AG→用→释放，反向再 AG，梯度 RS 归 owner，常驻量清单 | 层序时间线：参数卡片「聚→算→散」循环动画，前向正序/反向逆序两组箭头反向穿行 | 高 |

### 03-fsdp
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 7 | FSDP 单 unit 生命周期：常驻三分片 + 预取 AG 与上 unit 计算重叠 + RS 后释放 | 预取重叠泳道：unit ℓ−1 计算条与 unit ℓ 的 AG 通信条并行，reshard 释放点标记 | 中 |

### 04-memory-and-comms-accounting
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 17 | 显存决策树：16N 装得下→DDP/ZeRO-1；否则 ZeRO-2→ZeRO-3→hybrid/checkpoint/offload→TP/PP | 决策树图：容量判定线逐层分叉，各档标注显存档位与通信代价，末端汇到 TP/PP 出章指针 | 高 |

## Part 5 — 05-model-parallelism（5 章 12 块）

### 01-tensor-parallelism
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 8 | 列切/行切对照：列切前向免通信反向 1 次 AR，行切反之 | 双栏矩阵切分图：切分刀口动画，AR 通信标记分别落在反向（列切）与前向（行切） | 高 |
| 1 | 7 | 一层 TP 全貌：f/g 转换子夹 QKV 列切→attn→行切与 W1→W2 两支路，每层 4 次 AR | 层结构数据流：f/g 转换子门节点发光，数据流光点穿过两条支路，通信计数器累计 ×4 | 高 |

### 02-sequence-and-context-parallelism
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 6 | TP-only vs TP+SP 通信量对照表：f恒等↔g-AG、g-AR↔ḡ-RS，每方向每块恒等 | 双栏对照：复制区段/入口/出口三行对齐，通信量等式两侧同亮度平衡 | 中 |
| 1 | 7 | ring attention 轮转网格：4 rank × 4 round，Q_i×K_jV_j 命中矩阵 + (m,ℓ,O) 逐块合并 | 4×4 轮转网格动画：KV 块沿环逐 round 传递，✓ 格点亮累积，末尾合并成 causal softmax 全量 | 高 |

### 03-pipeline-parallelism
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 7 | GPipe 全前向-全反向格点图（p=4, m=4, T=14），前后 p−1 格气泡 | 甘特图填格动画：F 格先铺满再 B 格回流，气泡区半透明灰显 | 高 |
| 1 | 11 | 1F1B 格点图（p=4, m=8, T=22），B 级联逐段错开一格，含与 Megatron Figure 4 的口径注记 | 1F1B 甘特图：F/B 交错填格，B1 级联斜线高亮，气泡占比公式随填格浮现 | 高 |
| 2 | 6 | interleaved chunk 映射：p=4/v=2 共 8 chunk，一 micro 前向蛇形路径 d1.c1→…→d4.c8 | 设备×chunk 矩阵 + 蛇形路径动画：前向流线两次横贯设备行 | 中 |
| 3 | 10 | 反向拆分依赖图（B 链式不可推迟 / W 本地自由排序）+ 1F1B 梯形 vs ZB-H2 平行四边形轮廓 | 左 B/W 依赖小图 + 右双轮廓对照：W 块滑入尾部气泡把梯形推成平行四边形 | 高 |
| 4 | 5 | DualPipe 双向示意：正/反两向 micro-batch 对向流，正/反 chunk 并排使通信与计算互掩 | 双向流水线：两向流光对向穿行，all-to-all/P2P 通信条与对向计算条重叠 | 中 |

### 04-expert-parallelism
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 9 | EP 一次 MoE 前向：permute 复制→all-to-all dispatch→本地 expert FFN（各卡不等批量）→all-to-all combine 加权归约 | 四卡泳道动画：token 副本按路由跨卡流动（dispatch），本地 FFN 批量柱高低不一，combine 回流按 w 加权汇聚 | 高 |

### 05-combining-strategies
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 10 | rank 编号几何：16 rank 上 TP/CP/PP/DP 四族组的 stride 1/2/4/8 分层括号 | 16 rank 条形图：四层组括号逐层展开，stride 倍增与「TP 落节点内、DP 跨节点」标注 | 中 |
| 1 | 20 | 5D 并行配置决策流 ①–⑥：t→c→e→p→d 逐步定参 + 每步校验与回代总账 | 竖向六步决策流：每步判定菱形 + 校验旁注，失败回退虚线指回对应步骤 | 中 |

## Part 6 — 06-megatron-and-stacks（3 章 6 块）

### 01-megatron-architecture
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 18 | megatron/core 目录树（parallel_state / tensor_parallel / transformer / pipeline_parallel / optimizer 等） | 目录树（纯文本树，维持 algo 图版） | 低 |
| 1 | 32 | TransformerLayer TP 走读全图：x→norm→f→qkv 列切→RoPE→core_attn→proj 行切→g→norm→f→fc1→act→fc2→g | 竖向层栈走读动画：数据流光点自上而下穿过算子盒，f/g 转换子与两次集合通信门点亮，SP 变体注记随光点切换 | 高 |

### 02-megatron-runtime
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 10 | 16 rank 排布网格：t2/d2/p4 下 i_dp×i_pp 矩阵 + TP/DP/PP 三族组 stride 清单 | 2×8 网格 + 三族组括号分层展开（同 5.5#0 概念，Megatron 实例版） | 中 |
| 1 | 27 | train_step 伪代码走读：warmup/steady/cooldown 三段 + finish_grad_sync + optimizer.step 内部序列 | 伪代码流程（结构近 algo，维持 algo 图版） | 低 |

### 03-modern-training-stacks
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 5 | NeMo / Megatron-Core / Megatron-LM / Megatron-Bridge 四者关系与 HF 双向转换 | 组件关系图：Core 居中，NeMo/Megatron-LM 两侧汇入，Bridge 桥接 HF 生态双向箭头 | 中 |
| 1 | 11 | 训练栈↔推理栈接口：actor 训练引擎与 rollout 引擎间 prompts/经验数据/权重同步三向流 | 双栈环图：训练栈框与推理栈框相对，三条标注箭头成环（数据下、权重上） | 中 |

## Part 7 — 07-pretraining（5 章 6 块）

### 01-data-pipeline
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 8 | 数据管线五站：来源→过滤→去重→配比→tokenize+pack，各站挂方法名与 token 量级（36T→20T→[B,T]） | 横向五站流水线：token 量级随站递减的漏斗动画，方法标签逐站挂牌，终点 int32 序列条 | 高 |

### 02-recipes
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 13 | 四家 LR 曲线对照：DSv3 恒定+cosine+常数尾 / OLMo2 截断+线性 / Llama-3 cosine / GLM-4.5 cosine 全程 | 单坐标系四曲线绘制动画：各家线型分色，截断点与 mid-training 段标注 | 中 |

### 03-training-stability
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 11 | 信号→检测→动作三层映射表（loss/grad norm/update ratio/logit max 四行）+ spike score 周报注 | 三列映射（表格感强，维持 grid 图版） | 低 |

### 04-checkpointing-and-resumption
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 9 | dist-ckpt 重分片：DP=4 保存网格各写本地片→逻辑全量张量元数据→DP=2 加载网格跨边界拼装 | 左右网格+中间元数据卡：分片块从旧网格流入元数据再按新 offset 拼进新网格 | 高 |
| 1 | 10 | async save 三级泳道：GPU 训练不停→D2H staging（阻塞段）→后台 upload，两个 future 完成点 | 三泳道时间线：D2H 拷贝条与训练条重叠，staging/upload 两个 future 里程碑点亮 | 中 |

### 05-long-context-and-mid-training
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 8 | pre/mid/post 三阶段对照表：数据构成、窗长（4K→256K）、loss/schedule | 三行对照表（表格感强，维持 grid 图版） | 低 |

## Part 8 — 08-rl-math（5 章 9 块）

### 01-policy-gradient
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 3 | LLM-as-MDP 状态链：s_0=x 逐 token 转移，途中 r=0、末端 r_{T-1}=R(x,y) | 状态链动画：token 逐个挂链生长，奖励仅在末端节点绽放（RM/verifier 黑盒标记） | 中 |
| 1 | 4 | REINFORCE loop 三行伪代码：采样→算 G_t→拼梯度估计交 optimizer | 伪代码 loop（维持 algo 图版） | 低 |

### 02-actor-critic-and-gae
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 10 | actor-critic 数据流：rollout→critic 估值→δ_t→反向递推 Â→双损失 L_V/L_pg 一次 backward 同更 | 数据流图：δ 沿时间反向递推的回流动画，Â 分流进 L_V 与 L_pg 两损失盒汇合一次 backward | 中 |

### 03-ppo
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 11 | PPO clip 目标分段函数双图：A>0 时 min 截上、A<0 时 max 截下，梯度区/零梯度区标注 | 双折线图绘制动画：clip 平台在 r=1±ε 处折平，「有梯度/梯度=0」区间随 A 正负镜像翻转 | 高 |
| 1 | 19 | PPO 一轮循环：rollout（π_old 冻结）→GAE 冻结量→K epoch minibatch 更新→θ_old←θ 回环 | 三段式循环框图：rollout/GAE/更新三盒成环，冻结量与诊断（approx_kl/clipfrac）标签挂边 | 中 |

### 04-grpo-and-variants
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 12 | PPO-RLHF 四组件 vs GRPO 三组件对照：critic/GAE 消失，组内 G 条采样均值方差基线替代 | 双栏组件对照：GRPO 侧 critic 盒淡出、同 prompt 的 G 条奖励汇入 (r_i−mean)/std 基线公式 | 高 |

### 05-llm-rl-systems
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 10 | 一次 RL iteration 五步框：rollout→打分→advantage→多 epoch 更新→权重同步 | 五步循环框图：步骤逐格点亮成环，权重同步边标「每轮一次」 | 中 |
| 1 | 18 | single-controller 架构：编排进程居中，actor/critic 训练组、rollout 引擎组、ref/RM 推理组四向辐射，五步数据流编号 | 星型编排架构图：控制器居中脉冲，①–⑤ 编号数据流沿辐射边依序流动 | 高 |
| 2 | 9 | 训练→生成布局 resharding 算例：8 GPU 训练 (4,1,2) vs 生成 (2,1)d2，组内一次 AG 收 M/4，三家通信系数对照 | 双布局对照：训练分片与生成高并发分片的并集映射动画，all-gather 流量标签 | 中 |

## Part 9 — 09-post-training（5 章 7 块）

### 01-sft
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 8 | Llama-3 与 Qwen2.5 两系 chat template 文本（generation prompt 追加位标注） | 模板文本（字面 token 串，维持 algo 图版） | 低 |
| 1 | 5 | 多轮样本 mask 两策略：全轮监督 vs 末轮监督的 m_s 0/1 行对照 | token 段条 + 双 mask 行：监督位 1 逐段点亮（全轮三处 vs 末轮一处） | 中 |

### 02-rlhf-pipeline
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 12 | RLHF 三阶段全景：SFT(13k)→RM(33k)→PPO(31k)，数据形态从示范→排序→无标注，模型角色流转（π_SFT 兼 ref、RM 换 scalar head、critic 从 RM 初始化） | 三栏阶段流水线：数据量徽标随阶段流转，模型角色继承线（π_SFT→actor/ref、RM→critic）跨栏拉通 | 高 |
| 1 | 10 | InstructGPT PPO 单轮五步 loop：512 prompt 采样→合成奖励→GAE→8 minibatch 单 epoch 更新→权重同步 | 五步循环（结构同 8.5#0，超参实例版）；与 8.5#0 合并设计可二选一注册 | 中 |

### 03-dpo-family
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 17 | DPO 推导链五站：KL 约束目标→逐 prompt 变分→闭式最优解→reward 重参数化→BT 相消→DPO 目标 | 竖向推导链：五张公式卡逐级下落，β·log Z(x) 相消步骤高亮闪烁，终点 DPO loss 定格 | 高 |

### 04-rejection-sampling-and-rft
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 7 | RFT 一轮循环框：采 n 条→规则/RM 打分→top-k 筛选 D⁺→SFT 回灌，数据采后冻结 | 循环框图：采样→打分→筛选漏斗→回灌成环，「推理栈/训练栈」界标在环两侧 | 中 |

### 05-rlvr-and-reasoning
| block | 行数 | 块内容主题 | 建议场景图概念 | 优先级 |
|---|---|---|---|---|
| 0 | 37 | R1 配方全流程：V3-Base 分叉（R1-Zero 支线对照）→冷启动 SFT→推理 RL→拒绝采样+SFT→全场景 RL→R1→蒸馏支线 | 全流程树状阶段图：主线四阶段竖向贯通、R1-Zero 与蒸馏两条支线侧挂，checkpoint 节点标「数据发生器」角色 | 高 |

---

## 备注

- **与 models 侧的数量对照**：models 51 章注册 101 张场景图（平均每章 2 张，含 `afterHeading` 锚点插入的新增描述图）；training 侧 bare 块共 61 块，若只注册 replacesBlock 口径上限 61 张。本清单「高」档 25 张为最低交付目标，「中」档可按批次跟进。
- **跨章复用机会**：9.2#1 与 8.5#0 同为「RL 一轮五步循环」，9.2#1 可视 8.5#0 的场景图实现情况复用参数化版本；5.5#0 与 6.2#0 同为 rank 分组几何，亦可共享组件。
- **注册键示例**：`08-rl-math/03-ppo#0` → 图模块条目 `{ dim: 'training', slug: '08-rl-math/03-ppo', replacesBlock: 0, title: '…', svg: … }`。
