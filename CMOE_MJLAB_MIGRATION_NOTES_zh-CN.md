# CMoE 迁移至 MJLab 的实现说明与注意事项

本文记录将原始 [CMoE](https://github.com/Fudan-MAGIC-Lab/CMoE) 从 Isaac Gym/PhysX 迁移到本项目 MJLab/MuJoCo Warp 后实际遇到的问题、原因与最终实现。它不是通用的“稳定性调参清单”，而是一份迁移边界说明：任务定义、网络、PPO 参数、观测排列和奖励等内容应按原项目完整迁移；只有当 PhysX 与 MuJoCo Warp 对同一配置存在不同物理含义或不同数据结构时，才做明确、可追溯的引擎适配。当前任务入口位于 [`src/smp/rl/tasks/cmoe/__init__.py`](src/smp/rl/tasks/cmoe/__init__.py)，环境、资产、地形和专用算法分别位于 [`cmoe_env_cfg.py`](src/smp/rl/tasks/cmoe/cmoe_env_cfg.py)、[`asset.py`](src/smp/rl/tasks/cmoe/asset.py)、[`terrain.py`](src/smp/rl/tasks/cmoe/terrain.py) 与 [`src/smp/rl/cmoe`](src/smp/rl/cmoe)。

## 物理引擎边界：摩擦、接触和 NaN

这次迁移最容易误判的问题是训练在第一个或若干个 PPO 更新后报出 `normal expects all elements of std >= 0.0`。这个报错出现在动作分布采样处，但它不是最先发生的错误。实际链路是 MuJoCo Warp 的一次仿真步先产生非有限状态，`base_ang_vel` 随后进入历史观测，网络输出和动作分布再被 NaN 污染，最后 PyTorch 才在 `Normal.sample()` 处报错。因此，把标准差改成 log 参数化、降低 PPO 学习率或钳制网络输出都会改掉原算法，却没有解决产生 NaN 的物理状态。当前 CMoE 仍保留原始的 `init_std=1.0`、`learning_rate=1e-3` 和 `schedule="adaptive"`，见 [`cmoe_rl_cfg.py`](src/smp/rl/tasks/cmoe/cmoe_rl_cfg.py)。

原 Isaac Gym 配置把随机摩擦范围设为 `[0.0, 1.0]`，对应源码是本次迁移所用原项目副本的 [`legged_robot_config.py`](https://github.com/Hoshi-No-Ai/CMoE/blob/main/legged_gym/legged_gym/envs/base/legged_robot_config.py)。这个数值不能在 MuJoCo Warp 中不加解释地照搬。MuJoCo 的 `condim=3` 接触包含一个法向约束和两个切向滑动摩擦约束；当滑动摩擦为零或非常接近零时，MuJoCo Warp 的接触约束和 Newton/Cholesky 求解路径可能形成退化或病态系统，随后把 NaN 写入 `qacc`、`qvel` 和传感状态。MuJoCo 对 `condim` 与摩擦维度的定义见 [MuJoCo Computation 文档](https://mujoco.readthedocs.io/en/latest/computation/index.html#friction-cones)。MuJoCo Warp 上游已经有可复现的[零滑动摩擦产生 NaN](https://github.com/google-deepmind/mujoco_warp/issues/942)、[摩擦为 0.001 时仍出现 NaN](https://github.com/google-deepmind/mujoco_warp/issues/1517)以及 [Newton Cholesky 路径产生非有限状态](https://github.com/google-deepmind/mujoco_warp/issues/1415)的记录。

当前实现因此将所有 CMoE 碰撞几何的随机滑动摩擦明确设为 `[0.3, 1.0]`，而不是在算法层加入数值兜底：

```python
"foot_friction": EventTermCfg(
  func=dr.geom_friction,
  mode="reset",
  params={
    "asset_cfg": SceneEntityCfg("robot", geom_names=COLLISION_GEOM_PATTERN),
    "operation": "abs",
    "ranges": (0.3, 1.0),
    "shared_random": True,
  },
),
```

代码位于 [`cmoe_env_cfg.py`](src/smp/rl/tasks/cmoe/cmoe_env_cfg.py)，碰撞几何表达式 `.*_collision.*` 定义在 [`asset.py`](src/smp/rl/tasks/cmoe/asset.py)。这里事件名虽然沿用 `foot_friction`，实际匹配的是机器人全部命名碰撞体，并在同一个环境内共享一次随机结果。`0.3` 是本次迁移针对 MuJoCo Warp 接触路径作出的显式后端适配，不是对 CMoE 算法的重新调参。

“发生接触就结束本回合”也不能阻止这一类 NaN。环境每个控制步先执行四个物理子步，然后才计算 termination、reward 和 reset；如果接触求解器在某个 `sim.step()` 内已经把状态写成 NaN，终止管理器看到的是损坏后的状态。该顺序可直接查看 [`env.py`](src/smp/rl/tasks/cmoe/env.py)。当前仿真设置为 `timestep=0.005`、`decimation=4`，即 200 Hz 物理步和 50 Hz 策略步，并沿用 SMP 的 MuJoCo Newton 设置 `iterations=10`、`ls_iterations=20`，见 [`cmoe_env_cfg.py`](src/smp/rl/tasks/cmoe/cmoe_env_cfg.py)。解决顺序必须是修正导致病态接触的摩擦、碰撞几何或高度场，而不是把 reset 提前到一个尚未返回的物理子步之前。

## 地形与机器人资产：保持任务几何，转换碰撞表示

原 CMoE 地形以 5 cm 水平网格生成，这个分辨率同时影响台阶宽度、沟壑尺寸、障碍物位置和 77 维高度观测，所以不能整体改成 10 cm。直接把同一份 5 cm 高度场作为 MuJoCo Warp 碰撞高度场时，密集地形单元与脚部几何会产生大量候选接触，曾反复出现 `height field collision overflow, number of collisions >= 50`。该警告来自 MuJoCo Warp 的[高度场碰撞实现](https://github.com/google-deepmind/mujoco_warp/blob/main/mujoco_warp/_src/collision_convex.py)，其提示本身也要求减少高度场行列数或修改相撞几何。高分辨率碰撞场还会扩大接触缓冲区、增加显存占用、内核编译组合和每步窄相位计算量。

当前实现把“任务采样分辨率”和“物理碰撞分辨率”分开：原始地形仍按 5 cm 生成并用于观测与表面网格，送入碰撞器的高度场按两个方向各取一步，成为 10 cm 网格；高度值和 5 mm 垂直分辨率不变。

```python
_HORIZONTAL_SCALE = 0.05
_COLLISION_HORIZONTAL_SCALE = 0.1
_COLLISION_STRIDE = 2

collision_cfg = replace(self, horizontal_scale=_COLLISION_HORIZONTAL_SCALE)
output = _height_field_to_output(
  heights=raw[::_COLLISION_STRIDE, ::_COLLISION_STRIDE].T,
  cfg=collision_cfg,
  spec=spec,
  rng=rng,
)
output.instinct_surface_mesh = _height_field_to_hfield_surface_mesh(raw.T, self)
```

完整实现见 [`terrain.py`](src/smp/rl/tasks/cmoe/terrain.py)。这样处理保留了 CMoE 地形函数、难度含义和高度扫描的 5 cm 语义，同时减少了 MuJoCo Warp 真正参与接触计算的 hfield 单元数。MJLab 的地形生成、拼接、原点和 curriculum 组织方式可参考 [MJLab Terrain 文档](https://mujocolab.github.io/mjlab/main/source/terrain.html)及其[地形生成器源码](https://mujocolab.github.io/mjlab/main/_modules/mjlab/terrains/terrain_generator.html)。

机器人资产也不能把 SMP 的完整 29-DoF G1 XML 直接换个路径后使用。原 CMoE 只有双腿 12 个可动关节，腰、手臂和头部属于固定结构；关节顺序、默认姿态、质量、惯量、执行器力矩与速度上限都参与训练语义。当前 [`g1_12dof_cmoe.xml`](src/smp/rl/tasks/cmoe/assets/g1/xml/g1_12dof_cmoe.xml) 保留 CMoE 的 12 个下肢关节和固定上身，并由 [`asset.py`](src/smp/rl/tasks/cmoe/asset.py) 明确给出关节顺序、默认目标、刚度、阻尼、限幅和四子步内的动作延迟。固定结构在 MJCF 编译后会被折叠，所以运行时跟随 body 应使用真实存在的 `pelvis`，不能假定 URDF 中的 `torso_link` 仍是独立 body。

视觉网格与碰撞几何在 MJCF 中是两套职责。所有 STL mesh 只负责显示，设置为 `contype="0" conaffinity="0" group="1"`；真正参与碰撞的是带 `_collision` 名称的 sphere/capsule。球体和胶囊体减少了 mesh 接触时的 GJK/EPA、候选面与接触点数量，因而缩短启动阶段的碰撞内核准备和训练迭代耗时，也避免复杂网格在高度场上形成过多同时接触。MuJoCo Warp 上游的 [mesh 接触精度问题](https://github.com/google-deepmind/mujoco_warp/issues/1546)和 [mesh 与 primitive 接触问题](https://github.com/google-deepmind/mujoco_warp/issues/1104)说明了这条边界。碰撞体默认属于 `group="3"`，因此 play 时不显示；隐藏只是 viewer 显示选项，碰撞仍然生效。简化碰撞体时应保持脚底长度、宽度、离地高度以及躯干和四肢的实际覆盖范围，而不是用一个与原模型外形不符的大包围盒代替。

## 任务与算法语义：哪些必须原样保留

CMoE 不是换成 MJLab 默认 PPO 就能等价运行的普通 locomotion task。原方法包含历史观测编码、显式状态估计、地形潜变量、混合专家、prototype/contrastive 目标和对应的 rollout 数据，因此本项目保留了独立的 [`model.py`](src/smp/rl/cmoe/model.py)、[`algorithm.py`](src/smp/rl/cmoe/algorithm.py)、[`storage.py`](src/smp/rl/cmoe/storage.py)与 [`runner.py`](src/smp/rl/cmoe/runner.py)。Actor 输入仍为十帧、每帧 45 维 proprioception，加 77 维高度扫描，共 527 维；Critic 的特权观测、损失系数、PPO epoch/minibatch、KL 调度和初始动作标准差也按原实现定义。迁移时看到训练发散，不能先用 MJLab 默认网络、log-std、较低学习率或动作钳制替换这些参数，因为这会得到另一个算法，并掩盖前端物理状态是否已经损坏。

同样需要逐项核对的是 step 顺序、动作延迟、command 更新、扰动事件和 play 配置。原 CMoE 的 heading controller 会用航向误差生成角速度修正；play 若把 `ang_vel_z` 范围设成 `(0, 0)`，MJLab 的命令更新会把修正结果裁成零，机器人即使目标航向为 x 轴也无法主动纠偏。当前 play 保持 `lin_vel_x=0.8`、`lin_vel_y=0`、`heading=0`，同时给 heading controller 保留 `ang_vel_z=(-1, 1)`，对应代码仍在 [`cmoe_env_cfg.py`](src/smp/rl/tasks/cmoe/cmoe_env_cfg.py)。训练配置和 play 配置由同一个 task 注册中的 `env_cfg` 与 `play_env_cfg` 分开持有，因此仅修改 `if play:` 分支不会影响训练。原始随机地形 play 与顺序 course play 的切换位置是 [`__init__.py`](src/smp/rl/tasks/cmoe/__init__.py)；顺序 course 是迁移后的评估布局，不属于原始 CMoE 训练分布。

地形朝向、机器人初始朝向和速度命令必须使用同一坐标约定。MJLab 地形块在网格中的排布方向不等同于“机器人前进方向”，viewer 的 `azimuth` 也只改变相机而不改变物理朝向。需要分别检查地形数组的 x/y 索引、MJCF 根 body 朝向、初始 quaternion、`lin_vel_x` 和 heading，不能用旋转相机去修正训练坐标。课程地形还需要保持原始 40 种 terrain column 与 level 的映射，否则机器人会在错误难度或错误障碍类型上初始化。

## 诊断、性能与运行边界

为了确认非有限值最早出现在哪里，本项目在 [`runner.py`](src/smp/rl/cmoe/runner.py) 中按观测结构检查 command、`base_ang_vel`、projected gravity、joint state、上一动作与 height scan，并报告首次异常的 iteration、rollout step、env id、局部维度和对应动作；[`algorithm.py`](src/smp/rl/cmoe/algorithm.py) 则在 PPO minibatch 边界检查 observation、return、advantage 和模型输出。仿真配置同时启用 MJLab `NanGuardCfg(enabled=True)`，MuJoCo Warp 检测到状态异常时会把最近状态写入 `/tmp/mjlab/nan_dumps/`。这套日志的作用是定位错误边界，不改变张量、不替换 NaN、不跳过环境，也不修改训练结果。一次典型定位结果是 `after simulation: actor.history[0].base_ang_vel` 首先出现 NaN，由此可确定 PPO 标准差报错只是下游表现。

4096 个环境不会自动把单个仿真实例的显存跨多张 GPU 汇总；每个进程仍必须在自己的 GPU 上容纳复制后的 MuJoCo data、hfield、接触缓冲区、rollout 和网络。曾出现的 `Failed to allocate 393216000 bytes on device` 就是明确的显存不足，而不是训练数值发散。如果同一张卡上另有 Ollama 等进程占用约 8 GiB，4096 环境在 16 GiB 卡上会直接越界。多 GPU 训练用于让不同 rank 各自承担环境和梯度同步，不等于把多张卡的显存拼成一个地址空间。MuJoCo Warp 的 GPU 批量仿真和编译缓存机制见其[官方仓库](https://github.com/google-deepmind/mujoco_warp)。首次遇到某组模型形状、接触类型或 GPU 架构时会编译 Warp kernels；日志显示 `(compiled)` 时耗时较长，后续同一缓存命中显示 `(cached)`，这属于启动阶段而不是 PPO 卡住。

简化碰撞体、降低碰撞 hfield 分辨率和减少无用接触对是本次最直接的仿真性能优化，它们减少的是每个物理子步的碰撞工作量，收益会累积到数万次迭代。相反，减少 actor 网络规模虽然也可能变快，却改变了 CMoE 方法本身，不属于等价迁移。判断优化是否合格的标准是：观测与奖励使用的地形、机器人自由度、控制频率和算法参数保持不变，而仅替换后端用来计算相同物理边界的表示。

## 后续任务迁移核对表

| 边界 | 必须核对的内容 | 本次 CMoE 的落点 |
| --- | --- | --- |
| 任务注册 | task id、训练/播放配置、专用 runner 是否在包导入时注册 | [`tasks/cmoe/__init__.py`](src/smp/rl/tasks/cmoe/__init__.py) |
| 机器人资产 | 自由度、关节顺序、默认姿态、质量惯量、限位、固定 body、site 名称 | [`asset.py`](src/smp/rl/tasks/cmoe/asset.py)、[`g1_12dof_cmoe.xml`](src/smp/rl/tasks/cmoe/assets/g1/xml/g1_12dof_cmoe.xml) |
| 控制语义 | physics dt、decimation、动作缩放、PD 参数、动作延迟 | [`asset.py`](src/smp/rl/tasks/cmoe/asset.py)、[`cmoe_env_cfg.py`](src/smp/rl/tasks/cmoe/cmoe_env_cfg.py) |
| 接触语义 | collision geom、`condim`、摩擦随机化、接触终止顺序 | [`g1_12dof_cmoe.xml`](src/smp/rl/tasks/cmoe/assets/g1/xml/g1_12dof_cmoe.xml)、[`env.py`](src/smp/rl/tasks/cmoe/env.py) |
| 地形语义 | 生成分辨率、碰撞分辨率、难度、列/行映射、原点与高度扫描 | [`terrain.py`](src/smp/rl/tasks/cmoe/terrain.py) |
| 学习语义 | 观测顺序、历史长度、网络模块、分布、损失、PPO 参数和 storage | [`src/smp/rl/cmoe`](src/smp/rl/cmoe)、[`cmoe_rl_cfg.py`](src/smp/rl/tasks/cmoe/cmoe_rl_cfg.py) |
| 首个异常边界 | 仿真输出、rollout 写入和 PPO minibatch 分层检查，保留 env id 与动作 | [`runner.py`](src/smp/rl/cmoe/runner.py)、[`algorithm.py`](src/smp/rl/cmoe/algorithm.py) |

迁移完成的判断不是“任务能启动”或“第 0 次迭代能打印”，而是上述边界都有一一对应关系：原项目定义能直接表达的部分保持原值；无法直接表达的部分有明确的 PhysX → MuJoCo/MJLab 转换；任何性能修改只作用于后端表示，不悄悄改变任务或算法。CMoE 的许可证与本项目采用和修改的上游文件记录在 [`LICENSES/CMoE.txt`](LICENSES/CMoE.txt)和 [`NOTICE`](NOTICE) 中。
