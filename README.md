# sharpa_deploy_bridge：用训练好的 DexLearn（Sharpa 右手）在真机上抓真实物体

右臂 FANUC CRX + 右手 Sharpa Wave + 一个固定的 RealSense D435i。本仓库与实验室部署代码 `~/ws_fanuc` **完全解耦**：
不改、不编译部署代码，只通过运行中的 `bimanual_manipulation` launch 的话题通信。

```
拍照（D435i） → FoundationStereo 深度 → 物体点云 → DexLearn 生成抓取 → 过滤排序 → 转到机器人坐标 → 慢速执行
                └──────────── 不需要标定，也不动机器人，可以随时单独看结果 ────────────┘   需要标定     需要机器人
```

安全约定：会动机器人的命令默认**只规划不动**；加 `--execute` 后每一段都要按 Enter；Ctrl+C 只是停止发送（**不是急停**），
真机时手放在示教器急停上；先 mock 后真机。

---

## 终端类型（下文每个代码块开头都写全了环境，直接整块复制即可）

| 名称 | 做什么 | 环境 |
|---|---|---|
| **L-mock / L-real** | 启动实验室部署系统 | ROS + ws_fanuc（真机再加 Sharpa SDK）+ 域号，**不激活 venv** |
| **T-mock / T-real** | 本仓库里要读/写机器人的脚本 | 同上 + 工具 venv |
| **C** | 相机、点云、选抓取、离线计算 | 只激活工具 venv |

mock 用域号 198 且只在本机；真机域号 87。FoundationStereo 和 DexLearn 各有自己的 Python，命令里直接写了它们的路径。

---

# 0. 第一次使用：安装（只需做一次，全部在 `~/bodex_sharpa_deploy` 下，不需要 sudo）

### 0.1 更新代码

```bash
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
git pull
git log --oneline -1
```

预期：显示最新提交（与 GitHub 上一致）。

### 0.2 工具环境 + FoundationStereo（约 10 分钟，主要是下载）

```bash
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
bash install/install_tools.sh
```

做了什么：建工具 venv（OpenCV、RealSense、scipy、matplotlib，能看到 ROS 的 rclpy）；建 `fs_venv`（适配 RTX 5090 的
PyTorch CUDA 12.8 版）；下载 FoundationStereo 代码和官方权重并校验 sha256。
预期最后几行：`torch 2.x+cu128 cuda True NVIDIA GeForce RTX 5090`，两行 `ok .../model_best_bp2.pth`，`DONE`。
如果权重显示 `MISSING`（Google Drive 下载失败）：按提示用浏览器下载 `23-51-11`、`11-33-40` 两个文件夹放进
`~/bodex_sharpa_deploy/FoundationStereo/pretrained_models/`，再运行一次这个脚本（会跳过已完成的部分）。

### 0.3 DexLearn 推理环境（CPU，约 20–40 分钟，主要是编译）

```bash
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
bash install/install_dexlearn_cpu.sh
```

做了什么：在 `~/bodex_sharpa_deploy/miniforge3` 装 Miniforge；建与训练时相同版本的环境（Python 3.10、PyTorch 2.0.1、
MinkowskiEngine 0.5.4、pytorch3d 0.7.5，CPU 版，自带 gcc 和 OpenBLAS）；克隆 DexLearn 代码（与训练模型同一提交）。
为什么用 CPU：MinkowskiEngine 在 RTX 50 系列 + CUDA 12.8 上编译很不稳定；CPU 推理 200 个抓取约 1–2 秒，足够用，
且在服务器上验证过与训练环境结果一致（差异 ≤ 0.0002）。
预期最后：`OK torch 2.0.1+cpu | MinkowskiEngine 0.5.4 (CPU) | model class BaseModel` 和 `DONE`。
编译失败会打印日志最后 30 行和日志路径，把它发给我。

### 0.4 训练好的模型文件

模型（189 MB，不放在 GitHub）放在 `~/bodex_sharpa_deploy/models/dexlearn_sharpa_v3_s34_50k/`，内有 `config.yaml`、
`step_050000.pth`、`MANIFEST.json`。检查：

```bash
cd ~/bodex_sharpa_deploy/models/dexlearn_sharpa_v3_s34_50k && sha256sum config.yaml step_050000.pth
```

预期：`45ff30f4…7ea7  config.yaml` 和 `0c5e8ca8…fd9e  step_050000.pth`（与 MANIFEST.json 一致）。
这就是 V3 + 34 球 + kp12 + α1、8 卡 50k 步的模型（测试集全候选物理成功率 57.87%，似然 Top-10 75.88%）。
这个文件夹不在公开仓库里，由我们单独拷贝到主机；目录不存在或校验不一致时先不要往下做。

### 0.5 自检（不需要相机和机器人，约 2 分钟）

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/test_offline.py
python3 perception/test_offline.py
python3 calibration/test_hand_mount.py
FS_ROOT=~/bodex_sharpa_deploy/FoundationStereo ~/bodex_sharpa_deploy/fs_venv/bin/python perception/fs_depth.py --demo captures/fs_demo
```

预期：前三个各以 `ALL OK` 结束；最后一个打印 `captures/fs_demo: x.xx s, valid 85–95% ...`，
`captures/fs_demo/fs_vis.png` 中间是一张清晰的彩色深度图（桌上的杯子、纸巾盒）。

---

# A. 执行一个已有候选（验证执行链；已在真机做过：在空中、高处）

```bash
# 终端 L-mock：启动模拟的双臂双手
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=198
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
ros2 launch bimanual_manipulation bimanual_system.launch.py collision_force_limit:=false
```

预期：RViz 打开，双臂双手在默认姿态。

```bash
# 终端 T-mock：先只规划，再执行
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
source ~/bodex_sharpa_deploy/venv/bin/activate
export ROS_DOMAIN_ID=198
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 execute_grasp.py --grasp examples/master_chef_can_pose000_14_grasp.npy --release
python3 execute_grasp.py --grasp examples/master_chef_can_pose000_14_grasp.npy --release --execute
```

预期：第一条打印 JSON 规划报告（总时长约 74 s、`min_clearance_m` ≈ 0.149）并以 `DRY RUN only` 结束；
第二条每段按 Enter 后 RViz 里右臂慢慢转动、手张开、下降、握紧、张开抬起，最后打印 `Done`。

真机版本见 E 节的 L-real / T-real 写法。

---

# B. 相机标定（相机在 world 中的位置；只用右臂）

**为什么**：相机看到的点要换算到机械臂的坐标系才能去抓。**原理**：把一个打印的码刚性固定在右臂末端，让右臂摆
15–30 个姿态；每个姿态同时知道"法兰在 world 中的位置"（关节角正运动学）和"码在相机里的位置"（图像），
解 `A_i·Y = X·B_i` 得到相机位置 X（Y 是码相对法兰的位置，自动求出，不用量）。只用彩色图。

### B.1 确认你的码（自己打印的码也可以）

支持：我们的 ChArUco 板、**单个 ArUco / AprilTag 码**、**QR 二维码**。服务器上用模拟机器人 + 模拟相机（约 1 m）跑完整采集和求解，
20 个样本时手部工作区的误差：

| 标定物 | 结论 | 手部工作区误差 |
|---|---|---|
| ChArUco 板（A4） | GOOD | 0.2 mm |
| AprilTag 36h11，10 cm / 15 cm | GOOD | 3.2 mm / 3.8 mm |
| QR 二维码，12 cm / 18 cm | USABLE | **14 mm / 10 mm** |

QR 码的角点定位有系统偏差，会让相机位置整体偏约 1 cm，而且求解的结论检查不出这种偏差，**不建议用 QR 码**。
手头只有 QR 码的话，用下面这条生成一个 AprilTag（A4 能打 12–15 cm）：

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/make_board.py --out calib/board --apriltag 0.12      # 打印 calib/board/apriltag_print.pdf（100%）
```

用卡尺量码的**黑色正方形外边长**（不含白边，单位米），把码拿到相机前 0.4–0.8 m：

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/identify_target.py --size-m 0.120          # 换成你量的边长
```

预期：`found: 1 marker(s) of DICT_APRILTAG_36h11, ids [..]`（或 `found: QR code`），并写出 `calib/board/board.json`。
显示 `nothing detected` 时看 `calib/board/identify_input.png`：码要完整、平整、不反光。
（若用我们的 ChArUco 板：`python3 calibration/make_board.py --out calib/board`，打印后 `--set-measured 方格边长 码边长`。）

### B.2 检查相机和识别

```bash
# 终端 C（码拿在右手大致要去的位置）
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/camera_check.py --board calib/board/board.json
```

预期：列出所有 RealSense（含 D405），自动选中 `D435I`；USB 版本 3.x；`board detected in 10/10 frames`；
`reproj` < 0.5 px（QR 可到 1 px）；`position jitter` 几毫米以内。`calib/camera_check/camera_check.png` 上码中心画了坐标轴。

### B.3 把码固定到右臂末端

刚性固定（螺丝/夹具/扎紧的支架）在 J6 之后（转接件或手背），臂在起始姿态时码面朝向相机；不能装在手指上，不能装在 J5 及以前的连杆上。
标定全程码不能动。码会伸出手外，下面的 `--clearance` 用 0.15。

### B.4 采集（推荐方式 A：只读 + 示教器点动，脚本不发任何运动命令）

```bash
# 终端 L-real（只读启动：PC 只读状态，不控制机械臂）
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export PYTHONPATH=/opt/sharpa-wave-sdk/python:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib:$LD_LIBRARY_PATH
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
ros2 launch bimanual_manipulation bimanual_system.launch.py crx_mock:=false sharpa_backend:=sharpa_sdk read_only:=true rviz:=true
```

```bash
# 终端 T-real
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export PYTHONPATH=/opt/sharpa-wave-sdk/python:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib:$LD_LIBRARY_PATH
source ~/bodex_sharpa_deploy/venv/bin/activate
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/collect_handeye.py --out calib/run1 --board calib/board/board.json --mode manual
```

操作：示教器点动右臂（含绕不同方向的转动 15–35°，位置变化 ±10 cm），每停稳一次在终端 T-real 按 Enter；采 20–30 个后按 q。
预期每次：`sample N: 4 corners (或 24), reproj 0.xpx, frame spread 0.xmm, board 0.7 m away`；码没识别到会提示该样本不用。
先确认点动时 RViz 里的臂跟着动（只读模式下状态正常）。
（方式 B 自动：可写 launch 下 `--mode auto --clearance 0.15`，先不加 `--execute` 看规划，再加；脚本以 5°/s 走 15 个姿态，每步 Enter。）

### B.5 求解

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/solve_handeye.py calib/run1 --out calib/camera_extrinsics.json
```

预期：`verdict` 为 `GOOD`（平移 RMS ≤ 3 mm、留一法工作区偏移 ≤ 2 mm）或 `USABLE`；`camera_position_world_m` 在两臂之间、
比基座高（例如 y≈−0.45、z≈0.8–1.0）；`camera_viewing_direction_world` 的 z 为负（朝下看）。`REDO` 就补采更多、转动更大的姿态。

### B.6 验证（只读，不动机器人）

终端 L-real 保持只读启动；终端 T-real（环境同 B.4）运行：

```bash
python3 calibration/verify_extrinsics.py --extrinsics calib/camera_extrinsics.json --board calib/board/board.json
```

预期：`calib/verify/overlay.png` 上黄圈落在右臂各关节上（紫圈是手，手安装外参标好前可能整体偏几厘米）；
`board_check` 几毫米 / 1° 以内；`table.table_z_m` 是桌面在 world 中的高度（记下来）。

---

# C. 手安装外参（手相对法兰的位置；不用尺子量）

**为什么**：DexLearn 给出的是手掌根部 `right_hand_C_MC` 的位置，机械臂要的是法兰的位置；两者之间差一个"手怎么装在法兰上"
的固定变换，现在实验室配置里是占位值（平移 0）。**原理**：相机标好后，深度能看到手的真实位置，关节角给出法兰位置；用 Sharpa
手自己的 CAD 网格去贴合多个视角的深度点云，求出这个变换。仿真：从偏 13 mm / 40° 的初值恢复到 0.4 mm / 0.1°。

### C.1 采集 5–8 个视角（只读 launch 同 B.4；拆掉标定码；手保持一个姿态，例如张开）

```bash
# 终端 T-real
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export PYTHONPATH=/opt/sharpa-wave-sdk/python:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib:$LD_LIBRARY_PATH
source ~/bodex_sharpa_deploy/venv/bin/activate
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/collect_hand_views.py --out calib/hand_views --table-z <B.6 的 table_z_m>
```

操作：示教器把右臂移到手完整入镜、离相机 0.5–0.9 m、离桌面和其他物体远一些的位置，每次换一个腕部朝向，停稳按 Enter。
预期每次：`saved calib/hand_views/view_0N (raw depth valid xx%)`。

### C.2 深度 + 求解

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
FS_ROOT=~/bodex_sharpa_deploy/FoundationStereo ~/bodex_sharpa_deploy/fs_venv/bin/python perception/fs_depth.py calib/hand_views/view_*
python3 calibration/hand_mount_icp.py calib/hand_views/view_* --extrinsics calib/camera_extrinsics.json --out calib/hand_mount.json
```

预期：`verdict` 为 `GOOD`（平均内点 > 70%、留一视角变化 < 2 mm / 1°）或 `USABLE`；`view_*/hand_mount_overlay.png` 里
紫色的手与图中真实的手重合。之后执行时加 `--hand-mount calib/hand_mount.json`（只影响本仓库的规划，不改实验室配置）。

---

# D. 面对真实物体：点云 → DexLearn → 抓取（不需要标定，不动机器人）

把一个物体放在桌上右手够得到、相机看得清的位置（机械臂不要挡住），然后一条命令跑完 ①–⑤（标定做完后自动加 ⑥）：

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
bash run_scene.sh scene01
```

| 步骤 | 做什么 | 预期 / 去看 |
|---|---|---|
| ① 拍照 | D435I 拍彩色、红外左右图、RealSense 深度 | `saved captures/scene01: Intel RealSense D435I ..., baseline 50.x mm` |
| ② FoundationStereo | 用红外左右图算深度 | 约 1–3 s；`captures/scene01/fs_vis.png`：中间（FoundationStereo）应比右边（RealSense）更完整、边缘更干净 |
| ③ 物体点云 | 找桌面 → 取桌面以上的物体 → 桌面坐标系（z 向上） | `object_fs/overlay.png` 只有物体是红色；`height_above_table_m` ≈ 物体实际高度；`object_views.png` 三视图像该物体；`scene.ply` 可用 MeshLab 打开 |
| ④ DexLearn | 物体移到训练场景中心、取 1024 点 → 200 个抓取（含三个姿态和似然） | 约 1–2 s（CPU）；`saved .../grasps_table.npy` |
| ⑤ 过滤排序 | 去掉超关节限位、手碰桌、预抓时穿进物体的 → 按似然排序取前 10 | `passed` 通常 100–180 / 200；`grasps/selected_views.png`：灰 = 物体，红 = 手，应包住物体且不在桌面以下 |
| ⑥ 转到 world | 需要 `calib/camera_extrinsics.json` | 打印前 3 个抓取的 world 坐标和下一步的执行命令 |

选错了物体（例如选到了别的东西）：在 `captures/scene01/color.png` 上找物体上一个像素坐标 (u, v)，然后
`bash run_scene.sh scene01 --skip-capture --pixel u v`。想对比 RealSense 自带深度：`--skip-capture --depth raw`。

单独运行某一步、对比两种深度（`perception/compare_depth.py`）、各脚本参数：见各脚本开头说明（`python3 xxx.py --help`）。

---

# E. 执行抓取（需要 B、C；先 mock 再真机；第一次只做到 pregrasp）

⑥ 打印的命令形如下面这样（`table-z` 取 `object_fs/object.json` 里的 `table_z_world_m`）。先在 mock 里看规划和动作：

```bash
# 终端 L-mock：同 A 节
# 终端 T-mock
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
source ~/bodex_sharpa_deploy/venv/bin/activate
export ROS_DOMAIN_ID=198
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 execute_grasp.py --grasp captures/scene01/grasps/grasps_world.npy --index 0 --wrist-frame world \
    --table-z <table_z_world_m> --clearance 0.005 \
    --obstacles captures/scene01/object_fs/object_world.npy --hand-mount calib/hand_mount.json --stop-after pregrasp
```

预期：打印规划报告并以 `DRY RUN only` 结束。若 `PLAN REJECTED`（够不到 / 离左臂太近 / 离桌太近），换 `--index 1`、`2`…。
确认后加 `--execute` 在 mock 里看动作。然后真机：

```bash
# 终端 L-real（可写；保留碰撞力限制，等日志里限制器 ARMED）
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export PYTHONPATH=/opt/sharpa-wave-sdk/python:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib:$LD_LIBRARY_PATH
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
ros2 launch bimanual_manipulation bimanual_system.launch.py crx_mock:=false sharpa_backend:=sharpa_sdk rviz:=true viewer:=true
```

```bash
# 终端 T-real
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export PYTHONPATH=/opt/sharpa-wave-sdk/python:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib:$LD_LIBRARY_PATH
source ~/bodex_sharpa_deploy/venv/bin/activate
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 execute_grasp.py --grasp captures/scene01/grasps/grasps_world.npy --index 0 --wrist-frame world \
    --table-z <table_z_world_m> --clearance 0.005 \
    --obstacles captures/scene01/object_fs/object_world.npy --hand-mount calib/hand_mount.json --stop-after pregrasp --execute
```

预期：右臂以 ≤ 5°/s 移到 pregrasp 正上方 8 cm → 手张到 pregrasp → 竖直下降到 pregrasp（手在物体旁边、未接触）。
看位置是否合理（手是否真的在物体旁边、高度对不对），再去掉 `--stop-after pregrasp`（做到 squeeze）并加 `--release`（抓完张开抬起）。
回默认姿态：`ros2 run bimanual_manipulation move_to_default_pose.py --execute`（部署代码自带）。

### execute_grasp.py 主要参数（默认已很慢）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--arm-speed-deg / --arm-accel-deg` | 5 / 10 | 每个臂关节 °/s、°/s²；> 20 °/s 拒绝 |
| `--hand-speed-deg / --hand-accel-deg` | 10 / 20 | 每个手关节；> 30 °/s 拒绝 |
| `--wrist-frame` | scene | scene：把候选放到当前手位置上方的空中（A 节）；world：抓真实物体（E 节） |
| `--table-z` / `--clearance` | 0 / 0.10 m | 肘部以外及手的连杆原点必须高于桌面这么多（第 1 段只要求不低于起点） |
| `--hand-mount` | 无 | C 节结果；只替换规划模型里的法兰→手变换 |
| `--obstacles` / `--obstacle-clearance` | 无 / 0.02 m | 物体点云；到 pregrasp 之前手与物体保持距离 |
| `--stop-after` / `--release` | squeeze / 否 | 只做到某一步；抓完张手抬起 |

执行前自动拒绝：已有其他发送端（键盘/GUI）、驱动未订阅（read_only）、碰撞力限制器非 ARMED、规划后机器人动过。
执行中立即停发：反馈 > 0.5 s 不更新、限制器 TRIPPED、跟踪误差（臂 6°、手 20°）、Ctrl+C。

---

# 服务器上的验证（没有真机、真相机）

| 部分 | 结果 |
|---|---|
| 相机标定（mock 机器人 + 模拟相机完整流程，OpenCV 4.10 / 5.0） | 工作区误差：ChArUco 0.2 mm，AprilTag 3.2–3.8 mm，QR 10–14 mm（不建议） |
| FoundationStereo（官方 ViT-S，torch 2.7，无 xformers，离线） | 官方示例 960x540：2.0 s（A800），有效 90.7% |
| 点云（合成场景，无标定 / 有标定） | 红外→彩色配准中位误差 0.28 mm；物体高度 0.121 m（真值 0.12） |
| 真实双目图全链路（官方示例里的杯子，无标定） | 只选中杯子（12 cm）；200 个抓取 143 个通过 |
| DexLearn 接入（100 测试场景 × 100 候选，Bench 物理评测） | 本仓库链路 53.8%，原流程 56.1%，只改随机性对照 54.4%（差异在噪声内） |
| `install/install_dexlearn_cpu.sh`（新环境从头运行） | 克隆、编译、自检全部通过；与训练环境同输入同种子输出差 ≤ 0.0002；200 个抓取 1.3–2 s |
| `run_scene.sh`（主机同样的目录布局，真实双目示例图） | ②–⑤ 全部通过：深度 1.3 s，杯子 12 cm，DexLearn CPU 2 s，150/200 通过 |
| 手安装外参（合成） | 0.36 mm / 0.09° |
| world 抓取 → 执行器（mock） | 7 段动作完成 |

# 已知限制

- DexLearn 训练用的是模拟单视角点云（模拟 Azure Kinect，约 0.74 m）；真实 D435i 的视角和噪声不同，成功率要实测。
- 执行时手从 pregrasp 正上方 8 cm 竖直下降；很高的物体做侧面抓取时可能需要别的接近方向。
- 碰撞检查是近似的（连杆原点 / 网格采样点），臂与环境只检查了与左臂的距离。
