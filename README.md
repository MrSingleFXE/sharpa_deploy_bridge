# sharpa_deploy_bridge：用训练好的 DexLearn（Sharpa 右手）在真机上抓真实物体

右臂 FANUC CRX + 右手 Sharpa Wave + 固定的 RealSense D415。本仓库与实验室部署代码 `~/ws_fanuc` 解耦：
不改、不编译部署代码，只通过运行中的 `bimanual_manipulation` launch 的话题通信。

```
拍照（D415）→ FoundationStereo 深度 → 物体点云 → DexLearn 生成抓取 → 过滤排序 → 转到 world → 执行
```

安全约定：会动机器人的命令默认只规划不动，加 `--execute` 才执行；Ctrl+C 只是停止发送，**不是急停**，
真机时手放在示教器急停上；新流程先 mock 后真机。

---

## 终端类型（每个代码块开头都写全了环境，整块复制即可）

| 名称 | 做什么 | 环境 |
|---|---|---|
| **L-mock / L-real** | 启动实验室部署系统 | ROS + ws_fanuc（真机再加 Sharpa SDK）+ 域号，不激活 venv |
| **T-mock / T-real** | 本仓库里要读/写机器人的脚本 | 同上 + 工具 venv |
| **C** | 相机、点云、选抓取、离线计算 | 只激活工具 venv |
| **K** | 相机 ROS 驱动 + industrial_calibration 标定 | ROS + `~/bodex_sharpa_deploy/ws_calib`，不激活 venv |

mock 用域号 198 且只在本机；真机域号 87。

---

# A. 安装

### A.1 更新代码

```bash
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
git pull
git log --oneline -1
```

### A.2 工具环境 + FoundationStereo

```bash
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
bash install/install_tools.sh
```

### A.3 DexLearn 推理环境

```bash
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
bash install/install_dexlearn_cpu.sh
```

### A.4 标定工具 industrial_calibration

```bash
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
bash install/install_calibration.sh
```

### A.5 模型文件

把模型文件夹 `dexlearn_sharpa_v3_s34_50k`（含 `config.yaml`、`step_050000.pth`、`MANIFEST.json`）放到 `~/bodex_sharpa_deploy/models/` 里，然后检查：

```bash
cd ~/bodex_sharpa_deploy/models/dexlearn_sharpa_v3_s34_50k && sha256sum config.yaml step_050000.pth
```

预期：`45ff30f4…7ea7  config.yaml` 和 `0c5e8ca8…fd9e  step_050000.pth`。

### A.6 自检

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/test_offline.py
python3 perception/test_offline.py
python3 calibration/test_hand_mount.py
FS_ROOT=~/bodex_sharpa_deploy/FoundationStereo ~/bodex_sharpa_deploy/fs_venv/bin/python perception/fs_depth.py --demo captures/fs_demo
```

预期：前三个以 `ALL OK` 结束；最后一个打印 `captures/fs_demo: x.xx s, valid 85–95% ...`，`captures/fs_demo/fs_vis.png` 中间是清晰的深度图。

---

# B. 相机标定（industrial_calibration + ChArUco；相机移动后重做）

ChArUco 板刚性固定在右臂法兰上，右臂摆 25–30 个姿态，求出相机在 world 中的位姿。

### B.1 标定板

本实验室的板已打印，实测方格 0.035 m、码 0.026 m：

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/make_board.py --out calib/board --set-measured 0.0350 0.0260
```

重新打印时：先运行 `python3 calibration/make_board.py --out calib/board`，把 `calib/board/charuco_print.pdf` 按 100% 打印、
贴在硬板上，用卡尺量方格和码的边长（米），再用上面的 `--set-measured` 写回。

### B.2 固定板

可以拆下右手，把板刚性固定在法兰上（臂在起始姿态时板面朝向相机；全程不能动）。

### B.3 启动机器人（只读）和相机驱动

FANUC：急停已释放、不开 Free hand teaching、示教器点动（建议 10–30%）。`read_only:=true` 只读关节状态、发布 TF，不发运动指令。

```bash
# 终端 L-real：右臂真机只读 + 右手 mock（手拆下或装着都可以用）
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
ros2 launch bimanual_manipulation single_arm_hand.launch.py prefix:=right_ crx_mock:=false sharpa_backend:=mock read_only:=true rviz:=true
```

预期：示教器点动时 RViz 里的右臂跟着动。

```bash
# 终端 K1：相机驱动（D415 序列号 347322060270）
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
ros2 launch realsense2_camera rs_launch.py serial_no:="'347322060270'" rgb_camera.color_profile:=1280,720,15 align_depth.enable:=true
```

预期：日志里出现序列号并开始出流。换了相机时，在相机驱动启动前用终端 C 查序列号：
`python3 -c "from calibration.camera import list_devices; print(*list_devices(), sep='\n')"`。

### B.4 采集

```bash
# 终端 K2
source /opt/ros/jazzy/setup.bash
source ~/bodex_sharpa_deploy/ws_calib/install/setup.bash
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
ros2 launch industrial_calibration_ros data_collection.launch.xml static_camera:=true \
    camera_mount_frame:=world target_mount_frame:=right_ee_mount \
    camera_frame:=camera_color_optical_frame target_frame:=charuco_board \
    image_topic:=/camera/camera/color/image_raw config_file:=$PWD/calib/board/ic_target.yaml \
    save_path:=$PWD/calib/ic_data
```

示教器点动右臂（绕不同方向转 15–35°、位置变化 ±10 cm，板占画面 1/5–1/2），每次停稳后在 RViz 面板点采集，或在终端 K3（环境同 K2）：

```bash
ros2 service call /trigger std_srvs/srv/Trigger
```

K2 打印 `Observation added` 才算一个样本。采 25–30 个后保存：

```bash
ros2 service call /save std_srvs/srv/Trigger
```

预期：返回 `success: True, message: '.../calib/ic_data/<日期时间>'`。

### B.5 生成配置（内参 + 板 + 初值）

```bash
# 终端 T-real（相机驱动保持运行）
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
source ~/bodex_sharpa_deploy/venv/bin/activate
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/ic_config.py --data calib/ic_data/<日期时间> --out calib/ic_config.yaml
```

预期：`board found in N/N images`，`saved calib/ic_config.yaml`。

### B.6 求解

```bash
# 终端 K3
source /opt/ros/jazzy/setup.bash
source ~/bodex_sharpa_deploy/ws_calib/install/setup.bash
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
ros2 run sharpa_ic_tools ic_extrinsic_cli calib/ic_config.yaml calib/ic_data/<日期时间>/cal_data.yaml calib/ic_result.yaml
```

也可以用官方图形界面 `ros2 run industrial_calibration industrial_calibration_extrinsic_hand_eye_calibration_app`，加载同样的两个文件，
保存为 `calib/ic_result.yaml`。预期：`Optimization converged`，残差 < 1 px（上次 0.54 px）。

### B.7 转换并验证

```bash
# 终端 T-real（环境同 B.5）
python3 calibration/ic_result.py calib/ic_result.yaml --config calib/ic_config.yaml
python3 calibration/verify_extrinsics.py --camera ros --extrinsics calib/camera_extrinsics.json --no-hand    # 手装着时去掉 --no-hand
```

预期：`converged: true`；`calib/verify/overlay.png` 上黄圈落在右臂上；`table.table_z_m` 是桌面（灰色垫子上表面）高度，
执行时自动使用（上次 −0.157 m）。完成后 Ctrl+C 停掉终端 K1，否则后面拍照时相机被占用。

---

# C. 手安装外参（手相对法兰；B 之后、换过手的安装后重做）

用 Sharpa 手的网格去贴合多个视角的深度点云，求法兰→手的变换，结果 `calib/hand_mount.json` 在执行时自动使用。

### C.1 采集 5–8 个视角

拆掉标定板、装回右手。只开 L-real 和 T-real，**不开相机驱动 K1**（脚本自己打开 D415）。手只读：`read_only:=true` 也传给手节点。

```bash
# 终端 L-real：右臂真机只读 + 右手真机只读
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export PYTHONPATH=/opt/sharpa-wave-sdk/python:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib:$LD_LIBRARY_PATH
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
ros2 launch bimanual_manipulation single_arm_hand.launch.py prefix:=right_ crx_mock:=false sharpa_backend:=sharpa_sdk read_only:=true rviz:=true
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
python3 calibration/collect_hand_views.py --out calib/hand_views --table-z -0.157
```

操作：手保持一个形状不动，示教器把右臂移到手完整入镜、离相机 0.5–0.9 m、离桌面 ≥ 10 cm 的位置，每次换一个腕部朝向，停稳按 Enter；
按 `q` 结束。预期每次：`saved calib/hand_views/view_0N (raw depth valid xx%)`。

### C.2 求解

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
FS_ROOT=~/bodex_sharpa_deploy/FoundationStereo ~/bodex_sharpa_deploy/fs_venv/bin/python perception/fs_depth.py calib/hand_views/view_*
python3 calibration/hand_mount_icp.py calib/hand_views/view_* --extrinsics calib/camera_extrinsics.json --out calib/hand_mount.json
```

预期：`verdict` 为 `GOOD`（上次：内点 83%，留一视角 ≤ 0.97 mm / 0.45°）；`view_*/hand_mount_overlay.png` 里紫色的手与真手重合。

---

# D. 真实物体：拍照 → 点云 → DexLearn → 抓取

物体完整放在画面内、不贴画面边缘；放在两臂中线偏右臂 20–25 cm 更容易规划成功（两臂基座在 world y = 0 和 −0.91 m）。
相机驱动 K1 不能开着。

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
bash run_scene.sh scene01
```

| 步骤 | 做什么 | 预期 / 去看 |
|---|---|---|
| ① 拍照 | D415 彩色、红外左右图、RealSense 深度 | `saved captures/scene01: Intel RealSense D415 ...` |
| ② FoundationStereo | 红外左右图算深度 | 约 1–3 s；`fs_vis.png` |
| ③ 物体点云 | 找桌面 → 取桌面以上的物体 | 打印候选团并选中一个；`object_fs/overlay.png` 只有物体是红色；`height_above_table_m` ≈ 物体高度 |
| ④ DexLearn | 1024 点 → 200 个抓取 | 约 2 s（CPU） |
| ⑤ 过滤排序 | 去掉超手关节限位、手碰桌、预抓时穿进物体的，按似然保留前 30 | `grasps/selected_views.png`：红色的手包住灰色物体 |
| ⑥ 转到 world | 用 `calib/camera_extrinsics.json` | 打印下一步的执行命令 |

选物体：桌面以上 0.5–30 cm 的点连成团，去掉碰到画面边缘的（机械臂、人）和水平尺寸 > 0.30 m 的（`--max-size`），取剩下最大的。
选错时在 `captures/scene01/color.png` 上找物体上一点 (u, v)：`bash run_scene.sh scene01 --skip-capture --pixel u v`。
对比 RealSense 自带深度：`--skip-capture --depth raw`。

⑤ 只检查手本身；臂能否够到、离左臂和桌面的距离、下降路径在 E 规划时检查，所以 30 个里通常只有一部分能执行（上次 11/30）。

---

# E. 执行抓取

完整流程：移到抓取位置上方 8 cm → 手保持当前形状竖直下降 → **停一次等 Enter** → 手变成 pregrasp → grasp → squeeze →
保持夹紧抬起 10 cm → 放回 → 张开到 pregrasp → 上撤 8 cm → 右臂回默认姿态 → 右手回零。打印规划后 3 s 自动开始，除那一次 Enter 外连续执行。

默认按顺序尝试 ⑤ 保留的抓取（`--index auto`），用第一个通过全部规划检查的，并打印其余被拒的原因。

```bash
# 终端 L-mock：模拟的双臂双手
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=198
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
ros2 launch bimanual_manipulation bimanual_system.launch.py collision_force_limit:=false
```

```bash
# 终端 T-mock
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
source ~/bodex_sharpa_deploy/venv/bin/activate
export ROS_DOMAIN_ID=198
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 execute_grasp.py --grasp captures/scene01/grasps/grasps_world.npy --lift-place 0.10 --home
python3 execute_grasp.py --grasp captures/scene01/grasps/grasps_world.npy --lift-place 0.10 --home --execute
```

预期：第一条打印 `using grasp index N` 和规划报告，以 `DRY RUN only` 结束；第二条在 RViz 里做完整流程，最后 `Done`。真机：

```bash
# 终端 L-real（可写；等日志里碰撞力限制器 ARMED）
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
python3 execute_grasp.py --grasp captures/scene01/grasps/grasps_world.npy --lift-place 0.10 --home --execute
```

不加 `--home` 时停在松开后的位置，可用部署代码自带的 `ros2 run bimanual_manipulation move_to_default_pose.py --execute` 回默认位。

### execute_grasp.py 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--index` | auto | 指定用第几个抓取；auto = 第一个可行的 |
| `--lift-place DZ` / `--home` | 无 | squeeze 后保持夹紧抬起 DZ 米再放回、松开、上撤 / 最后回默认姿态 |
| `--confirm` | 有 `--lift-place` 时 pregrasp，否则 each | pregrasp：只在下降后按一次 Enter；each：每段前按 Enter；none：不停 |
| `--stop-after` / `--release` | squeeze / 否 | 只做到 pregrasp / grasp；抓完直接张手抬起 |
| `--arm-speed-deg` / `--arm-accel-deg` | 10 / 20 | 每个臂关节 °/s、°/s²；上限 20 °/s |
| `--hand-speed-deg` / `--hand-accel-deg` | 20 / 40 | 每个手关节；上限 30 °/s |
| `--table-z` | `calib/verify/verify.json` 的桌高 | 桌面（灰色垫子上表面）在 world 中的高度 |
| `--clearance` | 0.005 m（world）/ 0.10 m（scene） | 肘部以外的连杆和手必须高于桌面这么多 |
| `--hand-mount` | `calib/hand_mount.json` | C 节结果；`none` 用实验室占位值 |
| `--obstacles` / `--obstacle-clearance` | ⑥ 记录的物体点云 / 0.02 m | 移到抓取上方时手与物体保持 0.02 m，下降时 ≥ 5 mm；`none` 不检查 |
| `--wrist-frame` | 由抓取文件决定 | world：真实物体；scene：把仿真候选放到当前手位置上方 15 cm 的空中测试 |

执行前自动拒绝：已有其他发送端、驱动未订阅（read_only）、碰撞力限制器非 ARMED、规划后机器人动过。
执行中立即停发：反馈 > 0.5 s 不更新、限制器 TRIPPED、跟踪误差超过臂 6° / 手 20°、Ctrl+C。
碰撞检查是近似的（连杆原点和手的网格采样点），臂与环境只检查桌面和左臂（0.2 m）。
