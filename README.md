# sharpa_deploy_bridge：BODex / DexLearn Sharpa 抓取 → 真机（右臂 CRX + 右手 Sharpa）

与实验室部署工作空间 `~/ws_fanuc` **完全解耦**：不放进 `src/`、不 colcon 编译、不改任何部署代码，只通过运行中的
`bimanual_manipulation` launch（mock 或真机）的公开话题通信。所有会动机器人的命令默认只规划不动（dry run），
加 `--execute` 后每一段都要按 Enter；Ctrl+C 停止发送（**不是急停**，手放在示教器急停上）。

## 整体流程与解耦

```
① perception/capture.py       D435i 拍一组：彩色、红外左右图、RealSense 深度             不需要机器人/标定/GPU
② perception/fs_depth.py      红外左右图 → FoundationStereo 深度                         GPU（fs_venv）
③ perception/object_cloud.py  深度 → 桌面平面 → 物体点云（桌面坐标系，z 向上）            不需要标定；有标定时再给 world
④ grasping/dexlearn_infer.py  物体点云 → DexLearn（V3/S34 50k）→ 200 个抓取（桌面坐标系）  DexLearn 环境（服务器）
⑤ grasping/select_grasps.py   关节限位 / 桌面 / 预抓穿透过滤 → 按似然排序 → 可视化          纯 numpy
⑥ grasping/to_world.py        桌面坐标系 → world（需相机标定）→ execute_grasp.py          需相机标定 + 手安装外参
   execute_grasp.py           右臂 IK + 慢速轨迹 + 安全检查 → 执行                        需机器人（先 mock）

标定：calibration/（相机外参：眼在手外）  calibration/hand_mount_icp.py（手安装外参：深度 + 手模型配准）
```

没有标定也能做 ①–⑤：看点云（`overlay.png`、`scene.ply`、`object_views.png`）和生成的抓取（`selected_views.png`）。

## 终端类型（每个代码块开头都写全了环境；也可用 `source env.sh mock|real [launch]` 代替）

| 类型 | 用途 | 环境 |
|---|---|---|
| L-mock / L-real | 启动部署 launch | ROS + ws_fanuc（+ Sharpa SDK）+ 域号，**不激活 venv** |
| T-mock / T-real | 本仓库会读/写机器人话题的工具 | 同上 + `~/bodex_sharpa_deploy/venv` |
| C | 只用相机/离线计算（拍摄、点云、选抓取、离线检查） | 只激活 venv |
| F | FoundationStereo | `fs_venv` + `FS_ROOT` |
| S | DexLearn 推理（服务器） | `dexlearn` conda 环境 |

---

# 0. 一次性安装（部署主机）

```bash
# 代码
cd ~/bodex_sharpa_deploy && git clone https://github.com/MrSingleFXE/sharpa_deploy_bridge.git   # 已 clone 的用 git pull
# 工具 venv（继承系统 ROS 包；OpenCV/RealSense/scipy/matplotlib 装在 venv 里）
python3 -m venv --system-site-packages ~/bodex_sharpa_deploy/venv
~/bodex_sharpa_deploy/venv/bin/pip install "numpy<2" "opencv-contrib-python==4.10.0.84" pyrealsense2 pillow scipy matplotlib pyyaml
# FoundationStereo（RTX 5090 = Blackwell：需 PyTorch ≥ 2.7 + CUDA 12.8，驱动 ≥ 570；不装 xformers/flash-attn）
git clone https://github.com/NVlabs/FoundationStereo.git ~/bodex_sharpa_deploy/FoundationStereo
python3 -m venv ~/bodex_sharpa_deploy/fs_venv
~/bodex_sharpa_deploy/fs_venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
~/bodex_sharpa_deploy/fs_venv/bin/pip install omegaconf timm einops scipy opencv-python-headless imageio pillow gdown "numpy<2"
mkdir -p ~/bodex_sharpa_deploy/FoundationStereo/pretrained_models && cd ~/bodex_sharpa_deploy/FoundationStereo/pretrained_models
~/bodex_sharpa_deploy/fs_venv/bin/gdown --folder https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf
ls */    # 23-51-11/（ViT-L，默认）和 11-33-40/（ViT-S，更快）各有 cfg.yaml + model_best_bp2.pth
```

- gdown 失败就浏览器打开该链接手动下载文件夹放到 `pretrained_models/`。官方文件 sha256：23-51-11 `60e79bde…c5f1`，
  11-33-40 `8d7850b9…e6be`（`sha256sum */model_best_bp2.pth` 核对）。
- FoundationStereo 许可为 NVIDIA 研究/非商业用途。推理时完全离线（包装脚本让 DINOv2 用仓库自带代码、timm 骨干不联网下载、
  open3d 等推理用不到的模块用空模块替代）。
- RealSense 走 USB，主机同时接着 D405 时脚本默认只选 D435I（不唯一就拒绝并列出序列号，用 `--serial` 指定）。
  `pyrealsense2` 读相机需要 udev 规则；报权限/找不到设备时请管理员装 librealsense 规则（或 `ros-jazzy-realsense2-camera`）。

自检（都不需要相机和机器人）：

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/test_offline.py && python3 perception/test_offline.py && python3 calibration/test_hand_mount.py
```

```bash
# 终端 F
export FS_ROOT=~/bodex_sharpa_deploy/FoundationStereo
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
~/bodex_sharpa_deploy/fs_venv/bin/python perception/fs_depth.py --demo captures/fs_demo   # 期望约 1–2 s、有效 >85%
```

---

# A. 执行一个已有候选（已在真机验证：空中、高处）

```bash
# 终端 L-mock（先 mock）
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
python3 execute_grasp.py --grasp examples/master_chef_can_pose000_14_grasp.npy --release            # 只规划
python3 execute_grasp.py --grasp examples/master_chef_can_pose000_14_grasp.npy --release --execute  # 执行
```

真机：launch 加 `crx_mock:=false sharpa_backend:=sharpa_sdk`，保留碰撞力限制并等 `ARMED`。

```bash
# 终端 L-real
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
python3 execute_grasp.py --grasp examples/master_chef_can_pose000_14_grasp.npy --stop-after pregrasp --clearance 0.2 --anchor-offset 0 0 0.25
```

### execute_grasp.py 主要参数（默认已很慢）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--arm-speed-deg / --arm-accel-deg` | 5 / 10 | 每个臂关节 °/s、°/s²；> 20 °/s 拒绝 |
| `--hand-speed-deg / --hand-accel-deg` | 10 / 20 | 每个手关节；> 30 °/s 拒绝 |
| `--rate` | 100 Hz | 与 launch 的 `input_rate_hz` 一致 |
| `--wrist-frame` | scene | scene：把候选放到"当前手位置 + 偏移"的空中；world：候选已在 world 坐标（⑥ 的输出） |
| `--table-z` / `--clearance` | 0 / 0.10 m | 肘部（J3）以外及手的连杆原点必须高于桌面这么多（第 1 段只要求不低于起点） |
| `--hand-mount` | 无 | 标定的手安装外参（只替换规划模型里的法兰→手变换，launch 不用改） |
| `--obstacles` / `--obstacle-clearance` | 无 / 0.02 m | 物体点云（world）；到达 pregrasp 之前手与物体保持距离 |
| `--stop-after` / `--release` | squeeze / 否 | 只做到某一步；做完后张手并抬起 |

执行前自动拒绝：已有其他发送端、驱动未订阅（read_only）、碰撞力限制器非 ARMED、规划后机器人动过。
执行中停止：反馈 > 0.5 s 不更新、限制器 TRIPPED、跟踪误差（臂 6°、手 20°）、Ctrl+C。

---

# B. 相机标定（眼在手外，只用右臂）

原理：标定板刚性固定在右臂末端（J6 之后，或手背），右臂走 15–25 个姿态；每个姿态有法兰位姿 A_i（正运动学）和板在相机里的位姿
B_i（PnP），解 `A_i·Y = X·B_i` 得到 X = 相机在 world（左臂基座）中的位姿（Y 是板相对法兰，顺带求出）。只用彩色图。

1. 打印标定板（终端 C）：

```bash
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/make_board.py --out calib/board                                  # 5x7 格、30 mm 方格、A4
# 100% 打印、贴硬板；卡尺量一个方格和一个码的边长（米），写回：
python3 calibration/make_board.py --out calib/board --set-measured 0.0300 0.0220
```

2. 相机与识别检查（终端 C，标定板拿在手里放到右手大致位置，0.5–1.0 m）：

```bash
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/camera_check.py --board calib/board/board.json
```

期望：列出所有 RealSense 并自动选中 D435I、`detected 10/10`、`reproj` < 0.5 px、抖动 < 1–2 mm、USB 3。

3. 标定板刚性固定（螺丝/夹具；不能装在手指上，不能在 J5 及以前的连杆上）；`--clearance` 用 0.15 m 以上。

4. 采集。方式 A 手动（只读 launch + 示教器点动，脚本不发运动命令）：

```bash
# 终端 L-real（只读）
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

每到一个姿态、臂停稳后按 Enter；q 结束。需要绕至少两个不同方向的转动，15–25 个样本。
方式 B 自动（可写 launch，脚本在起点附近 ±6 cm、转 15–30° 生成 15 个姿态，5°/s，每步 Enter，最后回到起点）：
同一终端 T-real 运行 `python3 calibration/collect_handeye.py --out calib/run1 --board calib/board/board.json --mode auto --clearance 0.15`
（先不加 `--execute` 看规划，再加 `--execute`）。

mock 演练（终端 L-mock + T-mock，环境同 A）：
`python3 calibration/collect_handeye.py --out calib/sim_run --board calib/board/board.json --camera sim --sim-truth calibration/sim_truth_example.json --mode auto --execute`

5. 求解与验证：

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/solve_handeye.py calib/run1 --out calib/camera_extrinsics.json
```

结论 GOOD（平移 RMS ≤ 3 mm、留一法工作区偏移 ≤ 2 mm、≥ 12 样本、转轴差异 ≥ 45°）/ USABLE / REDO。

```bash
# 终端 T-real（只读，不动机器人）：叠加图 + 板复核 + 桌面高度
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
source ~/bodex_sharpa_deploy/venv/bin/activate
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/verify_extrinsics.py --extrinsics calib/camera_extrinsics.json --board calib/board/board.json
```

`calib/verify/overlay.png`：黄圈（臂）应落在臂上；紫圈（手）整体偏移说明手安装外参还没标；`table.table_z_m` 是桌面高度。

---

# C. 手安装外参（法兰 right_ee_mount → 手的安装面 right_hand_flange）——不用尺子量

BODex/DexLearn 的手腕是 `right_hand_C_MC`；它在 Sharpa 模型里位于手安装面上方 29.5 mm（已知），**未知的只有"手安装面相对机械臂
法兰"的变换**（`hand_mounts.yaml` 现在平移为 0、旋转为占位）。方法：相机标好后，深度相机能看到手在 world 里的真实位置，关节角能算出
法兰的位置，两者之间的固定变换就是安装外参。用手自己的 CAD 网格去配准深度（ICP），多个姿态联合求解，并从多个绕法兰轴的转角出发，
避免卡在错误解。服务器合成测试：从偏 13 mm / 40° 的占位值出发，恢复到 0.36 mm / 0.09°。

1. 采集（只读 launch 同 B-4；手保持一个姿态，例如零位张开；5–8 个姿态，手完整入镜、离桌面和其他物体远一些、腕部朝向尽量不同）：

```bash
# 终端 T-real（只读，脚本不发运动命令）
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export PYTHONPATH=/opt/sharpa-wave-sdk/python:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib:$LD_LIBRARY_PATH
source ~/bodex_sharpa_deploy/venv/bin/activate
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/collect_hand_views.py --out calib/hand_views --table-z <B-5 得到的桌面高度>
```

2. FoundationStereo 深度：

```bash
# 终端 F
export FS_ROOT=~/bodex_sharpa_deploy/FoundationStereo
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
~/bodex_sharpa_deploy/fs_venv/bin/python perception/fs_depth.py calib/hand_views/view_*
```

3. 求解：

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/hand_mount_icp.py calib/hand_views/view_* --extrinsics calib/camera_extrinsics.json --out calib/hand_mount.json
```

结论 GOOD（平均内点 > 70%、留一视角变化 < 2 mm / 1°、≥ 5 视角）/ USABLE / CHECK；看 `view_*/hand_mount_overlay.png`
（紫色 = 拟合的手，应与图里的手重合）。之后 `execute_grasp.py --hand-mount calib/hand_mount.json` 使用它
（不用改实验室的 `hand_mounts.yaml`；若想让 RViz 也显示正确，可把同样的 xyz/rpy 写到自己的 yaml，launch 时加 `hand_mounts:=/路径`）。

---

# D. 面对真实物体：点云 → DexLearn → 抓取 → 执行

物体放在桌上右手够得到、相机看得清的位置；机械臂不要挡住物体。

① 拍摄（终端 C，不需要 ROS）：

```bash
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 perception/capture.py --out captures/scene01
```

② FoundationStereo（终端 F）：

```bash
export FS_ROOT=~/bodex_sharpa_deploy/FoundationStereo
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
~/bodex_sharpa_deploy/fs_venv/bin/python perception/fs_depth.py captures/scene01
```

看 `captures/scene01/fs_vis.png`（红外 | FoundationStereo | RealSense）。可选对比（终端 C）：
`python3 perception/compare_depth.py captures/scene01 --roi x0 y0 x1 y1 [--extrinsics calib/camera_extrinsics.json --box ...]`。

③ 物体点云（终端 C；**没有标定也可以**，有标定再加 `--extrinsics`）：

```bash
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 perception/object_cloud.py captures/scene01 --depth fs --pixel <u v 物体上一个像素>
# 标定后：再加 --extrinsics calib/camera_extrinsics.json （输出 object_world.npy 和 T_world_table）
```

看 `object_fs/overlay.png`（红色 = 选中的物体）、`object_views.png`、`scene.ply`（MeshLab/CloudCompare 打开）。
桌面坐标系：z = 桌面法向（向上），x = 相机视线在桌面上的投影，原点 = 物体中心在桌面上的投影。

④ DexLearn 推理（DexLearn 需要 MinkowskiEngine 环境，目前在**服务器**；主机 5090 上没有装）。在你的 Mac 上中转文件，
先在 Mac 终端设置（值见你本地的项目笔记）：

```bash
DEPLOY=<部署主机 用户@IP>
SERVER=<服务器 ssh 别名>
SERVER_BRIDGE=<服务器上本仓库的路径>
SERVER_CONDA=<服务器 miniforge 目录>
SERVER_DEXLEARN=<服务器上 DexLearn 仓库路径>
```

```bash
# Mac：主机 → 服务器
ssh $SERVER "mkdir -p $SERVER_BRIDGE/captures/scene01/object_fs"
scp $DEPLOY:~/bodex_sharpa_deploy/sharpa_deploy_bridge/captures/scene01/object_fs/object_table.npy /tmp/object_table.npy
scp /tmp/object_table.npy $SERVER:$SERVER_BRIDGE/captures/scene01/object_fs/
```

```bash
# 终端 S（服务器；先 ssh $SERVER，把上面几个变量也在服务器终端里设置一次）
source $SERVER_CONDA/bin/activate dexlearn
export DEXLEARN_ROOT=$SERVER_DEXLEARN
export SHARPA_HAND_URDF=<服务器上 dual_sharpa_wave_ros2/urdf/right_sharpa_wave_with_flange.urdf>
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1
cd $SERVER_BRIDGE
python grasping/dexlearn_infer.py captures/scene01/object_fs/object_table.npy --out captures/scene01/grasps
```

```bash
# Mac：服务器 → 主机
ssh $DEPLOY "mkdir -p ~/bodex_sharpa_deploy/sharpa_deploy_bridge/captures/scene01/grasps"
scp $SERVER:$SERVER_BRIDGE/captures/scene01/grasps/grasps_table.npy /tmp/grasps_table.npy
scp /tmp/grasps_table.npy $DEPLOY:~/bodex_sharpa_deploy/sharpa_deploy_bridge/captures/scene01/grasps/
```

做法与训练一致：物体平移到训练场景中心 (0.753, 0.004)、随机取 1024 点、5 mm 体素、采样 200 个，再平移回来。

⑤ 过滤、排序、可视化（终端 C）：

```bash
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 grasping/select_grasps.py captures/scene01/grasps/grasps_table.npy captures/scene01/object_fs/object_table.npy
```

看 `grasps/selected_views.png`（灰 = 物体，红 = 手在 grasp 姿态）和 `selection.json`（每个候选的拒绝原因）。

⑥ 转到 world 并执行（需要 B 相机标定 + C 手安装外参；③ 要带 `--extrinsics` 重新跑一次）：

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 grasping/to_world.py captures/scene01/grasps/selected.npy captures/scene01/object_fs/object.json
```

它会打印执行命令。先 mock（终端 L-mock + T-mock，看规划是否可达），再真机（L-real + T-real）；第一次只做到 pregrasp：

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
    --table-z <object.json 里的 table_z_world_m> --clearance 0.005 \
    --obstacles captures/scene01/object_fs/object_world.npy --hand-mount calib/hand_mount.json --stop-after pregrasp
```

某个候选规划失败（够不到/IK/离桌）就换 `--index 1, 2, …`。

---

# 服务器上的验证（2026-10-14，没有真机/相机）

| 部分 | 验证 | 结果 |
|---|---|---|
| 相机标定 | 渲染板识别 + 含离群的求解（OpenCV 4.10 / 5.0）；mock + 模拟相机全流程 | 相机位姿 2.6 mm / 0.19°，手工作区误差 0.2 mm，结论 GOOD |
| FoundationStereo | 官方 ViT-S 权重、Python 3.12 + torch 2.7、无 xformers、离线 | 官方示例 960x540：2.0 s（A800），有效 90.7% |
| 点云 | 合成桌面+盒子：IR→彩色配准、桌面、物体（无标定 / 有标定） | 配准中位误差 0.28 mm；高度 0.121 m（真值 0.12） |
| 真实双目图全链路 | 官方示例（桌上的杯子）：② → ③（无标定）→ ④ → ⑤ | 只选中杯子（12 cm 高）；200 个抓取 143 个通过过滤 |
| DexLearn 适配器 | 100 个测试场景 × 100 候选，Bench 物理评测（kp12 协议） | 适配器 53.8%，同场景原流程 56.1%，只改随机性的对照 54.4% → 差异在采样噪声内 |
| 手安装外参 | 合成：6 视角、1.5 mm 深度噪声、臂杂点，从偏 13 mm / 40° 出发 | 恢复误差 0.36 mm / 0.09° |

未在真机/真相机上验证：RealSense 读取、手动采集模式、真实场景的点云质量与抓取成功率。

# 已知限制

- 训练用的是**模拟单视角点云**（相机约 0.74 m，Azure Kinect 模拟）；真实 D435i 视角/距离/噪声不同，属于模拟到真实的差距，需要实测。
- 执行时手从 pregrasp 正上方 8 cm 竖直下降；侧面抓取在物体很高时可能需要别的接近方向。
- 桌面/物体检查用的是连杆原点（执行器）和碰撞网格点（选抓取），不是完整的网格碰撞检测；臂与环境的碰撞只检查了左臂距离。
- DexLearn 推理目前在服务器；主机 5090 上装 MinkowskiEngine（CUDA 12.8）还没做。
