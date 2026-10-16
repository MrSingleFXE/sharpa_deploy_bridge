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
| **K** | 相机 ROS 驱动 + industrial_calibration 标定 | ROS + `~/bodex_sharpa_deploy/ws_calib`，**不激活 venv** |

mock 用域号 198 且只在本机；真机域号 87。FoundationStereo 和 DexLearn 各有自己的 Python，命令里直接写了它们的路径。

---

# 0. 第一次使用：安装（只需做一次，全部在 `~/bodex_sharpa_deploy` 下；只有 0.4 标定工具需要 sudo）

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

### 0.4 相机标定工具：ROS-Industrial industrial_calibration（约 10–20 分钟，需要 sudo）

```bash
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
bash install/install_calibration.sh
```

做了什么：apt 安装 RealSense ROS 2 驱动（`realsense2_camera`）、vcstool、rosdep；在**独立的** colcon 工作空间
`~/bodex_sharpa_deploy/ws_calib`（不碰 `ws_fanuc`）编译 industrial_calibration 1.2.0（核心库 + 手眼标定图形界面）、
industrial_calibration_ros2 1.2.0（ROS 2 采集节点 + RViz 面板）以及本仓库的 `ic_extrinsic_cli`（同一套算法的命令行版）。
运行前先 `deactivate` 掉 venv（脚本会检查）。
预期最后：打印 `industrial_calibration_ros`、`realsense2_camera` 的安装路径、`industrial_calibration_extrinsic_hand_eye_calibration_app`、
`ic_extrinsic_cli` 和 `DONE`。

### 0.5 训练好的模型文件

模型（189 MB，不放在 GitHub）放在 `~/bodex_sharpa_deploy/models/dexlearn_sharpa_v3_s34_50k/`，内有 `config.yaml`、
`step_050000.pth`、`MANIFEST.json`。检查：

```bash
cd ~/bodex_sharpa_deploy/models/dexlearn_sharpa_v3_s34_50k && sha256sum config.yaml step_050000.pth
```

预期：`45ff30f4…7ea7  config.yaml` 和 `0c5e8ca8…fd9e  step_050000.pth`（与 MANIFEST.json 一致）。
这就是 V3 + 34 球 + kp12 + α1、8 卡 50k 步的模型（测试集全候选物理成功率 57.87%，似然 Top-10 75.88%）。
这个文件夹不在公开仓库里，由我们单独拷贝到主机；目录不存在或校验不一致时先不要往下做。

### 0.6 自检（不需要相机和机器人，约 2 分钟）

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

# B. 相机标定：ROS-Industrial industrial_calibration + ChArUco（相机在 world 中的位置；只用右臂）

**为什么**：相机看到的点要换算到机械臂坐标系才能去抓。**怎么做**：ChArUco 板刚性固定在右臂末端，右臂摆 20–30 个姿态；
industrial_calibration 的采集节点在每个姿态记录一张图 + TF 给出的"world → 右臂法兰 `right_ee_mount`"位姿；
它的手眼标定（`static_camera: true`）求出 camera mount（= world）→ 相机 和 target mount（= 法兰）→ 板。
服务器上用 industrial_calibration 1.2.0 的同一套优化跑合成数据：相机位姿误差 0.10 mm / 0.05°（从偏 5 cm / 10° 的初值也收敛到同一解）。

### B.1 打印 ChArUco 板

```bash
# 终端 C
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/make_board.py --out calib/board
```

预期：`ChArUco 5 cols x 7 rows, square 30.0 mm, marker 22.0 mm, DICT_5X5_100 ...`。把 `calib/board/charuco_print.pdf`
按 **100% 实际尺寸**打印，平整贴在硬板上；用卡尺量一个方格和一个码的边长（米），写回（同时更新 industrial_calibration 的
目标配置 `calib/board/ic_target.yaml`）：

```bash
python3 calibration/make_board.py --out calib/board --set-measured 0.0300 0.0220     # 换成实测值
```

### B.2 固定板

刚性固定（螺丝/夹具）在 J6 之后（转接件或手背），臂在起始姿态时板面朝向相机；不能装在手指上或 J5 及以前的连杆上。全程不能动。

### B.3 启动机器人（只读）和相机 ROS 驱动

```bash
# 终端 L-real（只读：PC 只读状态、发布 TF，不控制机械臂）
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
export PYTHONPATH=/opt/sharpa-wave-sdk/python:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib:$LD_LIBRARY_PATH
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
ros2 launch bimanual_manipulation bimanual_system.launch.py crx_mock:=false sharpa_backend:=sharpa_sdk read_only:=true rviz:=true
```

D435I 的序列号（主机上同时有 D405，必须指定）：

```bash
# 终端 C（相机驱动启动前运行；会列出所有 RealSense）
source ~/bodex_sharpa_deploy/venv/bin/activate
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 -c "from calibration.camera import list_devices; print(*list_devices(), sep='\n')"
```

```bash
# 终端 K1：相机驱动（彩色 1280x720 + 对齐到彩色的深度）
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
ros2 launch realsense2_camera rs_launch.py serial_no:="'<D435I 序列号>'" rgb_camera.color_profile:=1280,720,15 align_depth.enable:=true
```

预期：`/camera/camera/color/image_raw`、`/camera/camera/color/camera_info` 有数据（`ros2 topic hz /camera/camera/color/image_raw` ≈ 15）。
若报 `rgb_camera.color_profile` 参数不认识，去掉这个参数（用默认分辨率也行，后面的内参直接从 camera_info 读）。
检查彩色图无畸变：`ros2 topic echo --once /camera/camera/color/camera_info | grep -A5 "^d:"` 应全为 0（D435i 彩色流通常如此）。

### B.4 采集（industrial_calibration_ros 的数据采集流程）

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

预期：RViz 打开，`/image_annotated` 里能看到板被识别（角点画出）。操作：示教器点动右臂（含绕不同方向转 15–35°、位置变化 ±10 cm，
板占画面 1/5–1/2），**每停稳一次**，在 RViz 面板点采集，或在另一个终端 K3（环境同 K2）运行：

```bash
ros2 service call /trigger std_srvs/srv/Trigger
```

终端 K2 打印 `Observation added` 才算一个样本（板没识别到就不会加）。采 25–30 个后保存：

```bash
ros2 service call /save std_srvs/srv/Trigger
```

预期：返回 `success: True, message: '.../calib/ic_data/<日期时间>'`，该目录下有 `cal_data.yaml`、`images/`、`poses/`。

### B.5 生成标定配置（内参 + 板 + 初值）

```bash
# 终端 T-real（相机驱动保持运行：从 camera_info 读内参）
source /opt/ros/jazzy/setup.bash
source ~/ws_fanuc/install/setup.bash
source ~/bodex_sharpa_deploy/venv/bin/activate
export ROS_DOMAIN_ID=87
unset ROS_AUTOMATIC_DISCOVERY_RANGE
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
python3 calibration/ic_config.py --data calib/ic_data/<日期时间> --out calib/ic_config.yaml
```

预期：`board found in N/N images`；`initial camera guess in world` 在两臂之间、比基座高；`saved calib/ic_config.yaml`。
（初值只是给 industrial_calibration 的起点；真正的求解在下一步。）

### B.6 求解（industrial_calibration）

图形界面（官方）：

```bash
# 终端 K3
source /opt/ros/jazzy/setup.bash
source ~/bodex_sharpa_deploy/ws_calib/install/setup.bash
ros2 run industrial_calibration industrial_calibration_extrinsic_hand_eye_calibration_app
```

在界面里加载配置 `calib/ic_config.yaml` 和数据 `calib/ic_data/<日期时间>/cal_data.yaml`，运行标定，"Save calibration..." 保存为
`calib/ic_result.yaml`。或者用同一套算法的命令行版（终端 K3）：

```bash
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
ic_extrinsic_cli calib/ic_config.yaml calib/ic_data/<日期时间>/cal_data.yaml calib/ic_result.yaml
```

预期：`using N / N observations`、`Optimization converged`、残差（GUI 显示的像素误差，或 `Final cost per observation` 开方）
通常 < 1 px；3D 重投影误差几毫米以内；没有大于 0.8 的相关系数（有则说明姿态变化不够，补采）。

### B.7 转成本仓库用的外参文件并验证

```bash
# 终端 T-real（环境同 B.5）
python3 calibration/ic_result.py calib/ic_result.yaml --config calib/ic_config.yaml
python3 calibration/verify_extrinsics.py --camera ros --extrinsics calib/camera_extrinsics.json
```

预期：`converged: true`；`camera_position_world_m` 在两臂之间、比基座高；`camera_viewing_direction_world` 的 z 为负；
`calib/verify/overlay.png` 上黄圈落在右臂各关节上（紫圈是手，C 节之前可能整体偏几厘米）；`table.table_z_m` 是桌面高度（记下来）。
标定完成后 **Ctrl+C 停掉相机驱动（终端 K1）**，否则后面 `run_scene.sh` 拍照时相机被占用。

---

# C. 手安装外参（手相对法兰的位置；不用尺子量）

**为什么**：DexLearn 给出的是手掌根部 `right_hand_C_MC` 的位置，机械臂要的是法兰的位置；两者之间差一个"手怎么装在法兰上"
的固定变换，现在实验室配置里是占位值（平移 0）。**原理**：相机标好后，深度能看到手的真实位置，关节角给出法兰位置；用 Sharpa
手自己的 CAD 网格去贴合多个视角的深度点云，求出这个变换。仿真：从偏 13 mm / 40° 的初值恢复到 0.4 mm / 0.1°。

### C.1 采集 5–8 个视角（只读 launch 同 B.3；相机驱动已停（K1）；拆掉标定板；手保持一个姿态，例如张开）

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
python3 calibration/collect_hand_views.py --out calib/hand_views --table-z <B.7 的 table_z_m>
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
| 相机标定：industrial_calibration 1.2.0 + ChArUco（合成数据，采集器同格式，25 张） | 相机 0.10 mm / 0.05°，手工作区 0.6 mm，残差 0.14 px；从偏 5 cm / 10° 的初值也收敛到同一解；OpenCV 4.10 打印的板被 4.6 识别且坐标系一致 |
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
