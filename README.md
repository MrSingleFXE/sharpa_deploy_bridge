# sharpa_deploy_bridge：BODex Sharpa 抓取 → 真机部署系统（右臂 + 右手）

与部署 ROS 工作空间 `~/ws_fanuc` **完全解耦**：不放进 `src/`、不需要 colcon 编译，不改任何部署代码。
只通过已运行的 `bimanual_manipulation` launch（mock 或真机）的公开话题通信。

```
输入  BODex 成功候选 *_grasp.npy（pregrasp/grasp/squeeze 各 29D：C_MC 手腕 xyz+wxyz + 22 关节）
处理  读当前臂/手状态和 /bimanual/robot_description
      → 把 BODex 场景放到空中（抓取手腕落在“当前手位置 + 15 cm”处，绕 z 选最省动作的朝向）
      → 右臂 IK（tip = right_hand_C_MC）→ 限速轨迹 → 桌面高度 / 关节限位 / 与左臂距离检查
输出  /crx5ia/joint_targets（right_J1..J6）和 /sharpa/right_hand/joint_command（22 关节），100 Hz
```

执行顺序：① 臂到 pregrasp 正上方 8 cm（关节空间）→ ② 手张到 pregrasp → ③ 竖直下降到 pregrasp →
④ 手腕+手指到 grasp → ⑤ 到 squeeze →（`--release`）手张回 pregrasp、竖直抬起 8 cm。
默认每一步前都要按 Enter。

目录：
- `execute_grasp.py` / `offline_check.py` / `bridge/`：抓取执行（下面"用法"）
- `calibration/`：相机（RealSense D435i，固定在两臂中间高处）与右臂的手眼标定（见文末"相机标定"）
- `tools/`：服务器 mock 与录制分析工具

## 依赖

部署主机已有的 ROS 2 Jazzy Python（rclpy、sensor_msgs、std_msgs）和 numpy。不需要额外安装。

## 用法（部署主机）

终端 1：先 mock

```bash
source /opt/ros/jazzy/setup.bash && source ~/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=198 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST   # mock 用隔离域
ros2 launch bimanual_manipulation bimanual_system.launch.py collision_force_limit:=false
```

终端 2：同样 source 和同一个 ROS_DOMAIN_ID

```bash
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
# 1) 只规划，不动（默认）：打印并保存 grasp_plan.json
python3 execute_grasp.py --grasp examples/master_chef_can_pose000_14_grasp.npy --release
# 2) 执行（每阶段按 Enter）
python3 execute_grasp.py --grasp examples/master_chef_can_pose000_14_grasp.npy --release --execute
```

mock 通过后再上真机：终端 1 改为真机 launch（`ROS_DOMAIN_ID=87`，`crx_mock:=false sharpa_backend:=sharpa_sdk`，
保留碰撞力限制并等到 `ARMED`），终端 2 的命令不变。建议先 `--stop-after pregrasp` 只做到 pregrasp。

回到默认姿态（部署代码自带）：`ros2 run bimanual_manipulation move_to_default_pose.py --execute`

## 速度与安全参数（默认已经很慢）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--arm-speed-deg / --arm-accel-deg` | 5 °/s / 10 °/s² | 每个臂关节；脚本拒绝 > 20 °/s |
| `--hand-speed-deg / --hand-accel-deg` | 10 °/s / 20 °/s² | 每个手关节；脚本拒绝 > 30 °/s |
| `--min-duration` | 3 s | 每段最短时间 |
| `--rate` | 100 Hz | 必须与 launch 的 `input_rate_hz`（默认 100）一致 |
| `--table-z` | 0.0 m | 桌面在 world 中的高度（world = 左臂基座；暂定与基座同高） |
| `--clearance` | 0.10 m | 右臂从肘部（J3）开始往外的所有连杆原点和手的所有连杆原点（肩部 J2 高度固定，不检查）都必须高于桌面这么多，整条轨迹逐点检查 |
| `--anchor-offset` | 0 0 0.15 | 抓取手腕位置 = 当前手位置 + 偏移；也可用 `--anchor X Y Z` 直接给 world 坐标 |
| `--approach-lift` | 0.08 m | pregrasp 正上方的预备点高度 |
| `--stop-after` | squeeze | 可只做到 pregrasp 或 grasp |
| `--arm-track-tol-deg / --hand-track-tol-deg` | 6° / 20° | 实际反馈偏离命令超过此值立即停发 |

执行前的自动检查：已有其他发送端（键盘/GUI）则拒绝；插值节点或手驱动未订阅（read_only）则拒绝；
碰撞力限制器不是 ARMED/DEGRADED 则拒绝；规划后机器人动过则拒绝。
执行中：反馈超过 0.5 s 不更新、限制器 TRIPPED、跟踪误差超限、Ctrl+C → 立即停止发送。
**停止发送不是急停**：臂会停在最后接受的目标，手保持最后的 MIT 目标。真机测试时手放在示教器急停上。

## 已知限制

- `hand_mounts.yaml` 的平移目前是 0（占位），所以真机手的实际位置与模型会差几厘米；只验证“能按输入执行动作”。
- 桌面检查用的是连杆原点 + 余量，不是网格碰撞；没有检查臂与手、臂与环境的碰撞（除左臂距离）。
- 臂的第一段（到预备点）是关节空间插值，路径不是直线，但逐点做了高度检查。
- `offline_check.py` 可在没有 ROS 的机器上用同一规划器检查（需要合成 URDF 文件）。

---

# 相机标定（RealSense D435i 固定，眼在手外；只用右臂）

## 原理（输入 → 处理 → 输出）

```
输入  打印的 ChArUco 标定板，刚性固定在右手（手背/手腕）上
      右臂走 15–25 个不同姿态；每个姿态记录：法兰位姿 A_i（正运动学，world 坐标）+ 板在相机中的位姿 B_i（PnP）
处理  解 A_i · Y = X · B_i：X = 相机在 world 中的位姿，Y = 板相对法兰的位姿（顺带求出，不需要量）
      OpenCV 4 种初值方法取最好 → 联合非线性优化 → 剔除离群样本 → 残差报告
输出  calib/camera_extrinsics.json：T_world_cam（world = 部署 URDF 的 world，即左臂基座；相机光学坐标系
      x 右 y 下 z 前），质量结论 GOOD / USABLE / REDO；verify 再给出桌面高度 table_z
```

用法兰（`right_ee_mount`）而不是手的坐标系，所以手的安装外参还没标定也不影响相机标定。
所有脚本都只读话题；只有 `collect_handeye.py --mode auto --execute` 会动右臂（慢速，每步 Enter），**从不控制手**。

## 0. 一次性准备（部署主机）

```bash
cd ~/bodex_sharpa_deploy/sharpa_deploy_bridge
# OpenCV（带 aruco）和 RealSense Python；用带系统包的 venv，不影响 ROS 的 Python
python3 -m venv --system-site-packages ~/bodex_sharpa_deploy/venv
~/bodex_sharpa_deploy/venv/bin/pip install "numpy<2" "opencv-contrib-python==4.10.0.84" pyrealsense2 pillow
source ~/bodex_sharpa_deploy/venv/bin/activate      # 之后每个标定终端都先 source ROS、ws_fanuc，再 activate
python3 calibration/test_offline.py                 # 应输出 ALL OK（不需要相机和机器人）
```

- 相机接 **USB 3** 口。`pyrealsense2` 读相机需要 RealSense 的 udev 规则；若报 "No device connected" 或权限错误，
  请管理员安装 librealsense 的 udev 规则（或 `sudo apt install ros-jazzy-realsense2-camera`，它会一起装好）。
- 不想用 pip 的 pyrealsense2 也可以：管理员装 `ros-jazzy-realsense2-camera`，用 ROS 话题读相机，下文命令加
  `--camera ros`（需用 `ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true` 启动相机；
  两种方式不能同时占用相机）。

## 1. 打印标定板

```bash
python3 calibration/make_board.py --out calib/board        # 5x7 格，30 mm 方格，22 mm 码，A4 可打
```

1. 打印 `calib/board/charuco_print_300dpi.png`（或 PDF），**100% / 实际尺寸**，不要"适应页面"。
2. 平整地贴在硬板（亚克力/铝板/厚泡沫板）上，不能弯、不能起泡。
3. 用卡尺量**一个方格**和**一个码**的边长，写回：

```bash
python3 calibration/make_board.py --out calib/board --set-measured 0.0300 0.0220   # 换成实测值（米）
```

方格尺寸量错 1%，相机距离就错约 1%，这一步不要省。

## 2. 检查相机和识别（不用机器人）

手拿标定板放在右手大概要去的位置（离相机 0.5–1.0 m），运行：

```bash
python3 calibration/camera_check.py --board calib/board/board.json            # 或 --camera ros
```

期望：列出 D435i 设备；`board detected in 10/10 frames`；`reproj` < 0.5 px；`position jitter` < 1–2 mm。
看 `calib/camera_check/camera_check.png`：板上画出了坐标轴。识别不到时检查光照、反光、板是否在画面中。

## 3. 把标定板装到右手上

- 用螺丝/扎带/夹具**刚性**固定在右手手背或手腕外侧，标定全过程不能松动、不能碰到手指。
- 臂在标定起始姿态时，板面大致朝向相机。标定过程中**不控制手**，手保持当前姿态即可。
- 板会伸出手外：桌面余量 `--clearance` 建议用 0.15 m 以上。

## 4. 先 mock 演练脚本（可选，强烈建议第一次做）

mock 里没有真实图像，用模拟相机（按真值渲染标定板）把采集→求解→报告整套流程跑一遍：

```bash
# 终端 1（隔离域）
export ROS_DOMAIN_ID=198 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
ros2 launch bimanual_manipulation bimanual_system.launch.py collision_force_limit:=false
# 终端 2（同一个域）
python3 calibration/collect_handeye.py --out calib/sim_run --board calib/board/board.json \
    --camera sim --sim-truth calibration/sim_truth_example.json --mode auto --execute
python3 calibration/solve_handeye.py calib/sim_run --out calib/sim_extrinsics.json
```

`sim_truth_example.json` 里的 `T_world_cam` 是假设的相机位置（两臂中间、约 0.95 m 高）。服务器 mock 上实测：13 个样本，
结论 GOOD，相机位姿误差 2.6 mm / 0.19°，换算到手的工作区误差 0.2 mm。

## 5. 真机采集（两种方式，任选）

终端 1 启动真机系统（真机域 87）：

```bash
source ~/.bashrc && source ~/ws_fanuc/install/setup.bash && unset ROS_AUTOMATIC_DISCOVERY_RANGE && export ROS_DOMAIN_ID=87
ros2 launch bimanual_manipulation bimanual_system.launch.py crx_mock:=false sharpa_backend:=sharpa_sdk rviz:=true
```

先用示教器/键盘把右臂移到一个**起始姿态**：板在画面中间、离相机 0.6–0.9 m、周围留出 ±10 cm 的空间。

**方式 A：手动（脚本不发任何运动命令，最保守）**

终端 1 改用只读启动（PC 不控制机械臂，只读状态），用示教器点动（关节点动或工具坐标系点动，含转动）移动右臂：

```bash
ros2 launch bimanual_manipulation bimanual_system.launch.py crx_mock:=false sharpa_backend:=sharpa_sdk read_only:=true rviz:=true
# 终端 2：采集
python3 calibration/collect_handeye.py --out calib/run1 --board calib/board/board.json --mode manual
```

每移到一个新姿态、臂停稳后，在终端 2 按 Enter 采集；q 结束。先确认点动时 RViz 里的臂跟着动（说明只读状态流在示教器
操作时仍然正常），否则改用方式 B。也可以在可写启动下用 `ros2 run dual_crx_control keyboard_control.py` 移动，
但键盘只能平移，单靠它得不到转动，样本不够好。示教器操作按实验室规程进行，脚本不检查示教器运动的安全性。

**方式 B：自动（脚本慢速移动右臂，每一步 Enter）**

```bash
# 先只规划，不动：打印每个姿态是否可达、总时长
python3 calibration/collect_handeye.py --out calib/run1 --board calib/board/board.json --mode auto \
    --clearance 0.15 --table-z <桌面高度，未知先用 0>
# 确认后执行（终端里每步 Enter；Ctrl+C 停止发送）
python3 calibration/collect_handeye.py --out calib/run1 --board calib/board/board.json --mode auto \
    --clearance 0.15 --table-z 0 --execute
```

自动模式：在起始姿态周围 ±6 cm、绕手的不同轴转 15–30° 生成 15 个姿态，速度 5°/s，逐点做桌面/左臂/限位检查，
跟踪误差 > 6°、反馈中断、限制器 TRIPPED、Ctrl+C 都立即停止发送；最后回到起始姿态。键盘/GUI 发送端必须先关掉。
`--pos-range`、`--rot-max-deg`、`--n` 可调；某个姿态看不到板时只是跳过该样本。

好样本的要求：至少 15 个；板在画面里占 1/5–1/2；相对相机倾斜 15–35°；**要有绕至少两个不同方向的转动**
（只平移不转动解不出来）；每次采集时臂完全停住。

## 6. 求解

```bash
python3 calibration/solve_handeye.py calib/run1 --out calib/camera_extrinsics.json   # 可以给多个采集文件夹
```

| 结论 | 条件 | 处理 |
|---|---|---|
| GOOD | 平移 RMS ≤ 3 mm，留一法工作区偏移 ≤ 2 mm，≥ 12 样本，转轴差异 ≥ 45° | 可用 |
| USABLE | 平移 RMS ≤ 6 mm，留一法偏移 ≤ 5 mm，≥ 8 样本，转轴差异 ≥ 30° | 可先用，建议补采 |
| REDO | 其他 | 补采更多/转动更大的姿态；检查板是否松动、尺寸是否实测 |

"留一法工作区偏移"：每次去掉一个样本重新求解，看手工作区里的点在 world 中移动多少——这才是抓取关心的精度。
单张图里板的朝向噪声（远处的小板约 0.5–1°）会让 `rms_deg` 偏大，它只报告、超过 1° 时给出警告（建议换 A3 大板），
不直接决定结论。

同时看 `camera_position_world_m` 是否合理（相机应在两臂之间、比基座高）、`camera_viewing_direction_world`
是否朝下朝向工作区；`per_sample` 里个别大残差的样本会被自动剔除。

## 7. 验证（只读，不动机器人）

```bash
python3 calibration/verify_extrinsics.py --extrinsics calib/camera_extrinsics.json --board calib/board/board.json
```

- `calib/verify/overlay.png`：黄圈 = 右臂各关节位置（应落在臂上），紫圈 = 手指尖/手根位置。
  臂对得上而手整体偏几厘米 → 相机标定没问题，是手安装平移（`hand_mounts.yaml`）还没标定。
- `board_check`：当前板的观测与标定预测之差，应 ≲ 3 mm / 0.3°。换一两个姿态再跑一次更可靠。
- `table`：从深度图拟合的桌面高度 `table_z_m`（world 坐标）。若画面里桌面不多，用 `--table-roi x0 y0 x1 y1`
  框出桌面区域。之后执行抓取时用 `--table-z <这个值>`。

完成后拆下标定板。把 `calib/camera_extrinsics.json`、`calib/verify/` 发给我，下一步接相机点云 → 物体位姿 → BODex/DexLearn 候选。

## 标定文件说明

| 文件 | 内容 |
|---|---|
| `calib/board/board.json` | 板的规格（实测尺寸） |
| `calib/run*/samples.jsonl`、`img_*.png` | 每个样本的关节角、法兰位姿、板位姿、识别质量和原图 |
| `calib/camera_extrinsics.json` | `T_world_cam`（4×4）、位置/四元数(wxyz)/rpy、`T_flange_board`、残差和结论 |
| `calib/verify/verify.json`、`overlay.png` | 验证结果、桌面高度 |
