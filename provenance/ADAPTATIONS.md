# 独立副本调整记录

所有调整仅发生在新建的 rack_grasp_vision 内。原始工程和 Conda 环境保持原状。

- box_in_basket/dataset/data.yaml：移除原工程绝对根路径，按 YAML 所在目录解析。
- train.py：预训练权重引用改为本包 weights/yolov8n-pose.pt，并使用本包缓存。该权重从原机器的第三方模型目录复制，详见 copy_inventory.json。
- evaluate.py：只增加本包缓存设置，保留默认第二次实验模型及评估口径。
- start_live.sh、weights/base.json：录像界面默认改为原首盒入口正在使用的篮架模型，而不是旧 YOLO26 第173轮模型。原首盒主入口的模型与参数没有改变。
- split_dataset.py：由原文件“split_dataset .py”复制并规范文件名；默认数据源/语义改为本包篮架数据；新导出 YAML 也不写死绝对根路径。
- scripts/view_d405_yolo.py：处理 Qt 窗口已被关闭后查询可见性产生的异常，避免正常关窗被记录为应用失败。原历史报告中存在该错误；新增三个回归测试。
- preview_image.py：增加独立离线图片检测与分格预览，不打开相机、不控制机械臂。单帧分格仅用于离线查看，不改变实时确认配置。
- requirements.txt、README.md：按现有 yolo-pose 环境和新入口整理，旧项目环境说明仅留在 original_project 供溯源。没有安装任何依赖。

复制清单包含2798个来源文件，不含运行缓存。除上述适配文件外，复制文件的名称与字节大小已核对；数据图片全部解码、标签全部解析、无文件缺失。检查不重新生成文件哈希。来源清单保留已有哈希信息的文件，模型加载器沿用其既有完整性校验约定。

验证在临时搬迁目录中执行，并在进程内拒绝读取原 yolo_pose_box 目录：数据解析、实际CPU图片推理、首盒侧栏、两种模式的合成帧MP4编解码及111项测试均通过，原目录访问尝试为0。临时验证目录随后搬回最终目录，因此测试记录中的 /tmp 路径为当时执行位置。

没有启动新训练，也没有启动相机流或机器人运动。当前执行进程无法使用CUDA，RealSense设备枚举报 could not initialize udev monitor；实机GPU/相机窗口、毫米定位和机械臂闭环仍待主机硬件验证。
