import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd
# 这些库包括基本的 Python 库，PyTorch，用于配置处理的 yaml，用于属性访问的 munch，以及用于实验跟踪的 wandb。此外，还导入了自定义模块和工具。

class SLAM:
    def __init__(self, config, save_dir=None):  # python slam.py --config configs/mono/tum/fr3_office.yaml
        start = torch.cuda.Event(enable_timing=True)  #创建了一个CUDA事件对象 start，用于记录开始时间。
        end = torch.cuda.Event(enable_timing=True)  #创建了一个CUDA事件对象 end，用于记录结束时间。

        start.record()
        # 计时器：使用 CUDA 事件来记录时间，评估性能。
        self.config = config
        self.save_dir = save_dir
        # 将 config 中的参数转换为 model_params、opt_params 和 pipeline_params 三个对象，并保存在类属性中。
        # 通过munchify 函数将 config["model_params"] 这个字典转换为了一个 munch 对象，这样可以更方便地访问其中的参数
        model_params = munchify(config["model_params"])  # 模型参数
        opt_params = munchify(config["opt_params"])  # 优化参数
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )
        # 模式设置：确定是否为实时模式，是否为单目相机，是否使用球谐函数，以及是否使用 GUI
        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]
        # sh系数
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        # 初始化高斯模型（并设置一些列参数）
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)  # 初始化高斯模型的学习率（空间学习率）
        self.dataset = load_dataset(  # 数据集加载：加载数据集。
            model_params, model_params.source_path, config=config
        )
        # 设置高斯模型的优化参数
        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]  # 设置背景颜色：黑色
        # 创建了一个 PyTorch 张量来表示背景颜色，并将其存储在类的属性中。这个张量会被放置在 CUDA 设备上。
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # 创建了两个多进程队列，用于前端和后端之间的通信。（此处注意：在config文件中设置 single_thread: False 默认采用多 GPU 或多 CPU ）
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()
        # 根据是否使用 GUI，选择使用 mp.Queue() 或者 FakeQueue() 创建一个模拟的队列对象 
        # 这样做的目的是根据程序的运行模式来选择不同类型的队列。如果程序需要与 GUI 进程进行通信，就使用真实的多进程队列；
        # 如果不需要，就使用模拟的队列，以避免在没有 GUI 的情况下引发异常。
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()
        # 更新了配置中的保存结果的目录路径和是否为单目摄像头
        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular
        # 创建了前端和后端对象，并设置属性
        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)
        # 前端配置：数据集、背景色、管道参数、队列等都赋给前端。用set_hyperparams() 方法设置前端的超参数。
        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()
        # 后端配置：高斯模型、背景色、相机范围、管道参数、优化参数、队列及实时模式都赋给后端。用set_hyperparams() 方法设置后端的超参数。
        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode
        
        self.backend.set_hyperparams()
        # 创建了一个用于显示参数的 GUI 对象，包含管道参数、背景、高斯模型和队列。
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )
        # 创建一个后台进程 backend_process 并开始执行后端的运行函数 self.backend.run()。
        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:  # 如果使用 GUI，创建一个 GUI 进程 gui_process 并开始执行。
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()  # 启动 GUI 进程，让它开始执行 GUI 的运行函数。
            time.sleep(5)  # 等待5秒
        
        backend_process.start()  # 启动后台进程，让它开始执行后端的运行函数。
        self.frontend.run()  # 前端运行：调用前端的run方法开始执行。
        backend_queue.put(["pause"])  #暂停后端：向后端队列发送暂停命令。

        end.record()  # 记录CUDA事件结束时间
        torch.cuda.synchronize()  # 同步CUDA：它确保在继续执行之前所有CUDA操作完成
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)  # 获取了前端处理的相机帧数量，用于计算帧率。
        FPS = N_frames / (start.elapsed_time(end) * 0.001)  # 计算了整个运行过程的时间，单位是秒。
        # FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")  # 日志记录：记录总时间和FPS
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")
        # 打印到终端查看（此处输出的是全部的时间）
        # print("Total time:", total_time, "seconds")
        # print("Total FPS:", FPS)
        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians  # 前端处理后的高斯模型参数
            kf_indices = self.frontend.kf_indices  # 关键帧索引
            # 计算 Absolute Trajectory Error (ATE)。这个函数接受相机位置、关键帧索引等参数，计算出评估结果。
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )
            # 执行渲染评估。这个函数接受相机位置、高斯模型参数、数据集等参数，计算出渲染结果。
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            # 创建一个表格对象，用于记录评估结果。
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(  # 将渲染评估结果添加到表格中
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()  # 清空前端队列：通过循环读取并丢弃 frontend_queue 中所有的消息，以确保队列在开始新的任务之前是空的，避免处理残留的旧消息。
            backend_queue.put(["color_refinement"])  # 触发颜色优化：向 backend_queue 队列发送一个 ["color_refinement"] 消息。这会通知后端开始颜色优化过程。
            while True:  # 在一个无限循环中等待从前端队列中获取同步信号，并从后端队列中获取最新的高斯模型参数。
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()  # 处理前端消息：从 frontend_queue 中获取消息 data。
                if data[0] == "sync_backend" and frontend_queue.empty():  # 如果消息类型是 "sync_backend" 并且 frontend_queue 为空：
                    gaussians = data[1]
                    self.gaussians = gaussians  # 提取并更新高斯模型 self.gaussians。
                    break
            # 消息队列：使用消息队列在前端和后端之间通信，使得前后端可以异步操作。
            # 同步机制：通过等待特定类型的消息（"sync_backend"）来确保前后端同步。
            # 这段代码的核心是在前后端之间进行同步，并确保在执行后端任务（如颜色优化）后，前端能够正确接收并更新数据，从而保证 SLAM 系统的正确运行。
            
            # 重新执行渲染评估，这次是在优化后的状态下。
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            # 将优化后的渲染评估结果添加到表格中，标签为 "After"。
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})  # 记录到 WandB：将评估结果记录到 Weights & Biases 以便可视化和分析，将表格对象作为字典的值，"Metrics" 作为键。
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)  # 将高斯模型和相机参数保存到指定目录中。

        backend_queue.put(["stop"])  # 停止后端：这会通知后端进程停止其运行。
        backend_process.join()  # 等待后端进程结束：调用 backend_process.join() 方法，这会阻塞主线程，直到 backend_process 进程完成并退出。
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))  # 通知 GUI 停止：向 q_main2vis 队列发送一个带有 finish=True 的 GaussianPacket 对象。这个消息会通知 GUI 进程停止运行。
            gui_process.join()  # 等待 GUI 进程结束：调用 gui_process.join() 方法，这会阻塞主线程，直到 gui_process 进程完成并退出。
            Log("GUI Stopped and joined the main thread")
        # 这段代码的核心目的是确保系统在完成所有任务后能够优雅地关闭所有子进程，并确保这些进程正确地与主线程合并。
    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)  # 读入config文件
    parser.add_argument("--eval", action="store_true")  # 直接进行评估

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)  # 读入config文件
    save_dir = None

    if args.eval:  # 如果输入了--eval参数
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True  # 设置评估渲染为True，这样就会执行渲染评估。并将结果保持
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])  # 就创建一个目录，该目录的路径在配置文件中指定。
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # 获取当前的日期和时间，并将其格式化为字符串。
        path = config["Dataset"]["dataset_path"].split("/")  # 将数据集路径按照"/"进行分割
        save_dir = os.path.join(  # 将数据集路径的最后三个部分和当前的日期和时间拼接在一起，作为保存结果的目录。
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]  # 将文件名（args.config）中的扩展名去除。
        config["Results"]["save_dir"] = save_dir  # 更新配置文件中保存结果的目录路径。
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        # 初始化WandB（Weights and Biases）的运行环境，用于跟踪实验结果。
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        # 定义了两个指标，用于在WandB上跟踪实验的进展。
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")
    # 初始化SLAM对象，传入了配置和保存结果的目录路径。
    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
