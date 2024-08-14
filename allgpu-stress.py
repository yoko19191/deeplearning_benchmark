#一个名为compute_time的变量，用于设置自定义计算时间。然后，在遍历所有可用的GPU并设置当前设备后，我们使用一个while循环来重复执行矩阵乘法操作，直到超过了设定的计算时间。在计算完成后，我们将结果张量复制回CPU并清空GPU显存。最后，重置开始时间以便于下一个设备的计算
import time
import torch
import multiprocessing as mp

# 设置tensor core
torch.backends.cuda.matmul.allow_tf32 = True

# 定义输入矩阵大小和批次大小
batch_size = 32
input_size = 2048
hidden_size = 2048
output_size = 512

# 自定义计算时间
compute_time = 1200  # 单位：秒

# 遍历所有可用GPU
device_count = torch.cuda.device_count()
processes = []
for i in range(device_count):
    device = torch.device(f"cuda:{i}")
    print(f"Creating process for device {device}")

    # 定义计算函数
    def compute():
        # 设置当前设备
        with torch.cuda.device(device):
            # 生成输入和输出tensor
            input_tensor = torch.randn(batch_size, input_size, hidden_size, dtype=torch.float32).cuda()
            output_tensor = torch.randn(batch_size, hidden_size, output_size, dtype=torch.float32).cuda()

            # 执行矩阵乘法
            start_time = time.time()
            while time.time() - start_time < compute_time:
                result_tensor = torch.matmul(input_tensor, output_tensor)

            # 将结果复制回CPU
            result_tensor = result_tensor.cpu()

            # 释放GPU显存
            torch.cuda.empty_cache()

            return time.time() - start_time

    # 创建进程
    p = mp.Process(target=compute)
    p.start()
    processes.append(p)

# 等待所有进程完成
for p in processes:
    p.join()

# 同步所有GPU
torch.cuda.synchronize()

print("All processes completed.")

