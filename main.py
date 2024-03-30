import subprocess


"""
请使用对应conda环境中的interpreter执行

手动安装：
pip install pandas numpy jieba tqdm jupyter

pytorch cpu版：
pip3 install torch torchvision torchaudio 
"""
def check_nvidia_smi():
    try:
        # 运行nvidia-smi命令
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        # 输出nvidia-smi命令的输出
        print(result.stdout)
        return True
    except FileNotFoundError:
        print("[ERROR]: nvidia-smi命令未找到，请确保已安装NVIDIA驱动和nvidia-smi命令。")
        return False

if __name__ == '__main__':
    packages_to_install = ['pandas', 'numpy', 'jieba', 'tqdm','jupyter']
    for package in packages_to_install:
        print(f"[INFO]: Installing... {package} now!")
        subprocess.run(['pip3', 'install', package, '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'])

    try:
        import pandas
        import numpy
        import jieba
        from tqdm import tqdm

        print("除PyTorch外, 所有包都成功安装！")
    except ImportError as e:
        print(f"安装失败：{e}, 请自行手动安装..")


    print(f"\n\n[INFO]: 是否安装PyTorch？（安装失败可以手动安装，自行查询安装教程）pytorch官方：https://pytorch.org"
          f"\n\t\tMAC用户可以直接选择安装CPU版, Torch提供了对MPS加速的支持"
          f"\n[WARN]: GPU版本需要你有NVIDIA的显卡，和CUDA驱动.."
          f"\n\t\t请选择 <CPU版: 1, GPU版: 2>: ",end="")

    in_ = input()
    assert in_ in ['1','2'], f"请正确输入选项, ==> {in_}"
    if in_ == "1":
        print(f"[INFO]: Installing... PyTorch now!")
        subprocess.run(['pip3', 'install', 'torch torchvision torchaudio'])
    else:
        if check_nvidia_smi():
            print(f"[INFO]: 查看所显示的cuda版本，选择对应cuda版本进行安装：<cuda==11.x,请输入11> <cuda==12.x, 请输入12>: ")
            in_ = input()
            assert in_ in ['11','12'], f"请正确输入选项, ==> {in_}"

            if in_ == '11':
                subprocess.run(['pip3', 'install', 'torch','torchvision','torchaudio','--index-url','https://download.pytorch.org/whl/cu118'])
            else:
                subprocess.run(['pip3', 'install', 'torch','torchvision','torchaudio','--index-url','https://download.pytorch.org/whl/cu121'])


    try:
        import torch

        print(f"PyTorch已安装, 可用GPU数量：{torch.cuda.device_count()}.")
    except ImportError as e:
        print(f"安装失败：{e}, 请自行手动安装..")