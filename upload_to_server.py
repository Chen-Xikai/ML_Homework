"""
upload_to_server.py
自动上传文件到云服务器并启动训练
"""

import os
import paramiko
import time

# 服务器配置
SERVER_HOST = '47.120.13.212'
SERVER_USER = 'root'
SERVER_PASSWORD = '2282678Kk'
SERVER_PORT = 22

# 本地路径
LOCAL_DIR = r'C:\Users\ASUS\Desktop\任务三'
LOCAL_DATA_DIR = r'C:\Users\ASUS\Desktop\task3\data'

# 服务器路径
REMOTE_DIR = '/root/siamese_project'


def create_ssh_client():
    """创建SSH连接"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER_HOST, port=SERVER_PORT, username=SERVER_USER, password=SERVER_PASSWORD)
    return client


def upload_file(sftp, local_path, remote_path):
    """上传单个文件"""
    file_size = os.path.getsize(local_path)
    print(f"  Uploading: {os.path.basename(local_path)} ({file_size/1024/1024:.1f}MB)")
    sftp.put(local_path, remote_path)
    print(f"  Done: {os.path.basename(local_path)}")


def upload_directory(sftp, local_dir, remote_dir):
    """上传整个目录"""
    for item in os.listdir(local_dir):
        local_path = os.path.join(local_dir, item)
        remote_path = f"{remote_dir}/{item}"
        
        if os.path.isfile(local_path):
            upload_file(sftp, local_path, remote_path)
        elif os.path.isdir(local_path):
            sftp.mkdir(remote_path)
            upload_directory(sftp, local_path, remote_path)


def main():
    print("=" * 60)
    print("Uploading to Cloud Server")
    print(f"Server: {SERVER_USER}@{SERVER_HOST}")
    print("=" * 60)
    
    # 连接服务器
    print("\n1. Connecting to server...")
    client = create_ssh_client()
    sftp = client.open_sftp()
    print("   Connected!")
    
    # 创建远程目录
    print("\n2. Creating directories...")
    try:
        sftp.mkdir(REMOTE_DIR)
    except:
        pass
    try:
        sftp.mkdir(f"{REMOTE_DIR}/dataset_cache")
    except:
        pass
    try:
        sftp.mkdir(f"{REMOTE_DIR}/data")
    except:
        pass
    try:
        sftp.mkdir(f"{REMOTE_DIR}/checkpoints")
    except:
        pass
    print("   Directories created!")
    
    # 上传代码文件
    print("\n3. Uploading code files...")
    code_files = ['model.py', 'dataset.py', 'evaluate.py', 'utils.py', 
                  'cache_dataset.py', 'cloud_train.py']
    for f in code_files:
        local_path = os.path.join(LOCAL_DIR, f)
        if os.path.exists(local_path):
            upload_file(sftp, local_path, f"{REMOTE_DIR}/{f}")
    
    # 上传缓存文件
    print("\n4. Uploading cache files...")
    cache_dir = os.path.join(LOCAL_DIR, 'dataset_cache')
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            local_path = os.path.join(cache_dir, f)
            upload_file(sftp, local_path, f"{REMOTE_DIR}/dataset_cache/{f}")
    
    # 上传数据文件（这里只上传目录结构，实际数据可能需要单独处理）
    print("\n5. Checking data directory...")
    # 注意：数据文件可能很大，需要确认是否要上传
    
    # 关闭SFTP
    sftp.close()
    
    # 在服务器上安装依赖并运行
    print("\n6. Installing dependencies on server...")
    stdin, stdout, stderr = client.exec_command(
        f"cd {REMOTE_DIR} && pip install torch torchvision numpy pillow scikit-learn matplotlib -q"
    )
    stdout.channel.recv_exit_status()
    print("   Dependencies installed!")
    
    # 启动训练
    print("\n7. Starting training...")
    stdin, stdout, stderr = client.exec_command(
        f"cd {REMOTE_DIR} && nohup python cloud_train.py > train_log.txt 2>&1 &"
    )
    stdout.channel.recv_exit_status()
    print("   Training started in background!")
    
    # 关闭连接
    client.close()
    
    print("\n" + "=" * 60)
    print("Upload Complete!")
    print("=" * 60)
    print(f"\nMonitor training with:")
    print(f"  ssh {SERVER_USER}@{SERVER_HOST}")
    print(f"  tail -f {REMOTE_DIR}/train_log.txt")
    print(f"\nDownload results with:")
    print(f"  scp {SERVER_USER}@{SERVER_HOST}:{REMOTE_DIR}/checkpoints/best_model.pth ./")


if __name__ == "__main__":
    main()
