# 将当前路径添加到环境变量中

# 1, 原始方式: 链式追加, 复用不易排查
echo "export PATH=$(pwd):$PATH" >> ~/.bashrc
source ~/.bashrc

echo "export PATH=$(pwd):$PATH" >> ~/.zshrc
source ~/.zshrc

===============================================================================================================================================

# 2, 统一管理路径: 定义专属变量
# 先定义一个专门用于存储自定义路径的变量（如 MY_CUSTOM_PATHS），后续新增路径时只需向该变量追加，最后统一通过一条 export 语句生效

# 用户自定义工具/环境路径（即非系统预装，而是自己下载、编译、conda安装的专属路径），放在MY_CUSTOM_PATHS
# 系统默认路径，保留在原$PATH中
# 后续新增路径只需追加到MY_CUSTOM_PATHS中即可
（1）目前在~/.zshrc以及~/.bashrc中配置:
新增1行: MY_CUSTOM_PATHS=" 自定义工具bin目录环境变量 " (该行在conda初始化前)

# 末尾仅需1句: export PATH="$MY_CUSTOM_PATHS:$PATH"
# 为防止末尾语句在多次export PATH中迭代产生冗余, 可以设置一个标记变量判断“是否已经导出过 PATH”，避免重复拼接
末尾仅需1句:
if [[ -z "$_HAS_EXPORTED_CUSTOM_PATH" ]]; then
    export PATH="$MY_CUSTOM_PATHS:$PATH"
    export _HAS_EXPORTED_CUSTOM_PATH="true"  # 标记为“已导出”
fi


(2) 如果要动态添加新的环境变量, 只需要添加到MY_CUSTOM_PATHS变量末尾即可

# 检查 MY_CUSTOM_PATHS 是否定义，未定义则初始化为空字符串；然后追加当前路径
(grep -q "MY_CUSTOM_PATHS=" ~/.zshrc || echo 'MY_CUSTOM_PATHS=""') && echo "MY_CUSTOM_PATHS=\"\$MY_CUSTOM_PATHS:$(pwd)\"" >> ~/.zshrc;
# 确保末尾有统一的 export 语句，无则添加
grep -q "export PATH=\"\$MY_CUSTOM_PATHS:\$PATH\"" ~/.zshrc || echo 'export PATH="$MY_CUSTOM_PATHS:$PATH"' >> ~/.zshrc;
# 生效配置
# 只需动态更新MY_CUSTOM_PATHS变量，PATH会自动更新，确保系统 PATH 环境变量中包含之前定义的 MY_CUSTOM_PATHS 路径
source ~/.zshrc

(3) 暂时有点小问题, 但是本质逻辑就是拼接, $MY_CUSTOM_PATHS:$PATH (前者静态,追加时更新; 后者有系统原始环境变量备份, 后续可还原)
