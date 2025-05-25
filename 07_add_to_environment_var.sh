# 将当前路径添加到环境变量中
echo "export PATH=$(pwd):$PATH" >> ~/.bashrc
source ~/.bashrc

echo "export PATH=$(pwd):$PATH" >> ~/.zshrc
source ~/.zshrc
