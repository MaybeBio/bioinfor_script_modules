# 软件渲染兜底方案
# 不用 ANGLE 硬件 GL，强制 Qt 全软件光栅化
# 一次性全部导出，再启动：

export QT_QPA_PLATFORM=xcb
export QT_OPENGL=software
export QT_QUICK_BACKEND=software
export QTWEBENGINE_CHROMIUM_FLAGS="--disable-gpu --disable-software-rasterizer=0"
chimerax
