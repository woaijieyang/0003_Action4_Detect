# Verified execution environment

Python3.12.3; torch2.8.0+cu128; torchvision0.23.0+cu128; NumPy2.5.2; SciPy1.18.1; Pillow12.3.0; PyYAML6.0.3. OpenCV4.13.0.92 is isolated in runtime/surface_deps. These are observed versions, not proof every other platform is supported.

Use pyproject extras data,train,calibration. Select the correct CUDA wheel index for your hardware; do not install globally on the shared server. Model/data assets are separate. The project-local environment has no pytest installed; existing unittest scripts and explicit real-data checks were used.
