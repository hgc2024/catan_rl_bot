@echo off
echo Cleaning up old venv...
if exist venv rmdir /s /q venv

echo Creating new venv with Python 3.10...
"C:\Users\henry-cao-local\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv

echo Activating venv...
call venv\Scripts\activate

echo Installing PyTorch (CUDA 12.1)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing requirements...
pip install -r requirements.txt

echo Setup Complete!
