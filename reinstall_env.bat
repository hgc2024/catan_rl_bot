@echo off
echo Deleting old venv...
rmdir /s /q venv

echo Creating new venv with Python 3.10...
"C:\Users\henry-cao-local\AppData\Local\Programs\Python\Python310\python.exe" -m venv venv

echo Activating venv...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing requirements...
pip install -r requirements.txt

echo Environment setup complete!
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
pause
