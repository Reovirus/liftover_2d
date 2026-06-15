#!/bin/bash
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "Устанавливаем зависимости для архитектуры ARM64..."
    arch -arm64 python3 -m pip install -r requirements.txt
else
    echo "Устанавливаем зависимости для архитектуры x86_64..."
    python3 -m pip install -r requirements.txt
fi