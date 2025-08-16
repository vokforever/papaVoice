import torch


def check_gpu():
    print("=== ДИАГНОСТИКА GPU ===")
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"cuDNN версия: {torch.backends.cudnn.version()}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Количество GPU: {torch.cuda.device_count()}")
        print(f"Текущий GPU: {torch.cuda.current_device()}")
        print(f"Название GPU: {torch.cuda.get_device_name(0)}")
        print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} ГБ")

        # Тестовый тензор на GPU
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            print(f"Тестовый тензор на GPU: {x}")
            print("GPU работает корректно!")
        except Exception as e:
            print(f"Ошибка при работе с GPU: {e}")
    else:
        print("GPU недоступен")


if __name__ == "__main__":
    check_gpu()