import torch


class CUDAHandler:
    def __init__(self, use_cuda=True, cuda_device=None):
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device
        self.device = None
        self.cuda_available = torch.cuda.is_available()

    def check_cuda(self):
        if self.use_cuda and self.cuda_available:
            if self.cuda_device:
                if self.cuda_device.startswith('cuda:') and torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    device_idx = int(self.cuda_device.split(':')[1])
                    if device_idx < device_count:
                        self.device = torch.device(self.cuda_device)
                        print(f"Using specified device: {self.cuda_device} - {torch.cuda.get_device_name(device_idx)}")
                    else:
                        print(
                            f"Requested CUDA device {self.cuda_device} does not exist. Using the first available CUDA device.")
                        self.device = torch.device('cuda:0')
                else:
                    print("Invalid CUDA device specified. Using the first available CUDA device.")
                    self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cuda:0')
                print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("No GPU detected or CUDA not requested. Using CPU.")

        print(f"Current device: {self.device}")
        return self.device
