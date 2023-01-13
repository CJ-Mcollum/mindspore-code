import os
import psutil
import gc

# def free_memory(m_size):
#     use_size = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
#     while True:
#         if use_size <= m_size:
#             break

def free_memory():
    gc.collect()


if __name__ == '__main__':
    m_size = 300
    free_memory()
