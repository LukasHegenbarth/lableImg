import subprocess
from typing import Text

class Gpu_Info():
    def __init__(self):
        pass
        

    def get(self):
        # self.sp = subprocess.Popen(['nvidia-smi', '-q', '--display=Utilization,Memory'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = subprocess.check_output(['nvidia-smi', '-q', '--display=Utilization,Memory'])
        # out_str = self.sp.communicate()
        out_list = result.decode("utf-8").split('\n')
        mem_total = out_list[10].split(':')[1]
        mem_used = out_list[11].split(':')[1]
        mem_free = out_list[12].split(':')[1]

        gpu_max = out_list[25].split(':')[1]
        gpu_min = out_list[26].split(':')[1]
        gpu_avg = out_list[27].split(':')[1]

        return(mem_total, mem_used, mem_free, gpu_max, gpu_min, gpu_avg)



