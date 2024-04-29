import subprocess
import psutil
import time

def monitor_process(pid):
    """ Monitor the CPU and memory usage of a process. """
    p = psutil.Process(pid)
    cpu_percentages = []
    memory_usages = []
    
    while p.is_running():
        try:
            # Get CPU and memory usage
            cpu = p.cpu_percent(interval=1)
            memory = p.memory_info().rss  # rss is the Resident Set Size
            cpu_percentages.append(cpu)
            memory_usages.append(memory)
        except psutil.NoSuchProcess:
            break
    
    return cpu_percentages, memory_usages

def run_program(command):
    """ Run a program and monitor its performance. """
    # Start the program
    start_time = time.time()
    proc = subprocess.Popen(command, shell=True)
    
    # Monitor the program
    try:
        cpu_usage, memory_usage = monitor_process(proc.pid)
    finally:
        proc.terminate()
        proc.wait()

    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Average CPU usage: {sum(cpu_usage) / len(cpu_usage):.2f}%")
    print(f"Peak memory usage: {max(memory_usage) / (1024**2):.2f} MB")  # Convert bytes to MB

if __name__ == "__main__":
    program_command = "python gpu_mandel.py"
    run_program(program_command)
