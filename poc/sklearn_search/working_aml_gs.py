from subprocess import Popen

# Subprocess Command
cmd = "python3 working_aml.py"
p = Popen(cmd.split())
print("Process ID:", p.pid)

# Check the status of process
# poll() method returns 'None' if the process is running else returns the exit code
print(p.poll())
