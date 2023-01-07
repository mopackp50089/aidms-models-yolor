import subprocess
import os

def bash_command(cmd):
    result = subprocess.Popen(['/bin/bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    std_output, std_error = result.communicate()
    # print(text)
    return result.returncode, std_output.decode("utf-8"), std_error.decode("utf-8")

def main():
    os.system(
        f"uwsgi --ini /workspace/aidms/lib/exec/inference_restfulapi.ini"
    )

if __name__ == "__main__":
    main()