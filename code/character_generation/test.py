### test files to generate individual character bios 

import subprocess 
import sys 

def run_script(path):
    command = [sys.executable, path]
    output = subprocess.check_output(command)
    return output.decode('utf-8')

result = run_script('./code/character_generation/generate.py')

# newlines = []
# for i in range(len(result)):
#     if result[i] == '\n':
#         newlines.append(i)
#         print(result[:i])
# print(newlines)

print(result)