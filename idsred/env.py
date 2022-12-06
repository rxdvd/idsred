import os
from dotenv import load_dotenv

def set_workdir(path='data'):

    with open('.env', 'w') as file:
        file.write('# Working directory variables\n')
        file.write(f'WORKDIR={path}')
        proc_dir = os.path.join(path, 'processing')
        file.write(f'PROCESSING={proc_dir}')

