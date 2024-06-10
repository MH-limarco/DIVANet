import os, subprocess, zipfile, logging

http = 'https://cchsu.info/files/images.zip'

os_dir = {'posix':'liunx',
          'nt':'windows'
          }

__all__ = ['check_file']

block_name = 'test_file'

def check_file(data_name, _http=http):
    logging.info(f'OS: {os_dir[os.name]}')
    logging.info(f'{block_name}: Dataset name - {data_name}')

    if data_name not in os.listdir('dataset'):
        logging.warning(f'{block_name}: Dataset not exist')
        download_data(data_name, _http)
        return True

    else:
        for i in []:
            if i not in os.listdir(f'dataset/{data_name}'):
                download_data(data_name, _http)
                break
        logging.info(f'{block_name}: Dataset exist')
        return True

def download_data(data_name, _http):
    logging.info(f'{block_name} : Download http - {_http}\nStart download...')
    _download_name = _http.split('/')[-1]

    command = {'posix':['wget', 'rm'],
               'nt':['curl', 'del']
               }[os.name]

    if os.name == 'posix':
        subprocess.call([command[0], _http])
    else:
        subprocess.call([command[0], _http, '-o', f'{os.getcwd()}/{_download_name}'])

    with zipfile.ZipFile(_download_name, 'r') as zip_ref:
        zip_ref.extractall(f'dataset//{data_name}')

    if os.name == 'posix':
        subprocess.call([command[1], _download_name])
    else:
        subprocess.call(['cmd', '/c', command[1], f'{_download_name}'])
    logging.info(f'{block_name}: Download done')
