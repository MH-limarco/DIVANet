import logging
from itertools import zip_longest

def get_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
def logging_table(col, value, table_name='', it='', min_ncol=0, dict_mode=False):
    assert len(col) == len(value)

    it = it if not isinstance(it, (str, int)) else [it]*len(value)
    value = [(str(round(v, 2)) if isinstance(v, (int, float)) else v) + (i if i else '') for v, i in zip_longest(value, it)]
    _col_str = ''
    for c, v in zip(col, value):
        _max = max([len(c), len(v), min_ncol]) + 2
        _str = '|{:^' + f'{_max}' + '}'
        _col_str += _str

    col = (_col_str+'|').format(*col)
    value = (_col_str+'|').format(*value)
    table_name = f"|{table_name:^{len(value)-2}}|\n" if table_name else ''

    if not dict_mode:
        return f'\n{table_name}{col}\n{value}\n'
    else:
        return {'table':table_name, 'col':col, 'value':value}

def table_with_fix(col, fix_len=13):
    formatted_str = '|'.join(f'{x:^{fix_len}}' for x in col)
    return f'|{formatted_str}|'


if __name__ == '__main__':
    col = ['abc', 'b', 'c']
    value = [1, 2, 3]
    table_name = None
    it =['G']
    print(logging_table(col, value, table_name, it, dict_mode=True))
    a = table_with_fix(col, fix_len=0)
    print(a)