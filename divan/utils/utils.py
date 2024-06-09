import inspect

__all__ = ["set_arg", "set_kargs"]

def set_arg(_class, kargs=None):
    if kargs is None:
        frame_func = inspect.currentframe().f_back
        args, _, _, values = inspect.getargvalues(frame_func)
        if 'self' in args:
            args.remove('self')
        for arg in args:
            setattr(_class, arg, values[arg])
    else:
        set_kargs(_class, kargs)

def set_kargs(_class, kargs):
    assert isinstance(kargs, dict)
    kargs.pop('self', None)
    for arg, value in kargs.items():
        setattr(_class, arg, value)


if __name__ == '__main__':
    class test_class:
        def __init__(self, d=None):
            set_arg(self, d)
            print(f"self.test: {self.test}\n"
                  f"self.val: {self.val}")
    test_class({'test':1, 'val':2})