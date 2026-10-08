#!/usr/bin/env python3

import importlib
import sys


def check_package(name: str, module_name: str | None = None) -> bool:
    module_name = module_name or name
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', getattr(module, 'version', 'unknown'))
        print(f'{name}: PASS ({version})')
        return True
    except Exception as exc:
        print(f'{name}: FAIL ({type(exc).__name__})')
        print('Install with: python -m pip install -r requirements.txt')
        return False


def main() -> int:
    print('Python:', sys.version.splitlines()[0])
    print('Executable:', sys.executable)
    success = True
    success = check_package('numpy') and success
    success = check_package('pandas') and success
    success = check_package('serial', 'serial') and success
    success = check_package('yaml', 'yaml') and success
    success = check_package('zhinst') and success
    print('Environment check:', 'PASS' if success else 'FAIL')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
