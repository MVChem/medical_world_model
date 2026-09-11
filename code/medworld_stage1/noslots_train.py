"""Compatibility CLI; all variants share training.py."""
from training import main, optimizer_for, train, validate

if __name__ == '__main__':
    main('noslots')
