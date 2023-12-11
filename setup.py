from setuptools import setup, Extension

module = Extension('symnmf',
                   sources=['symnmfmodule.c', 'symnmf.c'],)

setup(name='SymNMF',
      version='1.0',
      description='Symmetric Non-negative Matrix Factorization Module',
      ext_modules=[module])
