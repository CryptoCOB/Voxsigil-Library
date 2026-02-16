import marshal, types, os
paths=[r'd:\01_1\Nebula\core\quantum\__pycache__\entropy_collector.cpython-313.pyc',
       r'd:\01_1\Nebula\nebula\sdk\__pycache__\wallet_sdk.cpython-313.pyc']

def walk(co,names,strings):
    names.update(co.co_names)
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            walk(c,names,strings)
        elif isinstance(c,str):
            strings.add(c)

for p in paths:
    if not os.path.exists(p):
        print('MISSING',p)
        continue
    with open(p,'rb') as f:
        header=f.read(16)  # py 3.13 header
        code=marshal.load(f)
    names=set(); strings=set(); walk(code,names,strings)
    names=[n for n in names if n and not n.startswith('__')]
    print('\nPYC:',p)
    print('NAMES:',sorted(names))
    print('STRING_CONSTS_FIRST25:',sorted(list(strings))[:25])
