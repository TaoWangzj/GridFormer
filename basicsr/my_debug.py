from os import path as osp
root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))

print(__file__)
print(osp.join(__file__, osp.pardir, osp.pardir))
print(root_path)

