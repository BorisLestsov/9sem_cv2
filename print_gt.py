import pickle
from sys import argv


with open(argv[1], 'rb') as f:
	a = pickle.load(f)

for i in a:
	print(i)
