import pickle
from sys import argv


with open(argv[1], 'rb') as f:
	for j in range(8):
		a = pickle.load(f)

		if j == 4:
			for i in a:
				print(i)
