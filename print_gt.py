import pickle
from sys import argv


with open(argv[1], 'rb') as f:
<<<<<<< HEAD
	for j in range(8):
		a = pickle.load(f)

		if j == 4:
			for i in a:
				print(i)
=======
	a = pickle.load(f)

for i in a:
	print(i)
>>>>>>> 7b88f3b9c78c5d3a920a2cdb67a691ff32f43ddd
