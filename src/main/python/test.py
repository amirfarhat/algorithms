import heap, priorityqueue, stack

TYPES = (heap, priorityqueue, stack)

def main():
	for datatype in TYPES:
		datatype.main() # run the main function
		print()

if __name__ == '__main__':
	main()
