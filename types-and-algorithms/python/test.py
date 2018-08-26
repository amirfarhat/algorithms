import heap, priorityqueue, stack, queue 

TYPES = (heap, priorityqueue, stack, queue)

def main():
	for datatype in TYPES:
		datatype.main() # run the main function
		print()

if __name__ == '__main__':
	main()
