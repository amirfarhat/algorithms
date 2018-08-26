from heap import MaxHeap, MinHeap

# ------------------------------------------ QUEUES

class MaxPriorityQueue:

	def __init__(self):
		self.max_heap = MaxHeap()

	def __len__(self):
		return len(self.max_heap)

	def insert(self, element):
		self.max_heap.insert(element)

	def best(self):
		return self.max_heap.best()

	def pop_best(self):
		return self.max_heap.pop_best()

	def set(self, old_element, new_element):
		while old_element in self.max_heap.array:
			index_of_old_element = self.max_heap.array.index(old_element)
			self.max_heap.set(index_of_old_element, new_element)

	def __repr__(self):
		return repr(self.max_heap)

	def __str__(self):
		return str(self.max_heap)



class MinPriorityQueue:

	def __init__(self):
		self.min_heap = MinHeap()

	def __len__(self):
		return len(self.min_heap)

	def insert(self, element):
		self.min_heap.insert(element)

	def best(self):
		return self.min_heap.best()

	def pop_best(self):
		return self.min_heap.pop_best()

	def set(self, old_element, new_element):
		while old_element in self.min_heap.array:
			index_of_old_element = self.min_heap.array.index(old_element)
			self.min_heap.set(index_of_old_element, new_element)
		
	def __repr__(self):
		return repr(self.min_heap)

	def __str__(self):
		return str(self.min_heap)

# ------------------------------------------ TEST

def test_max_priority_queue():
	# test insert, len, best, pop_best
	max_pq = MaxPriorityQueue()
	assert 0 == len(max_pq)

	max_pq.insert(-13)
	assert 1   == len(max_pq)
	assert -13 == max_pq.best()
	assert -13 == max_pq.pop_best()
	assert 0   == len(max_pq)

	max_pq.insert(1)
	max_pq.insert(2)
	max_pq.insert(3)
	assert 3 == len(max_pq)
	assert 3 == max_pq.best()
	assert 3 == max_pq.pop_best() # should pop 3
	assert 2 == len(max_pq)
	assert 2 == max_pq.best()
	assert 2 == max_pq.pop_best() # should pop 2
	assert 1 == len(max_pq)
	assert 1 == max_pq.best()
	assert 1 == max_pq.pop_best() # should pop 1
	assert 0 == len(max_pq)

	# test set
	max_pq.insert(-10)
	assert 1   == len(max_pq)
	assert -10 == max_pq.best()

	max_pq.set(-10, 5)
	assert 1 == len(max_pq)
	assert 5 == max_pq.best() # should now be 5
	assert 5 == max_pq.pop_best()
	assert 0 == len(max_pq)

	max_pq.insert(7)
	max_pq.insert(7)
	assert 2 == len(max_pq)
	
	max_pq.set(7, 11)
	assert 2 == len(max_pq)
	assert 11 == max_pq.best()
	assert 11 == max_pq.pop_best() # should pop the first 11
	assert 1 == len(max_pq)
	assert 11 == max_pq.best()
	assert 11 == max_pq.pop_best() # should pop the last 11
	assert 0 == len(max_pq)

def test_min_priority_queue():
	# test insert, len, best, pop_best
	min_pq = MinPriorityQueue()
	assert 0 == len(min_pq)

	min_pq.insert(1111)
	assert 1    == len(min_pq)
	assert 1111 == min_pq.best()
	assert 1111 == min_pq.pop_best() # should pop 1111
	assert 0    == len(min_pq)

	min_pq.insert(5)
	min_pq.insert(2)
	min_pq.insert(10)
	assert 3 == len(min_pq)
	assert 2 == min_pq.best()
	assert 2 == min_pq.pop_best() # should pop 2
	assert 2 == len(min_pq)
	assert 5 == min_pq.best()
	assert 5 == min_pq.pop_best() # should pop 5
	assert 1 == len(min_pq)
	assert 10 == min_pq.best()
	assert 10 == min_pq.pop_best() # should pop 10
	assert 0 == len(min_pq)

	# test set
	min_pq.insert(-16)
	min_pq.insert(15)
	min_pq.insert(14)
	assert 3  == len(min_pq)
	assert -16 == min_pq.best()

	min_pq.set(-16, -32)
	assert 3   == len(min_pq)
	assert -32 == min_pq.pop_best() # should pop -32
	assert 2   == len(min_pq)
	assert 14  == min_pq.pop_best() # should pop 14
	assert 1   == len(min_pq)
	assert 15  == min_pq.pop_best() # should pop 15
	assert 0   == len(min_pq)

# ------------------------------------------ MAIN

def main():
	test_max_priority_queue()
	print("Max priority queue tests pass")

	test_min_priority_queue()
	print("Min priority queue tests pass")

if __name__ == '__main__':
	main()

		
		