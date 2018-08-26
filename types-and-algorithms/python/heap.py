import abc

# ------------------------------------------ HEAPS

class Heap(abc.ABC):
	"""
	Abstract Base Class for the Heap data structure
	"""

	def __init__(self):
		self.array = list()

	def _swap(self, i, j):
		if any(not self._in_bounds(index) for index in [i, j]):
			raise Exception('At least one index is out of bounds')
		self.array[i], self.array[j] = self.array[j], self.array[i]

	def _in_bounds(self, i):
		return 0 <= i < len(self)

	def _satisfies_heap_property(self):
		if len(self) <= 1:
			return True
		for i in range(1, len(self)):
			satsifies = self._element_i_is_better_than_j(Heap.parent(i), i) or self.array[Heap.parent(i)] == self.array[i]
			if not satsifies:
				return False
		return True

	@classmethod
	def parent(cls, i):
		return (i - 1) // 2

	@classmethod
	def left(cls, i):
		return 2 * i + 1

	@classmethod
	def right(cls, i):
		return 2 * i + 2

	@classmethod
	def sort(cls, list_of_elements, reverse = False):
		if len(list_of_elements) <= 1:
			return list(list_of_elements)
		heap = MaxHeap() if reverse else MinHeap()
		heap.array = list_of_elements
		heap.build()
		sorted_elements = []
		while len(heap) != 0:
			sorted_elements.append(heap.pop_best())
		return sorted_elements

	@abc.abstractmethod
	def _element_i_is_better_than_j(self, i, j):
		pass

	@abc.abstractmethod
	def _element_i_is_worse_than_j(self, i, j):
		pass

	def __len__(self):
		return len(self.array)

	def __repr__(self):
		return repr(self.array)

	def __str__(self):
		return str(self.array)

	def heapify(self, i):
		left = Heap.left(i)
		right = Heap.right(i)
		# determine if left or i is best
		if self._in_bounds(left) and self._element_i_is_better_than_j(left, i):
			best_index = left
		else:
			best_index = i
		# now update if right is even better
		if self._in_bounds(right) and self._element_i_is_better_than_j(right, best_index):
			best_index = right
		# finally, swap i with the best element unless it is i
		if best_index != i:
			self._swap(i, best_index)
			self.heapify(best_index)

	def build(self):
		for i in range((len(self) - 1) // 2, -1, -1):
			self.heapify(i)

	def best(self):
		if len(self) == 0:
			raise Exception('No best element in empty heap')
		return self.array[0]

	def pop_best(self):
		best_element = self.best()
		n = len(self)
		# now swap best and last elements
		self.array[0], self.array[n-1] = self.array[n-1], self.array[0] 
		# then remove the last element
		self.array.pop()
		# and finally max heapify the root
		self.heapify(0)
		return best_element

	def insert(self, element):
		minus_infinity = -1 * float('inf')
		self.array.append(minus_infinity)
		self.set(len(self) - 1, element)

	def set(self, i, new_element):
		self.array[i] = new_element
		# while i > 0 and self.array[Heap.parent(i)] > self.array[i]:
		while i > 0 and self._element_i_is_worse_than_j(Heap.parent(i), i):
			self._swap(i, Heap.parent(i))
			i = Heap.parent(i)



class MaxHeap(Heap):

	def _element_i_is_better_than_j(self, i, j):
		return self.array[i] > self.array[j]

	def _element_i_is_worse_than_j(self, i, j):
		return self.array[i] < self.array[j]



class MinHeap(Heap):
	
	def _element_i_is_better_than_j(self, i, j):
		return self.array[i] < self.array[j]

	def _element_i_is_worse_than_j(self, i, j):
		return self.array[i] > self.array[j]

# ------------------------------------------ TEST

def test_heap():
	# test parent, left, right
	indices = [ 0, 1, 2, 3,  4,  5,  6,  7,  8,  9]
	parents = [-1, 0, 0, 1,  1,  2,  2,  3,  3,  4]
	lefts =   [ 1, 3, 5, 7,  9, 11, 13, 15, 17, 19]
	rights =  [ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
	for i, parent, left, right in zip(indices, parents, lefts, rights):
		assert parent == Heap.parent(i)
		assert left   == Heap.left(i)
		assert right  == Heap.right(i)

	# test heapsort
	assert [] == Heap.sort([])

	assert [0.6543] == Heap.sort([0.6543])
	assert [9173] == Heap.sort([9173], reverse = True)

	assert [1, 2, 3.14] == Heap.sort([3.14, 1, 2])
	assert [1,-1,-1,-1] == Heap.sort([-1, -1, -1, 1], reverse = True)
	assert [-3.1, -0.123, 1.3, 2.9, 3.1, 4.0, 5.0, 6.0] == Heap.sort([2.9, -0.123, 6.0, 3.1, 4.0, 1.3, 5.0, -3.1])

def test_max_heap():
	# test insert, len, best, pop_best
	max_heap = MaxHeap()
	assert 0 == len(max_heap)
	
	max_heap.insert(1337)
	assert 1    == len(max_heap)
	assert 1337 == max_heap.best()
	assert 1337 == max_heap.pop_best()
	assert 0    == len(max_heap)

	max_heap.insert(-11)
	max_heap.insert(2)
	max_heap.insert(4)
	max_heap.insert(9)
	assert 4   == len(max_heap)
	assert 9   == max_heap.best()
	assert 9   == max_heap.pop_best() # should remove 9
	assert 3   == len(max_heap)
	assert 4   == max_heap.best()
	assert 4   == max_heap.pop_best() # should remove 4
	assert 2   == len(max_heap)
	assert 2   == max_heap.best()
	assert 2   == max_heap.pop_best() # should remove 2
	assert 1   == len(max_heap)
	assert -11 == max_heap.best()
	assert -11 == max_heap.pop_best() # should remove 2
	assert 0   == len(max_heap)

	# test build
	max_heap.array = [-57, -2, -1, 0, 1, 2]
	max_heap.build()
	assert [2, 1, -1, 0, -2, -57] == max_heap.array
	assert 2   == max_heap.pop_best()
	assert 1   == max_heap.pop_best()
	assert 0   == max_heap.pop_best()
	assert -1  == max_heap.pop_best()
	assert -2  == max_heap.pop_best()
	assert -57 == max_heap.pop_best()

def test_min_heap():
	# test insert, len, best, pop_best
	min_heap = MinHeap()
	assert 0 == len(min_heap)

	min_heap.insert(-666)
	assert 1    == len(min_heap)
	assert -666 == min_heap.best()
	assert -666 == min_heap.pop_best()
	assert 0    == len(min_heap)

	min_heap.insert(99)
	min_heap.insert(0)
	min_heap.insert(-3)
	min_heap.insert(-3) # purposely a duplicate
	min_heap.insert(6)
	assert 5  == len(min_heap)
	assert -3 == min_heap.best()
	assert -3 == min_heap.pop_best() # should remove a -3
	assert 4  == len(min_heap)
	assert -3 == min_heap.best()
	assert -3 == min_heap.pop_best() # should remove -3
	assert 3  == len(min_heap)
	assert 0  == min_heap.best()
	assert 0  == min_heap.pop_best() # should remove 0
	assert 2  == len(min_heap)
	assert 6  == min_heap.best()
	assert 6  == min_heap.pop_best() # should remove 6
	assert 1  == len(min_heap)
	assert 99 == min_heap.best()
	assert 99 == min_heap.pop_best() # should remove 99
	assert 0  == len(min_heap)

	# test build
	min_heap.array = [-1,0,-2,0,1,3]
	min_heap.build()
	assert [-2, 0, -1, 0, 1, 3] == min_heap.array
	assert -2 == min_heap.pop_best()
	assert -1 == min_heap.pop_best()
	assert 0  == min_heap.pop_best()
	assert 0  == min_heap.pop_best()
	assert 1  == min_heap.pop_best()
	assert 3  == min_heap.pop_best()

# ------------------------------------------ MAIN

def main():
	test_heap()
	print('Heap tests pass')

	test_max_heap()
	print('Max heap tests pass')
	
	test_min_heap()
	print('Min heap tests pass')

if __name__ == '__main__':
	main()
