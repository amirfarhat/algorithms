import abc

# ------------------------------------------ HEAPS

class HeapNode:
	def __init__(self, key, value = None):
		self.key = key if isinstance(key, (int, float)) else str(key)
		self.value = float(value) if value is not None else float(self.key)

	def copy(self):
		return HeapNode(self.key, self.value)

	def _assert_same_type(self, other):
		assert type(self) == type(other), "Expected same type but got {} and {}".format(type(self), type(other))

	def __eq__(self, other): # ==
		self._assert_same_type(other)
		return self.value == other.value

	def __ne__(self, other): # !=
		self._assert_same_type(other)
		return self.value != other.value

	def __lt__(self, other): # <
		self._assert_same_type(other)
		return self.value < other.value

	def __le__(self, other): # <=
		self._assert_same_type(other)
		return self.value <= other.value

	def __gt__(self, other): # >
		self._assert_same_type(other)
		return self.value > other.value

	def __ge__(self, other): # >=
		self._assert_same_type(other)
		return self.value >= other.value

	def __repr__(self):
		return 'HeapNode("{}", {})'.format(self.key, self.value)

	def __str__(self):
		return '<{}, {}>'.format(self.key, self.value)


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
			satsifies = self._better_than(Heap.parent(i), i) or self.array[Heap.parent(i)] == self.array[i]
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
	def sort(cls, elements, reverse = False):
		if len(elements) <= 1:
			return list(elements)
		
		t = type(elements[0])
		heap = MaxHeap() if reverse else MinHeap()
		heap.array = list(map(HeapNode, elements))
		heap.build()
		sorted_elements = []
		while len(heap) != 0:
			sorted_elements.append(heap.pop_best())
		return sorted_elements

	@abc.abstractmethod
	def _better_than(self, i, j):
		pass

	@abc.abstractmethod
	def _worse_than(self, i, j):
		pass

	def __len__(self):
		return len(self.array)

	def __repr__(self):
		return repr(self.array)

	def __str__(self):
		return str(self.array)

	def heapify(self, i):
		L, R = Heap.left(i), Heap.right(i)
		# determine if left or i is best
		if self._in_bounds(L) and self._better_than(L, i):
			best = L
		else:
			best = i
		# now update if right is even better
		if self._in_bounds(R) and self._better_than(R, best):
			best = R
		# finally, swap i with the best element unless it is i
		if best != i:
			self._swap(i, best)
			self.heapify(best)

	def build(self):
		for i in range((len(self) - 1) // 2, -1, -1):
			self.heapify(i)

	def _best(self):
		if len(self) == 0:
			raise Exception('No best element in empty heap')
		return self.array[0]

	def best(self):
		return self._best().key

	def pop_best(self):
		best = self._best()
		n = len(self)
		# now swap best and last elements
		self._swap(0, n-1)
		# then remove the last element
		self.array.pop()
		# and finally max heapify the root
		self.heapify(0)
		return best.copy().key

	def insert(self, key, value = None):
		self.array.append(None)
		self.set(len(self) - 1, key, value)

	def set(self, i, key, value = None):
		node = HeapNode(key, value)
		self.array[i] = node
		# while i > 0 and self.array[Heap.parent(i)] > self.array[i]:
		while i > 0 and self._worse_than(Heap.parent(i), i):
			self._swap(i, Heap.parent(i))
			i = Heap.parent(i)



class MaxHeap(Heap):

	def _better_than(self, i, j):
		return self.array[i] > self.array[j]

	def _worse_than(self, i, j):
		return self.array[i] < self.array[j]



class MinHeap(Heap):
	
	def _better_than(self, i, j):
		return self.array[i] < self.array[j]

	def _worse_than(self, i, j):
		return self.array[i] > self.array[j]

# ------------------------------------------ TEST

def test_heap_node():
	a10 = HeapNode('A', 10)
	assert 'A' == a10.key
	assert 10 == a10.value
	assert '<A, 10.0>' == str(a10)
	assert a10 == eval(repr(a10))
	assert not a10 != eval(repr(a10))

	zz6 = HeapNode('zz', 6)
	assert 'zz' == zz6.key
	assert 6 == zz6.value
	assert '<zz, 6.0>' == str(zz6)
	assert zz6 == eval(repr(zz6))
	assert not zz6 != eval(repr(zz6))
	
	minus_one = HeapNode(-1)
	assert -1 == minus_one.key
	assert -1 == minus_one.value
	assert '<-1, -1.0>' == str(minus_one)
	assert minus_one == eval(repr(minus_one))
	assert not minus_one != eval(repr(minus_one))

	assert minus_one < zz6 < a10
	assert minus_one <= minus_one <= zz6 <= zz6 <= a10 <= a10
	assert a10 > zz6 > minus_one
	assert a10 >= a10 >= zz6 >= zz6 >= minus_one >= minus_one
	assert minus_one != zz6 != a10

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
	max_heap.array = list(map(HeapNode, [-57, -2, -1, 0, 1, 2]))
	max_heap.build()
	assert list(map(HeapNode, [2, 1, -1, 0, -2, -57])) == max_heap.array
	assert 2   == max_heap.pop_best()
	assert 1   == max_heap.pop_best()
	assert 0   == max_heap.pop_best()
	assert -1  == max_heap.pop_best()
	assert -2  == max_heap.pop_best()
	assert -57 == max_heap.pop_best()
	assert 0   == len(max_heap)

	# test custom key, value with set
	max_heap = MaxHeap()
	max_heap.insert('A', 30) 
	max_heap.insert('B', 20) 
	max_heap.insert('C', 60) 
	
	assert  3  == len(max_heap)
	assert 'C' == max_heap.best()
	assert 'C' == max_heap.pop_best()

	assert  2  == len(max_heap)
	assert 'A' == max_heap.best()
	assert 'A' == max_heap.pop_best()
	
	assert  1  == len(max_heap)
	assert 'B' == max_heap.best()
	assert 'B' == max_heap.pop_best()
	assert  0  == len(max_heap)

	max_heap.insert('1', 999) 
	max_heap.insert('2', 6) 
	max_heap.insert(0, -1) 

	max_heap.set(max_heap.array.index(HeapNode(0, -1)), 0, 1000)

	assert 3 == len(max_heap)
	assert 0 == max_heap.best()
	assert 0 == max_heap.pop_best()

	assert 2   == len(max_heap)
	assert '1' == max_heap.best()
	assert '1' == max_heap.pop_best()

	assert 1   == len(max_heap)
	assert '2' == max_heap.best()
	assert '2' == max_heap.pop_best()
	assert 0   == len(max_heap)


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
	min_heap.array = list(map(HeapNode, [-1,0,-2,0,1,3]))
	min_heap.build()
	assert list(map(HeapNode, [-2, 0, -1, 0, 1, 3])) == min_heap.array
	assert -2 == min_heap.pop_best()
	assert -1 == min_heap.pop_best()
	assert 0  == min_heap.pop_best()
	assert 0  == min_heap.pop_best()
	assert 1  == min_heap.pop_best()
	assert 3  == min_heap.pop_best()

# ------------------------------------------ MAIN

def main():
	test_heap_node()
	print('Heap node tests pass')

	test_heap()
	print('Heap tests pass')

	test_max_heap()
	print('Max heap tests pass')
	
	test_min_heap()
	print('Min heap tests pass')

if __name__ == '__main__':
	main()
