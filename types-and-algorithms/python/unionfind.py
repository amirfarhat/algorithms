import abc


# ABC & HELPER CLASSES
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////


class UnionFindABC(abc.ABC):
	@abc.abstractmethod
	def make_set(self, x, value = None):
		pass
	
	@abc.abstractmethod
	def union(self, x, y):
		pass
	
	@abc.abstractmethod
	def find_set(self, x):
		pass


class UFNode:
	def __init__(self, key):
		self.key = str(key)

	def copy(self):
		return UFNode(self.key)

	def _compare(self, other, comparator):
		if not isinstance(other, UFNode):
			raise TypeError('Cannot compare UnionFind nodes with non-UnionFind nodes')
		return comparator(self.key, other.key)

	def __eq__(self, other): # ==
		return self._compare(other, lambda a, b : a == b)

	def __ne__(self, other): # !=
		return self._compare(other, lambda a, b : a != b)

	def __lt__(self, other): # <
		return self._compare(other, lambda a, b : a < b)

	def __le__(self, other): # <=
		return self._compare(other, lambda a, b : a <= b)

	def __gt__(self, other): # >
		return self._compare(other, lambda a, b : a > b)

	def __ge__(self, other): # >=
		return self._compare(other, lambda a, b : a >= b)

	def __hash__(self):
		return hash(self.key)

	def __str__(self):
		return self.key

	__repr__ = __str__


# UNION-FIND DATA STRUCTURE
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////


class UnionFind(UnionFindABC):
	def __init__(self):
		self._elt_to_node = dict()
		self._node_to_set = dict()
		self.sets = []

	def _get_node(self, x):
		# get the node object corresponding to x
		if x not in self._elt_to_node:
			raise KeyError('{} not found in this UnionFind'.format(str(x)))
		return self._elt_to_node[x]

	def _find_set(self, x):
		# get the set containing the element x
		node = self._get_node(x)
		return self._node_to_set[node] 

	def make_set(self, x):
		"""
		Makes a new set to contain the element `x`,
		which is assumed to not already be part of this
		UnionFind
		"""
		# do nothing if x already in this UnionFind
		if x in self._elt_to_node:
			return
		
		# make a node and a set for x
		node = UFNode(x)
		new_set = { node }
		self._elt_to_node[x] = node
		self._node_to_set[node] = new_set

		# keep track of this new set
		self.sets.append(new_set)
			
	def find_set(self, x):
		"""
		Returns the set in which the element `x` is contained.
		Assumes `x` is already part of this UnionFind
		"""
		node_set = self._find_set(x)
		return { n.copy().key for n in node_set }

	__getitem__ = find_set
	
	def union(self, x, y):
		"""
		Combines the set 
		"""
		# get node and set that contains it for both x and y
		nx, sx = self._get_node(x), self._find_set(x)
		ny, sy = self._get_node(y), self._find_set(y)

		# only union if nx and ny in different sets
		if sx is not sy:
			# apply union by rank - add smaller set to larger one
			large, small = (sx, sy) if len(sx) > len(sy) else (sy, sx)
			for n in small:
				large.add(n)
				self._node_to_set[n] = large
			self.sets.remove(small)

	def count_items(self):
		"""
		Return the number of items stored in this UnionFind
		"""
		return len(self._elt_to_node)

	__hash__ = None

	def __len__(self):
		return len(self.sets)

	def __iter__(self):
		for s in self.sets:
			yield { n.key for n in s }

	def __str__(self):
		c = ', '
		inside = c.join('{' + c.join(str(i) for i in sorted(s)) + '}' for s in sorted(self.sets))
		return '{' + inside + '}'

	__repr__ = __str__


# TESTS
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////


def test_union_find():
	UF = UnionFind()

	# {}
	assert 0 == len(UF)
	assert 0 == UF.count_items()
	assert '{}' == str(UF)

	# { {A} }
	for _ in range(2):
		UF.make_set('A')
		assert 1 == len(UF)
		assert 1 == UF.count_items()
		assert '{{A}}' == str(UF)
		assert {'A'} == UF.find_set('A')
		assert [{'A'}] == sorted(UF)

	# { {A}, {B} }
	UF.make_set('B')
	assert 2 == len(UF)
	assert 2 == UF.count_items()
	assert '{{A}, {B}}' == str(UF)
	assert {'A'} == UF['A']
	assert {'B'} == UF['B']
	assert [{'A'}, {'B'}] == sorted(UF)	
	
	# { {A}, {B}, {C} }
	UF.make_set('C')
	assert 3 == len(UF)
	assert 3 == UF.count_items()
	assert '{{A}, {B}, {C}}' == str(UF)
	assert {'A'} == UF['A']
	assert {'B'} == UF['B']
	assert {'C'} == UF['C']
	assert [{'A'}, {'B'}, {'C'}] == sorted(UF)	
	
	# { {A, B}, {C} }
	UF.union('A', 'B')
	assert 2 == len(UF)
	assert 3 == UF.count_items()
	assert '{{A, B}, {C}}' == str(UF)
	assert {'A', 'B'} == UF['A']
	assert {'A', 'B'} == UF['B']
	assert {'C'} == UF['C']
	assert [{'A', 'B'}, {'C'}] == sorted(UF)	

	# { {A, B}, {C}, {D} }
	UF.make_set('D')
	assert 3 == len(UF)
	assert 4 == UF.count_items()
	assert '{{A, B}, {C}, {D}}' == str(UF)
	assert {'A', 'B'} == UF['A']
	assert {'A', 'B'} == UF['B']
	assert {'C'} == UF['C']
	assert {'D'} == UF['D']
	assert [{'A', 'B'}, {'C'}, {'D'}] == sorted(UF)

	# { {A, B}, {C}, {D}, {E} }
	UF.make_set('E')
	assert 4 == len(UF)
	assert 5 == UF.count_items()
	assert '{{A, B}, {C}, {D}, {E}}' == str(UF)
	assert {'A', 'B'} == UF['A']
	assert {'A', 'B'} == UF['B']
	assert {'C'} == UF['C']
	assert {'D'} == UF['D']
	assert {'E'} == UF['E']
	assert [{'A', 'B'}, {'C'}, {'D'}, {'E'}] == sorted(UF)

	# { {A, B}, {C, E}, {D} }
	UF.union('E', 'C')
	assert 3 == len(UF)
	assert 5 == UF.count_items()
	assert '{{A, B}, {C, E}, {D}}' == str(UF)
	assert {'A', 'B'} == UF['A']
	assert {'A', 'B'} == UF['B']
	assert {'C', 'E'} == UF['C']
	assert {'D'} == UF['D']
	assert {'C', 'E'} == UF['E']
	assert [{'A', 'B'}, {'C', 'E'}, {'D'}] == sorted(UF)

	# { {A, B, D}, {C, E} }
	UF.union('B', 'D')
	assert 2 == len(UF)
	assert 5 == UF.count_items()
	assert '{{A, B, D}, {C, E}}' == str(UF)
	assert {'A', 'B', 'D'} == UF['A']
	assert {'A', 'B', 'D'} == UF['B']
	assert {'C', 'E'} == UF['C']
	assert {'A', 'B', 'D'} == UF['D']
	assert {'C', 'E'} == UF['E']
	assert [{'A', 'B', 'D'}, {'C', 'E'}] == sorted(UF)

	# { {A, B, C, D, E} }
	UF.union('C', 'A')
	assert 1 == len(UF)
	assert 5 == UF.count_items()
	assert '{{A, B, C, D, E}}' == str(UF)
	assert {'A', 'B', 'C', 'D', 'E'} == UF['A']
	assert {'A', 'B', 'C', 'D', 'E'} == UF['B']
	assert {'A', 'B', 'C', 'D', 'E'} == UF['C']
	assert {'A', 'B', 'C', 'D', 'E'} == UF['D']
	assert {'A', 'B', 'C', 'D', 'E'} == UF['E']
	assert [{'A', 'B', 'C', 'D', 'E'}] == sorted(UF)


if __name__ == '__main__':
	try:
		test_union_find()
		print('UnionFind tests pass')
	except AssertionError as e:
		print('UnionFind tests fail')
		raise e
