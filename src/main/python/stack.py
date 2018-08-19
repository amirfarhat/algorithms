
# ------------------------------------------ STACK

class Stack:
	
	def __init__(self):
		self.array = list()

	@classmethod
	def of(cls, *args):
		stack = Stack()
		for arg in args:
			stack.push(arg)
		return stack

	def push(self, element):
		self.array.append(element)

	def pop(self):
		if self.is_empty():
			raise Exception('Cannot pop from an empty stack')
		return self.array.pop()

	def is_empty(self):
		return len(self.array) == 0

	def __len__(self):
		return len(self.array)

	def __repr__(self):
		return repr(self.array)

	def __str__(self):
		return str(self.array)

# ------------------------------------------ TEST

def test_stack_is_empty():
	stack = Stack()

	# stack size: 0
	assert Stack().is_empty()
	assert Stack.of().is_empty()

	# stack size: 1
	stack.array = [3.14159]
	assert not stack.is_empty()
	assert not Stack.of(-0.618).is_empty()

	# stack size: 2
	stack.array = [5, 6]
	assert not stack.is_empty()
	assert not Stack.of(0, 0).is_empty()

	# stack size: 3
	stack.array = [3, 6, -9]
	assert not stack.is_empty()
	assert not Stack.of(275634, 243685, -28367).is_empty()

def test_stack_push():
	stack = Stack()

	# count push elements: 1
	stack.push(15)
	assert [15] == stack.array 

	# count push elements: 3
	stack.push(0)
	stack.push(0) # intentional duplicate
	stack.push(6)
	assert [15, 0, 0, 6] == stack.array

	# count push elements: 6
	stack.push(-1)
	stack.push(-2)
	stack.push(-3)
	stack.push(-4)
	stack.push(-5)
	stack.push(-6132865346)
	assert [15, 0, 0, 6, -1, -2, -3, -4, -5, -6132865346] == stack.array

def test_stack_pop():
	stack = Stack()
	stack.array = [0, 2, 4, 6, 8, 9, -1000]
	assert -1000 == stack.pop()
	assert 9     == stack.pop()
	assert 8     == stack.pop()
	assert 6     == stack.pop()
	assert 4     == stack.pop()
	assert 2     == stack.pop()
	assert 0     == stack.pop()

# ------------------------------------------ MAIN

def main():
	test_stack_push()
	print("Stack push tests pass")

	test_stack_pop()
	print("Stack pop tests pass")

	test_stack_is_empty()
	print("Stack is_empty tests pass")

if __name__ == '__main__':
	main()

