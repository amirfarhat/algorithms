
# ------------------------------------------ STACK

class Stack:
	
	def __init__(self):
		self.array = list()

	@classmethod
	def of(cls, *args):
		stack = Stack()
		assert isinstance(args, (list, tuple, iter))
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

def test_stack_length():
	# stack length: 0
	assert 0 == len(Stack())
	assert 0 == len(Stack.of())

	# stack length: 1
	length_one_stack = Stack()
	length_one_stack.push(7331)
	assert 1 == len(length_one_stack) == len(Stack.of(-10287.236))

	# stack length: 2
	length_two_stack = Stack()
	length_two_stack.push(-163)
	length_two_stack.push(36389)
	assert 2 == len(length_two_stack) == len(Stack.of(-0, +0))

	# stack length: 3
	length_three_stack = Stack()
	length_three_stack.push("one")
	length_three_stack.push("three")
	length_three_stack.push("two")
	assert 3 == len(length_three_stack) == len(Stack.of("good", "day", "sir"))

	# stack length: 4
	length_four_stack = Stack()
	length_four_stack.push(True)
	length_four_stack.push(True)
	length_four_stack.push(True)
	length_four_stack.push(False)
	assert 4 == len(length_four_stack) == len(Stack.of("i", "like", "trains", "choo-choo"))

	# stack length: 50
	length_fifty_stack = Stack()
	for i in range(50):
		length_fifty_stack.push(i ** 2)
	assert 50 == len(length_fifty_stack)

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

def test_static_stack_factory_method():
	# stack size: 0
	stack = Stack.of()
	assert stack.is_empty()
	assert 0 == len(stack) 
	assert [] == stack.array
	
	# stack size: 1
	stack = Stack.of("hello")
	assert not stack.is_empty()
	assert 1 == len(stack)
	assert ["hello"] == stack.array
	assert "hello" == stack.pop()

	# stack size: > 1 
	stack = Stack.of(-2, -1, 0, 1, 2)
	assert not stack.is_empty()
	assert 5 == len(stack)
	assert [-2, -1, 0, 1, 2] == stack.array
	assert 2 == stack.pop()

	# stack size: >> 1 
	stack = Stack.of(1,2,3,4,5,6,7,8,9)
	assert not stack.is_empty()
	assert 9 == len(stack)
	assert [1,2,3,4,5,6,7,8,9] == stack.array
	stack.push(10)
	assert 10 == len(stack)
	assert [1,2,3,4,5,6,7,8,9,10] == stack.array
	assert 10 == stack.pop()
	assert 9 == len(stack)
	assert [1,2,3,4,5,6,7,8,9] == stack.array

# ------------------------------------------ MAIN

def main():
	test_stack_push()
	print("Stack push tests pass")

	test_stack_pop()
	print("Stack pop tests pass")

	test_stack_length()
	print("Stack length tests pass")

	test_static_stack_factory_method()
	print("Stack static factory method tests pass")

	test_stack_is_empty()
	print("Stack is_empty tests pass")

if __name__ == '__main__':
	main()

