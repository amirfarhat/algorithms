
# ------------------------------------------ QUEUE

class Queue:

	def __init__(self):
		self.array = []
		self.head = None

	def _clear(self):
		self.array = []
		self.head = None

	@classmethod
	def of(cls, *args):
		queue = Queue()
		for arg in args:
			queue.enqueue(arg)
		return queue

	def enqueue(self, element):
		if self.head == None:
			self.head = 0
		self.array.append(element)

	def enqueue_all_of(self, elements):
		for element in elements:
			self.enqueue(element)

	def dequeue(self): # does not remove the element, just returns it
		if self.head == None:
			raise Exception('Cannot dequeue from an empty queue')
		dequeued_element = self.array[self.head]
		self.head = (self.head + 1) % len(self.array)
		return dequeued_element

	def is_empty(self):
		return len(self.array) == 0

	def __len__(self):
		return len(self.array)

	def __repr__(self):
		return repr(self.array)

	def __str__(self):
		return str(self.array)

# ------------------------------------------ TEST

def test_queue_enqueue():
	queue = Queue()
	assert []   == queue.array
	assert None == queue.head

	queue.enqueue(5)
	assert [5] == queue.array
	assert 0   == queue.head

	queue.enqueue(3)
	queue.enqueue(4)
	queue.enqueue(-5)
	assert [5, 3, 4, -5] == queue.array
	assert 0 == queue.head

def test_queue_enqueue_all_of():
	# queue size: 0
	queue = Queue()
	queue.enqueue_all_of([3,1,2])
	assert 3 == len(queue)
	assert not queue.is_empty()
	assert 3 == queue.dequeue()
	assert 1 == queue.dequeue()
	assert 2 == queue.dequeue()

	# queue size: 1
	queue = Queue.of(9.12)
	queue.enqueue_all_of([])
	assert 1 == len(queue)
	assert not queue.is_empty()
	assert 9.12 == queue.dequeue()
	queue.enqueue_all_of([3.14, -0.618])
	assert 3 == len(queue)
	assert 9.12 == queue.dequeue()
	assert 3.14 == queue.dequeue()
	assert -0.618 == queue.dequeue()
	assert 3 == len(queue)

	# queue size: > 1
	queue = Queue.of(1,2,3,4)
	queue.enqueue_all_of([5.0])
	assert 5 == len(queue)
	assert not queue.is_empty()
	assert [x for x in range(1, 6)] == queue.array

	# queue size: >> 1
	queue = Queue()
	for x in range(-10, 11):
		queue.enqueue(x)
	queue.enqueue_all_of([0, 0])
	assert 23  == len(queue)
	assert -10 == queue.dequeue()
	assert -9  == queue.dequeue()
	assert -8  == queue.dequeue()
	assert -7  == queue.dequeue()
	assert -6  == queue.dequeue()
	assert -5  == queue.dequeue()
	assert -4  == queue.dequeue()
	assert -3  == queue.dequeue()
	assert -2  == queue.dequeue()
	assert -1  == queue.dequeue()
	assert 0   == queue.dequeue()
	assert 1   == queue.dequeue()
	assert 2   == queue.dequeue()
	assert 3   == queue.dequeue()
	assert 4   == queue.dequeue()
	assert 5   == queue.dequeue()
	assert 6   == queue.dequeue()
	assert 7   == queue.dequeue()
	assert 8   == queue.dequeue()
	assert 9   == queue.dequeue()
	assert 10  == queue.dequeue()
	assert 0   == queue.dequeue()
	assert 0   == queue.dequeue()

def test_queue_dequeue():
	# queue size: 0
	queue = Queue()
	throws_exception = False
	try: 
		queue.dequeue()
	except Exception:
		throws_exception = True
	assert throws_exception
	assert 0 == len(queue)

	# queue size: 1
	queue = Queue.of(-982)
	assert -982 == queue.dequeue()
	assert 1 == len(queue)

	# queue size: 2
	queue = Queue.of(-1, -2)
	assert -1 == queue.dequeue()
	assert -2 == queue.dequeue()
	assert 2 == len(queue)
	
	# queue size: 3
	queue = Queue.of("b", "c", "D")
	assert "b" == queue.dequeue()
	assert "c" == queue.dequeue()
	assert "D" == queue.dequeue()
	assert 3 == len(queue)

	# queue size: 4
	queue = Queue.of(0.1, 2.3, 4.567, 8.9)
	assert 0.1   == queue.dequeue()
	assert 2.3   == queue.dequeue()
	assert 4.567 == queue.dequeue()
	assert 8.9   == queue.dequeue()
	assert 4     == len(queue)

def test_queue_length():
	# queue size: 0
	assert 0 == len(Queue()) == len(Queue.of())

	# queue size: 1
	queue = Queue()
	queue.array = [1]
	assert 1 == len(queue) == len(Queue.of(2))

	# queue size: 2
	queue.array = [314, 159]
	assert 2 == len(queue) == len(Queue.of(628, 318))

	# queue size: 3
	queue.array = [6, 6, 6]
	assert 3 == len(queue) == len(Queue.of(9, 9, 8))

	# queue size: 4
	queue.array = [1, 0, 2, -1]
	assert 4 == len(queue) == len(Queue.of("four", "beautiful", "different", "strings"))

def test_queue_static_factory_method():
	# queue size: 0
	queue = Queue()
	assert 0 == len(queue)
	assert queue.is_empty()
	assert [] == queue.array

	# queue size: 1
	queue.enqueue(901)
	assert 1 == len(queue)
	assert not queue.is_empty()
	assert [901] == queue.array
	assert 901 == queue.dequeue()
	assert 1 == len(queue)
	
	# queue size: > 1
	queue._clear()
	queue.enqueue_all_of([0, 1, 10, 100, 1000, 10000])
	assert 6 == len(queue)
	assert not queue.is_empty()
	assert [0, 1, 10, 100, 1000, 10000] == queue.array
	
	# queue size: >> 1
	queue._clear()
	queue.enqueue_all_of([x for x in range(-10**5, 10**5 + 1)])
	assert not queue.is_empty()
	assert [x for x in range(-10**5, 10**5 + 1)] == queue.array

def test_queue_is_empty():
	queue = Queue() # is empty
	assert queue.is_empty()

	queue.array = [1,2,3] # is not empty
	assert not queue.is_empty()

# ------------------------------------------ MAIN

def main():
	test_queue_enqueue()
	print('Queue enqueue tests pass')

	test_queue_enqueue_all_of()
	print('Queue enqueue_all_of tests pass')

	test_queue_dequeue()
	print('Queue dequeue tests pass')

	test_queue_length()
	print('Queue length tests pass')

	test_queue_static_factory_method()
	print('Queue static factory method tests pass')

	test_queue_is_empty()
	print('Queue is_empty tests pass')


if __name__ == '__main__':
	main()