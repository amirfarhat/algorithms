import random
from collections import deque

# dimension of the maze
N = 50

# meaning of cell values
FREE = '-'
WALL = '|'
START = 'S'
END = 'E'
PATH = '@'

# lower and upper bound on number of walls
MIN_WALLS = max(N, N // 2)
MAX_WALLS = N*N // 2


def make_maze():
	# get random number of walls between MIN_WALLS and MAX_WALLS
	num_walls = random.randint(MIN_WALLS, MAX_WALLS)

	# initialize maze to be all free spaces
	maze = [[FREE for j in range(N)] for i in range(N)]

	# get `num_walls` many randomly distributed 
	# (i, j) coordinates and place a wall there
	wall_spots = set()
	while len(wall_spots) < num_walls:
		i = random.randint(0, N - 1)
		j = random.randint(0, N - 1)
		wall_spots.add((i, j))
		maze[i][j] = WALL

	# choose a starting spot (si, sj)
	si, sj = 0, 0
	maze[si][sj] = START

	# choose an ending spot (ei, ej)
	ei, ej = N - 1, N - 1
	maze[ei][ej] = END

	return maze


def pretty(maze):
	return '\n'.join(' '.join(str(cell) for cell in row) for row in maze) + '\n'


def bfs(nodes, edges, start, end):
	q = deque()
	q.append([ start ])
	seen = { start }
	while len(q) > 0:
		path = q.popleft()
		for neighbor in edges[path[-1]]:
			if neighbor in seen: continue
			seen.add(neighbor)
			npath = path + [ neighbor ]
			if neighbor == end:
				return npath
			q.append(npath)
	return None


def solvable(maze):
	# get all free spot coordinates (i, j) 
	# and record start and end coords
	start, end = None, None
	free_spots = set()
	for i in range(N):
		for j in range(N):
			if maze[i][j] != WALL:
				free_spots.add((i, j))

			if maze[i][j] == START:
				start = (i, j)

			if maze[i][j] == END:
				end = (i, j)
	assert start is not None
	assert end is not None

	# map all free spots to neighboring spots
	free_neighbors = dict()
	for fi, fj in free_spots:
		free_neighbors.setdefault((fi, fj), set())
		for ni, nj in [(fi + 1, fj), (fi - 1, fj), (fi, fj + 1), (fi, fj - 1)]:
			if (ni, nj) in free_spots:
				free_neighbors[(fi, fj)].add((ni, nj))
				free_neighbors.setdefault((ni, nj), set()).add((fi, fj))

	# nodes are free spots, and neighbors are other free spots
	nodes = free_spots
	edges = free_neighbors

	# do bfs from start to end to detect a path
	path = bfs(nodes, edges, start, end)

	if path is not None:
		maze_copy = [[cell for cell in row] for row in maze]
		for i, j in path[1:-1]:
			maze_copy[i][j] = PATH
		return (path is not None, maze_copy)
	return (path is not None, None)


if __name__ == '__main__':
	maze = make_maze()
	print(pretty(maze))
	
	can_solve, solution = solvable(maze)

	if can_solve:
		print('MAZE IS SOLVABLE')
		print()
		print(pretty(solution))
	else:
		print('MAZE IS NOT SOLVABLE')