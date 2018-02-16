import json
from collections import Counter, namedtuple
from multiprocessing import Pool, cpu_count
from os.path import getsize
from time import clock

__author__ = 'Ettore Forigo'
__license__ = 'GPL'
__date__ = '04/06/2017'
__version__ = '1'
__status__ = 'Development'


MULTIPROCESSING_THRESHOLD = 200000


class HuffmanHeap(list):
	Element = namedtuple('Element', ('weight', 'encodings'))

	def push(self, weight, encodings):
		element = HuffmanHeap.Element(weight, encodings)

		if not self or weight >= self[-1].weight:
			self.append(element)
			return

		for i in range(len(self)):
			if weight < self[i].weight:
				self.insert(i - 1, element)
				return


def create_encoding_table(bytes_):
	heap = HuffmanHeap()

	for byte, count in Counter(bytes_).items():
		heap.push(count, {byte: ''})

	while len(heap) - 1:
		lo, hi = heap.pop(0), heap.pop(0)

		for byte in lo.encodings:
			lo.encodings[byte] = '0' + lo.encodings[byte]
		for byte in hi.encodings:
			hi.encodings[byte] = '1' + hi.encodings[byte]

		heap.push(lo.weight + hi.weight, {**lo.encodings, **hi.encodings})

	return heap[0].encodings


def segment(array, parts):
	avg = len(array) / parts
	last = 0.0

	while last < len(array):
		yield array[int(last):int(last + avg)]
		last += avg


def encoding_worker(data, table):
	return ''.join(table[byte] for byte in data)


def encode(data, table, pool):
	tasks = [pool.apply_async(encoding_worker, args=(part, table)) for part in segment(data, cpu_count())]
	del data, table
	return ''.join(task.get() for task in tasks)


def segment_bytewise(string, parts):
	avg = len(string) / parts
	avg = avg + (8 - avg % 8)
	last = 0

	while last < len(string):
		yield string[int(last):int(last + avg)]
		last += avg


def pack_bytes_worker(string):
	return bytearray(int(string[i:i + 8], 2) for i in range(0, len(string), 8))


def pack_bytes(string, pool):
	tasks = [pool.apply_async(pack_bytes_worker, args=(part,))
	         for part in segment_bytewise(string + '0' * (8 - len(string) % 8), cpu_count())]
	del string
	return bytearray(b''.join(task.get() for task in tasks))


def encode_file(filename, table_filename, compressed_filename, encoding='UTF-8', silent=False):
	start_time = clock()

	silent or print('Reading... ', end='')
	with open(filename, 'rb') as file:
		uncompressed_bytes = file.read()
	silent or print('Done')

	if not uncompressed_bytes or len(uncompressed_bytes) == 1:
		raise ValueError("Not enough data to compress")

	silent or print('Creating tables... ', end='')
	encoding_table = create_encoding_table(uncompressed_bytes)
	decoding_table = {value: key for key, value in encoding_table.items()}
	decoding_table['size'] = len(uncompressed_bytes)
	silent or print('Done')

	silent or print('Writing table... ', end='')
	with open(table_filename, 'w+', encoding=encoding) as file:
		json.dump(decoding_table, file)
	silent or print('Done')

	del decoding_table

	pool = Pool(cpu_count())

	silent or print('Compressing... ', end='')
	if len(uncompressed_bytes) < MULTIPROCESSING_THRESHOLD:
		compressed_string = encoding_worker(uncompressed_bytes, encoding_table)
	else:
		silent or print('using multiprocessing... ', end='')
		compressed_string = encode(uncompressed_bytes, encoding_table, pool)
	silent or print('Done')

	del uncompressed_bytes, encoding_table

	silent or print('Preparing data using multiprocessing... ', end='')
	compressed_bytes = pack_bytes(compressed_string, pool)
	silent or print('Done')

	del compressed_string
	pool.close()
	pool.terminate()

	silent or print('Writing... ', end='')
	with open(compressed_filename, 'wb+') as file:
		file.write(compressed_bytes)
	silent or print('Done')
	silent or print('Encoded in {:.4f} seconds, {:.2%} Space savings'.format(
		clock() - start_time,
		1 - (getsize(compressed_filename) + getsize(table_filename)) / getsize(filename))
	)


def unpack_bytes_worker(bytes_):
	return ''.join('{0:08b}'.format(byte) for byte in bytes_)


def unpack_bytes(bytes_, pool):
	tasks = [pool.apply_async(unpack_bytes_worker, args=(part,)) for part in segment(bytes_, cpu_count())]
	del bytes_
	return ''.join(task.get() for task in tasks)


def decode(string, table, size):
	uncompressed_bytes = b''
	string = iter(string)

	buffer = ''
	while len(uncompressed_bytes) < size:
		buffer += next(string)
		if buffer in table:
			uncompressed_bytes += bytes([table[buffer]])
			buffer = ''

	return uncompressed_bytes


def decode_file(filename, table_filename, uncompressed_filename, encoding='UTF-8', silent=False):
	start_time = clock()

	silent or print('Reading file...', end='')
	with open(filename, 'rb') as file:
		compressed_bytes = file.read()
	silent or print('Done')

	pool = Pool(cpu_count())

	silent or print('Preparing data using multiprocessing... ', end='')
	compressed_string = unpack_bytes(compressed_bytes, pool)
	silent or print('Done')

	pool.close()
	pool.terminate()

	silent or print('Reading table... ', end='')
	with open(table_filename, encoding=encoding) as file:
		decoding_table = json.load(file)

	size = decoding_table.pop('size')
	silent or print('Done')

	silent or print('Uncompressing...', end='')
	uncompressed_bytes = decode(compressed_string, decoding_table, size)
	silent or print('Done')

	silent or print('Writing... ', end='')
	with open(uncompressed_filename, 'wb+') as file:
		file.write(uncompressed_bytes)
	silent or print('Done')
	silent or print('Decoded in {:.4f} seconds'.format(clock() - start_time))


def main():
	in_name = 'in.bmp'
	table_name = 'table.json'
	compressed_name = 'compressed.bin'
	out_name = 'out.bmp'

	encode_file(in_name, table_name, compressed_name)
	decode_file(compressed_name, table_name, out_name)

	with open(in_name, 'rb') as in_, open(out_name, 'rb') as out:
		assert in_.read() == out.read()


if __name__ == '__main__':
	main()
