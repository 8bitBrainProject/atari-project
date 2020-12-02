import random

class RingBuffer:

    def __init__(self, size):
        
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, element):
        
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)

        if (self.end == self.start):
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
    
        if (self.end < self.start):
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):

        for i in range(len(self)):
            yield self[i]

    def sample_random_batch(self, sample_size):

        
        if (sample_size >= self.__len__()):
            return self.data[len(self.data) - sample_size : ]
        else:
            start_index = random.randrange(0, (self.__len__() - sample_size))
            return self.data[(self.start + start_index) % len(self.data) : 
                             (self.start + start_index) % len(self.data) + sample_size]

class ReplayMemory:

    def __init__(self, size):
        self.data = [None] * (size)
        self.size = size
        self.len = 0

    def append(self, element):    
        
        self.data.append(element)
        removed_element = self.data.pop(0)

        if (removed_element == None):
            self.len += 1

    def random_sample_batch(self, sample_size):
        
        if (sample_size > self.len):
            return self.data
        else:
            start_index = random.randrange(0, (self.len - sample_size + 1))
            return self.data[start_index : (start_index + sample_size)]
            
