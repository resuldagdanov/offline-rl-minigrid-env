import copy
import numpy as np


class HashTable(object):

    def __init__(self, buffer_size: int) -> None:
        # initiate our buffer with empty transitions
        self.buffer = [None] * buffer_size
        self.copied_buffer = [None] * buffer_size

    def __setitem__(self, hash_key: int, state: np.ndarray) -> None:
        self.add(hash_key, state)
    
    def __getitem__(self, hash_key: int) -> np.ndarray:
        # retreats state array of this state hash key
        return self.get(hash_key)
    
    def hash_index(self, hash_key: int) -> int:
        # get the index of our array for a specific string key
        return hash_key % len(self.buffer)
    
    def get_with_key(self, hash_key: int) -> np.ndarray:
        # queue transition when hash key is given
        index = hash_key % len(self.buffer)
        return self.buffer[index][0][0], self.buffer[index][0][1]
    
    def add(self, hash_key: int, state: np.ndarray) -> None:
        # add a value to our array by its key
        index_key = self.hash_index(hash_key=hash_key)

        if self.buffer[index_key] is None:
            # this index is empty and we should initiate a list and append our key-value-pair to it
            self.buffer[index_key] = state

    def get(self, hash_key: int) -> np.ndarray:
        # get a value to our array by its key
        index_key = self.hash_index(hash_key=hash_key)

        if self.buffer[index_key] is not None:
            # return state array of a unique hash key corresponding to this state
            return self.buffer[index_key]

        else:
            # this state is not previously stored
            raise KeyError()

    def is_full(self) -> bool:
        # determines if the hash-table is too populated
        items = 0

        # count how many indexes in our array that is populated with transitions
        for item in self.buffer:
            if item is not None:
                items += 1
        
        # return bool value based on if the amount of populated items are more than half the length of the list
        return items > len(self.buffer) / 2
    
    def double(self) -> None:
        # double the list length and re-add values
        ht2 = HashTable(buffer_size=len(self.buffer)*2)

        for i in range(len(self.buffer)):
            if self.buffer[i] is None:
                continue
            
            # since our list is now a different length, we need to re-add all of our values to 
            # the new list for its hash to return correct index
            for kvp in self.buffer[i]:
                ht2.add(kvp[0], kvp[1])
        
        # finally we just replace our current list with the new list of values that we created in ht
        self.buffer = ht2.array
    
    def save_buffer(self) -> None:
        # saving replay buffer transitions to be resetted later
        self.copied_buffer = copy.deepcopy(self.buffer)
    