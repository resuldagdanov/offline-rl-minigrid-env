import copy
import random
import numpy as np


class HashTable(object):

    def __init__(self, buffer_size: int) -> None:
        # initiate our buffer with empty transitions
        self.buffer = [None] * buffer_size
        self.copied_buffer = [None] * buffer_size

    def __setitem__(self, state: tuple, transition: np.ndarray) -> None:
        # state: flattened state tuple
        # transition: numpy array of (state, action, next_state, reward, terminal)
        self.add(state, transition)
    
    def __getitem__(self, state: tuple) -> list:
        # retreats transition of this state hash key
        return self.get(state)
    
    def hashing(self, state: tuple) -> int:
        # get the index of our array for a specific string key
        return hash(state) % len(self.buffer)
    
    def get_with_key(self, hash_key: int) -> np.ndarray:
        # queue transition when hash key is given
        index = hash_key % len(self.buffer)
        return self.buffer[index][0][0], self.buffer[index][0][1]
    
    def add(self, state: tuple, transition: np.ndarray) -> None:
        # add a value to our array by its key
        index_key = self.hashing(state=state)

        if self.buffer[index_key] is not None:
            # this index already contain some values; it means that this add MIGHT be an update
            # to a key that already exist, instead of just storing the value we have to first look if the key exist
            key_value = self.buffer[index_key][0]

            # if key is found, then update its current value (transition) to the new value (transition)
            if key_value[0] == state:
                # add new transition that has the same current state as an existing transition
                self.buffer[index_key][0].append(transition)

            else:
                # if no breaks was hit in the for loop, it means that no existing key was found, 
                # so we can simply just add it to the end as this will be the first transition for this state
                # NOTE: initial element is a flattened current state, so no need for else statement
                self.buffer[index_key].append([state, transition])
        
        else:
            # this index is empty. We should initiate a list and append our key-value-pair to it
            self.buffer[index_key] = []
            self.buffer[index_key].append([state, transition])

    def get(self, hash_key: int) -> list:
        # get a value by key index
        index_key = hash_key % len(self.buffer)

        if self.buffer[index_key] is None:
            # this state is not previously stored
            raise KeyError()
        
        else:
            # return all transitions that has the same current state as the key
            if len(self.buffer[index_key][0]) == 2:
                # selected_transition = self.buffer[index_key][0][1]
                selected_transition = [self.buffer[index_key][0][1]]

                # re-fill the buffer with the copied buffer for the corresponding state transitions
                self.buffer[index_key] = copy.deepcopy(self.copied_buffer[index_key])

            # buffer has all transitions for the current state, pop each consecutive transition for the current state
            else:
                poping_id = random.randint(1, len(self.buffer[index_key][0]) - 1)

                # randomly pop indices and remove edges from the tree list
                # selected_transition = self.buffer[index_key][0].pop(poping_id)
                selected_transition = list(self.buffer[index_key][0][1:])
            
            return selected_transition
    
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
