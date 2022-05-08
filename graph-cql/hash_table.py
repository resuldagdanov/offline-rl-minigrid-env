import numpy as np


class HashTable(object):

    def __init__(self, buffer_size: int):
        # initiate our buffer with empty transitions
        self.buffer = [None] * buffer_size

    def __setitem__(self, state: tuple, transition: np.ndarray):
        # state: concatted flattened state tuple
        # transition: numpy array of (state, action, next_state, reward, terminal)
        self.add(state, transition)
    
    def __getitem__(self, state: tuple):
        # retreats transition of this state hash key
        return self.get(state)
    
    def hash(self, state: tuple):
        # get the index of our array for a specific string key
        return hash(state) % len(self.buffer)
        
    def add(self, state: tuple, transition: np.ndarray):
        # add a value to our array by its key
        index_key = self.hash(state)

        if self.buffer[index_key] is not None:
            # this index already contain some values; it means that this add MIGHT be an update
            # to a key that already exist, instead of just storing the value we have to first look if the key exist
            key_value = self.buffer[index_key][0]
            
            # if key is found, then update its current value to the new value
            if key_value[0] == state:
                key_value[1] = transition

            else:
                # if no breaks was hit in the for loop, it means that no existing key was found, 
                # so we can simply just add it to the end
                self.buffer[index_key].append([state, transition])
        
        else:
            # this index is empty. We should initiate a list and append our key-value-pair to it
            self.buffer[index_key] = []
            self.buffer[index_key].append([state, transition])
    
    def get(self, state: tuple):
        # get a value by key
        index_key = self.hash(state)

        if self.buffer[index_key] is None:
            # this state is not previously stored
            raise KeyError()
        
        else:
            # loop through all key-value-pairs and find if our key exist;
            # if it does then return its value (transition)
            for key_value in self.buffer[index_key]:

                # returning a transition for corresponding hash key
                if key_value[0] == state:
                    return key_value[1]
        
        # if no return was done during loop, it means key didn't exist
        raise KeyError()
        
    def is_full(self):
        # determines if the hash-table is too populated
        items = 0

        # count how many indexes in our array that is populated with transitions
        for item in self.buffer:
            if item is not None:
                items += 1
        
        # return bool value based on if the amount of populated items are more than half the length of the list
        return items > len(self.buffer) / 2
    
    def double(self):
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
