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
    
    def get_with_prob(self, hash_key: int, current_network, target_network, compute_td_loss) -> np.ndarray:
        # get a value by key index
        index_key = hash_key % len(self.buffer)

        if self.buffer[index_key] is None:
            # this state is not previously stored
            raise KeyError()
        
        else:
            # return all transitions that has the same current state as the key
            if len(self.buffer[index_key][0]) == 2:
                selected_transition = self.buffer[index_key][0][1]

                # re-fill the buffer with the copied buffer for the corresponding state transitions
                self.buffer[index_key] = copy.deepcopy(self.copied_buffer[index_key])

            # buffer has all transitions for the current state, pop each consecutive transition for the current state
            else:
                td_errors = []

                # return all transitions that has the same current state as the key
                all_transitions = self.buffer[index_key][0][1:]

                states = torch.from_numpy(np.stack([e.state for e in all_transitions if e is not None])).float().to(self.device)
                actions = torch.from_numpy(np.vstack([e.action for e in all_transitions if e is not None])).long().to(self.device)
                rewards = torch.from_numpy(np.vstack([e.reward for e in all_transitions if e is not None])).float().to(self.device)
                next_states = torch.from_numpy(np.stack([e.next_state for e in all_transitions if e is not None])).float().to(self.device)
                dones = torch.from_numpy(np.vstack([e.done for e in all_transitions if e is not None]).astype(np.uint8)).float().to(self.device)

                all_transitions = (states, actions, rewards, next_states, dones)

                # calculate td error with the given q value functions for corresponding states' transitions
                td_errors = compute_td_loss(current_network, target_network, all_transitions)

                # filter out negative td-losses
                td_errors[td_errors < 0.0] = 0.0

                td_errors = td_errors.detach().cpu().numpy()
                td_errors = np.reshape(td_errors, (len(td_errors)))

                # calculate probability distribution of td-errors
                probs = td_errors / sum(td_errors)

                # buffer has all transitions for the current state, pop with td_error probability
                #poping_id = np.random.choice(range(1, len(self.buffer[index_key][0])), p=probs)
                
                poping_id = int(np.argmin(td_errors, axis=0)) + 1
                # poping_id = random.randint(1, len(self.buffer[index_key][0]) - 1)

                #print("poping_id : ", type(poping_id), type(np.argmax(td_errors, axis=0)), poping_id, np.argmax(td_errors, axis=0))

                # randomly pop indices and remove edges from the tree list
                selected_transition = self.buffer[index_key][0].pop(poping_id)
                
            return selected_transition
