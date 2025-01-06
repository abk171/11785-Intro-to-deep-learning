import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        Initialize instance variables.

        Parameters:
        -----------

        symbol_set [list[str]]:
            Vocabulary of all symbols (excluding the blank symbol).

        """
        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """
        Perform greedy search decoding.

        Input:
        -----

        y_probs [np.array, shape=(len(symbols) + 1, seq_length, batch_size)]
            Probability distribution for symbols. Note that the batch size is always 1 for the first part, 
            but if you plan to implement this for the second part, consider batch size.

        Returns:
        -------

        decoded_path [str]:
            Compressed sequence of symbols, i.e., sequence after removing blank symbols or repeated symbols.

        path_prob [float]:
            Forward probability of the greedy path.

        """

        decoded_path = []  # Store the decoded path
        blank = 0  # Index for the blank symbol
        path_prob = 1  # Initialize path probability to 1

        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. Update path probability by multiplying by the current maximum probability
        # 4. Select the most probable symbol and append to decoded path
        # 5. Compress the sequence (done inside or outside the loop)

        symbols_len, seq_len, batch_size = y_probs.shape  # Get the length of symbols, sequence length, and batch size
        self.symbol_set = ["-"] + self.symbol_set  # Add blank symbol to the beginning of the symbol set
        for batch_itr in range(batch_size):
            
            path = " "  # Initialize path as a space
            path_prob = 1  # Initialize path probability to 1
            for i in range(seq_len):
                max_idx = np.argmax(y_probs[:, i, batch_itr])  # Find the index of the symbol with the highest probability at the current time step
                path_prob *= y_probs[max_idx, i, batch_itr]  # Update path probability
                if path[-1] != self.symbol_set[max_idx]:  # If the current symbol differs from the previous one
                    path += self.symbol_set[max_idx]  # Add the symbol to the path
        
            path = path.replace('-', '')  # Remove blank symbols from the path
            decoded_path.append(path[1:])  # Add the path to the decoded path list

        return path[1:], path_prob  # Return the decoded path and path probability


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """
        Initialize instance variables.

        Parameters:
        -----------

        symbol_set [list[str]]:
            Vocabulary of all symbols (excluding the blank symbol).

        beam_width [int]:
            Beam width used to select the top k hypotheses for extension.

        """
        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        Perform beam search decoding.

        Input:
        -----

        y_probs [np.array, shape=(len(symbols) + 1, seq_length, batch_size)]
            Probability distribution for symbols. Note that the batch size is always 1 for the first part, 
            but if you plan to implement this for the second part, consider batch size.

        Returns:
        -------
        
        forward_path [str]:
            Sequence of symbols with the best path score (forward probability).

        merged_path_scores [dict]:
            All final merged paths and their scores.

        """
        self.symbol_set = ['-'] + self.symbol_set  # Add blank symbol to the beginning of the symbol set
        symbols_len, seq_len, batch_size = y_probs.shape  # Get the length of symbols, sequence length, and batch size
        bestPaths = dict()  # Store current best paths
        tempBestPaths = dict()  # Store temporary best paths
        bestPaths['-'] = 1  # Initialize best path as the blank symbol with a score of 1

        # Iterate over sequence length
        for t in range(seq_len):
            sym_probs = y_probs[:, t]  # Get symbol probability distribution at the current time step
            tempBestPaths = dict()  # Reset temporary paths

            # Iterate over current best paths
            for path, score in bestPaths.items():

                # Iterate over all symbols
                for r, prob in enumerate(sym_probs):
                    new_path = path  # Initialize new path as the current path

                    # Update the new path
                    if path[-1] == '-':  # If the last symbol in the current path is a blank symbol
                        new_path = new_path[:-1] + self.symbol_set[r]  # Replace the last symbol with the current symbol
                    elif (path[-1] != self.symbol_set[r]) and not (t == seq_len-1 and self.symbol_set[r] == '-'):
                        new_path += self.symbol_set[r]  # If the current symbol differs from the last one and is not a blank symbol at the end, append the current symbol

                    # Update probability in temporary paths
                    if new_path in tempBestPaths:
                        tempBestPaths[new_path] += prob * score  # Accumulate path probability
                    else:
                        tempBestPaths[new_path] = prob * score  # Initialize path probability
                    

            # Get top k best paths and reset best paths
            if len(tempBestPaths) >= self.beam_width:
                bestPaths = dict(sorted(tempBestPaths.items(), key=lambda x: x[1], reverse=True)[:self.beam_width])

        # Get the highest scoring path
        bestPath = max(bestPaths, key=bestPaths.get)
        finalPaths = dict()
        for path, score in tempBestPaths.items():
            if path[-1] == '-':
                finalPaths[path[:-1]] = score  # Remove the trailing blank symbol from the path
            else:
                finalPaths[path] = score  # Keep the path as is
        return bestPath, finalPaths  # Return the best path and all final merged paths with their scores
