import re
from collections import defaultdict

class BasicTokenizer:
    def train(self, text, token_list_length):
        byte_flow = BasicTokenizer.text_2_byte_flow(text)

        self.encode_rules = list() # pair, new token

        for merge0 in range(token_list_length):
            pair2count = BasicTokenizer.get_stats(byte_flow)
            most_pair = BasicTokenizer.get_most_count_pair(pair2count)

            new_token = 256 + merge0

            byte_flow = BasicTokenizer.merge(byte_flow, most_pair, new_token)

            self.encode_rules.append((most_pair, new_token))

    def encode(self, text):
        byte_flow = BasicTokenizer.text_2_byte_flow(text)
        for (pair0, pair1), new_token in self.encode_rules:
            i = 0
            while i < len(byte_flow) - 1:
                if byte_flow[i] == pair0 and byte_flow[i + 1] == pair1:
                    byte_flow = byte_flow[:i] + [new_token] + byte_flow[i + 2:]
                else:
                    i += 1
        return byte_flow

    def decode(self, byte_flow):
        for (pair0, pair1), new_token in reversed(self.encode_rules):
            i = 0
            while i < len(byte_flow):
                if byte_flow[i] == new_token:
                    byte_flow = byte_flow[:i] + [pair0, pair1] + byte_flow[i + 1:]
                else:
                    i += 1
        original_str = bytes(byte_flow).decode('utf-8')
        return original_str

    @staticmethod
    def text_2_byte_flow(text):
        words = re.findall(r'\b\w+\b', text)

        byte_flow = list()
        for word in words:
            byte_flow.extend(word.encode('utf-8'))
            byte_flow.append(32)

        return byte_flow

    @staticmethod
    def get_stats(byte_flow, counts=None):
        """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
        """
        counts = {} if counts is None else counts
        for pair in zip(byte_flow, byte_flow[1:]): # iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def get_most_count_pair(pair2count):
        most_pair = None
        max_count = None
        for pair, count in pair2count.items():
            if max_count is None or max_count < count:
                most_pair = pair
                max_count = count
        return most_pair

    @staticmethod
    def merge(byte_flow, pair, idx):
        """
        In the list of integers (byte_flow), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: byte_flow=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        new_byte_flow = []
        i = 0
        while i < len(byte_flow):
            # if not at the very last position AND the pair matches, replace it
            if byte_flow[i] == pair[0] and i < len(byte_flow) - 1 and byte_flow[i+1] == pair[1]:
                new_byte_flow.append(idx)
                i += 2
            else:
                new_byte_flow.append(byte_flow[i])
                i += 1
        return new_byte_flow