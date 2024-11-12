
# Step 1) Get a sorted list of all unique characters that occur in this text
# Hint: set is useful for getting unique elements in a sequence
chars = sorted(list(set(text)))

# Step 2) Create the dictionaries str_to_int and int_to_str
str_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_str = {i: ch for i, ch in enumerate(chars)}

# Step 3) Define encode and decode functions
def encode(text, str_to_int):
    ids = [str_to_int[c] for c in text]
    return ids

def decode(ids, int_to_str):
    text_list = [int_to_str[id] for id in ids]
    return ''.join(text_list)

# Step 4) Test your implementation on "My dog Leo is extremely cute."
input_text = "My dog Leo is extremely cute."
ids = encode(input_text, str_to_int)
decoded_text = decode(ids, int_to_str)
assert input_text == decoded_text
