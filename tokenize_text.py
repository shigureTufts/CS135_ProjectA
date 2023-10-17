def tokenize_text(raw_text):
    ''' Transform a plain-text string into a list of tokens

    We assume that *whitespace* divides tokens.

    Args
    ----
    raw_text : string

    Returns
    -------
    list_of_tokens : list of strings
        Each element is one token in the provided text
    '''
    list_of_tokens = raw_text.split()  # split method divides on whitespace by default
    for pp in range(len(list_of_tokens)):
        cur_token = list_of_tokens[pp]
        # Remove punctuation
        for punc in ['?', '!', '_', '.', ',', '"', '/', '-', '1', '2', '3', '4', '5',
                     '6', '7', '8', '9', '0', '&', '\'', '*', ':']:
            cur_token = cur_token.replace(punc, "")
        # Turn to lower case
        clean_token = cur_token.lower()
        # Replace the cleaned token into the original list
        list_of_tokens[pp] = clean_token
    return list_of_tokens