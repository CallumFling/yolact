"""
Dynamic.txt file format:
key value
key2 value2

NO whitespace allowed
"""


def read():
    result = {}
    try:
        with open("dynamic.txt", "r") as file:
            for line in file:
                key, value = line.split()
                result[key] = float(value)
    except Exception as e:
        print(e)
        __import__("pdb").set_trace()
    return result
