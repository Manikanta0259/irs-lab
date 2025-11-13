#program9
def search_pattern(pattern, text):
    m = len(pattern)
    n = len(text)
    for i in range(n - m + 1):
        j = 0
        while j < m and text[i + j] == pattern[j]:
            j += 1
        if j == m:
            print(f"Pattern found at index {i}")

if __name__ == "__main__":
    text1 = input("Enter the Text: ")
    pattern1 = input("Enter the Pattern: ")
    search_pattern(pattern1, text1)
