import pandas as pd

n = pd.read_csv(r"C:\Users\Sudhiksha\Documents\sem 5- AIML LAB\AI\nqueens.csv")["N"][0]
board = [-1] * n

def solve(row):
    if row == n:
        return True

    for col in range(n):
        if all(board[i] != col and
               abs(board[i] - col) != abs(i - row)
               for i in range(row)):
            board[row] = col

            if solve(row + 1):
                return True

            board[row] = -1

    return False

if solve(0):
    for col in board:
        print("." * col + "Q" + "." * (n-col-1))
else:
    print("No solution")