import os

IGNORE = {"venv", ".venv", "__pycache__", ".git", ".idea", ".vscode"}

def print_tree(start_path=".", prefix=""):
    items = sorted(os.listdir(start_path))
    items = [item for item in items if item not in IGNORE]

    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = i == len(items) - 1

        connector = "└── " if is_last else "├── "
        print(prefix + connector + item)

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    print_tree()