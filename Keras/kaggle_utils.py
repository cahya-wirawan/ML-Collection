
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    column_width = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * column_width
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(column_width) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(column_width) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(column_width) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
