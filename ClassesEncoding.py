def encode(column):
    y = []
    for i in column:
        if i == 'BOMBAY':
            y.append(0)
        elif i == 'CALI':
            y.append(1)
        elif i == 'SIRA':
            y.append(2)
    return y
