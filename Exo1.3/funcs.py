def arithmetic(nombre):
    if len(nombre) <= 1: 
        return True

    diff = nombre[1] - nombre[0]

    for i in range(1, len(nombre)):
        if nombre[i] - nombre[i - 1] != diff:
            return False

    return True
