import numpy as np

def get_neighbours(pixel):
    j = pixel[0][0]
    i = pixel[0][1]
    return [(j+1,i), (j-1,i), (j,i+1), (j,i-1)]


def dfs(img,j,i,width,height):
    img_processed = np.full(img.shape, False, dtype=bool)
    pixel_ls = []
    count = 0
    todo = [(j,i)]
    while todo:
        j,i = todo.pop()
        if not (0 <= j < height) or not (0 <= i < width) or (img[j,i] == 0).all():
            continue
        if img[j,i] == 255:
            pixel_ls.append((j,i))
            img_processed[j,i] = True
            count += 1
            neighbours = get_neighbours([(j,i)])
        for k1,k2 in neighbours:
            check_white = False
            check_unprocessed = False
            if img[k1,k2] == 255 :
                check_white = True
            if img_processed[k1,k2] == False:
                check_unprocessed = True
            if check_white == True and check_unprocessed == True:
                todo += [(k1,k2)]

    return pixel_ls, count

def apply_correction(r,circ):
    r_eq = (r.perimeter/(2*np.pi)) + 0.5
    correction = (1 - (0.5/r_eq))**2
    new_circ = min(circ* correction, 1)
    return new_circ