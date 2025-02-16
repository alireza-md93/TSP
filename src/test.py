# importing the multiprocessing module
import multiprocessing

n = 0

def print_cube(num):
    """
    function to print cube of given num
    """
    print("Cube: {}".format(num * num * num))
    n=n+1

def print_square(num):
    """
    function to print square of given num
    """
    print("Square: {}".format(num * num))

if __name__ == "__main__":
    # creating processes
    p1 = [multiprocessing.Process(target=print_square, args=(i, )) for i in range(2)]
    p2 = [multiprocessing.Process(target=print_cube, args=(i, )) for i in range(2)]

    for p in p1:
        p.start()
    for p in p2:
        p.start()
    
    for p in p1:
        p.join()
    for p in p2:
        p.join()

    # both processes finished
    print("Done!")
