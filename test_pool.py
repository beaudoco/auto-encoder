import multiprocessing
import time
   
  
def square(x):
    print(x[0])
    return x[0] * x[0]
   
if __name__ == '__main__':
    pool = multiprocessing.Pool()
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    inputs = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9)]
    outputs = pool.map(square, inputs)
    pool.close()
    print("Input: {}".format(inputs))
    print("Output: {}".format(outputs))