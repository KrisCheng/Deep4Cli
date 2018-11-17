from matplotlib import pyplot
import pylab as plt
import numpy as np

def drawPillar():   
    n_groups = 5
    means_men = (20, 35, 30, 35, 27)  
    means_women = (25, 32, 34, 20, 25)  
       
    fig, ax = plt.subplots()  
    index = np.arange(n_groups)  
    bar_width = 0.35  
       
    opacity = 1
    rects1 = plt.bar(index, means_men, bar_width,alpha=opacity, color='b',label='RMSE')  
    rects2 = plt.bar(index + bar_width, means_women, bar_width,alpha=opacity,color='g',label='MAE')  
       
    plt.xlabel('Layers')  
    plt.ylabel('Scores')  
    plt.title('Different Layer')  
    plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))  
    plt.ylim(0,40);  
    plt.legend();  
    
    plt.tight_layout(); 
    plt.show()

if __name__ == '__main__':
    drawPillar()