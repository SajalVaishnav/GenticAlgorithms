import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import ga
import time



np.set_printoptions(threshold=np.inf)


class solution:
    def __init__(self, W1, W2, b1, b2):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2
        self.fitness = 0

def Sigmoid(Z):
    return 1/(1+np.exp(-Z))

def Relu(Z):
    return np.maximum(0,Z)

def dRelu2(dZ, Z):    
    dZ[Z <= 0] = 0    
    return dZ

def dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def dSigmoid(Z):
    s = 1/(1+np.exp(-Z))
    dZ = s * (1-s)
    return dZ

class dlnet:
    def __init__(self, x, y):
        self.debug = 0;
        self.X=x
        self.Y=y
        self.Yh=np.zeros((1,self.Y.shape[1])) 
        self.L=2
        self.dims = [9, 15, 1] 
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.GAloss = []
        self.GDloss = []
        self.lr=0.003
        self.sam = self.Y.shape[1]
        self.threshold=0.5
        
    def nInit(self):    
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))        
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))                
        return 

    def forward(self):    
        Z1 = self.param['W1'].dot(self.X) + self.param['b1'] 
        A1 = Relu(Z1)
        self.ch['Z1'],self.ch['A1']=Z1,A1
        
        Z2 = self.param['W2'].dot(A1) + self.param['b2']  
        A2 = Sigmoid(Z2)
        self.ch['Z2'],self.ch['A2']=Z2,A2

        self.Yh=A2
        loss=self.nloss(A2)
        return self.Yh, loss

    def nloss(self,Yh):
        loss = (1./self.sam) * (-np.dot(self.Y,np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T))    
        return loss

    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))    
        
        dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])    
        dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
        dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
                            
        dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])        
        dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
        dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T) 
        
        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        
        return

    def pred(self,x, y):  
        self.X=x
        self.Y=y
        comp = np.zeros((1,x.shape[1]))
        pred, loss= self.forward()    
    
        for i in range(0, pred.shape[1]):
            if pred[0,i] > self.threshold: comp[0,i] = 1
            else: comp[0,i] = 0
    
        print("Acc: " + str(np.sum((comp == y)/x.shape[1])))
        
        return comp
    
    def gd(self, X, Y, iter = 3000):
        np.random.seed(1)                         
    
        self.nInit()
    
        for i in range(0, iter):
            Yh, loss=self.forward()
            self.backward()
            if(i % 5 == 0):
                self.GDloss.append(loss)
                

        plt.plot(np.squeeze(self.GDloss))
        plt.ylabel('Loss')
        plt.xlabel('Iter')
        plt.title("Lr =" + str(self.lr))
        plt.show()
    
        return
    
    def GA(self, sol_per_pop = 5, num_parents_mating = 2, num_generations = 20000, mutation_percent = 10):
        initial_pop_weights = []
        for _ in np.arange(0, sol_per_pop):
            input_HL1_weights = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
            HL1_output_weights = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])

            initial_pop_weights.append(np.array([input_HL1_weights, 
                                                        HL1_output_weights]))

        pop_weights_mat = np.array(initial_pop_weights)
        pop_weights_vector = ga.mat_to_vector(pop_weights_mat)

        for generation in range(num_generations):
            
            # converting the solutions from being vectors to matrices.
            pop_weights_mat = ga.vector_to_mat(pop_weights_vector, 
                                            pop_weights_mat)
            # Measuring the fitness of each chromosome in the population.
            fitness = self.fitness(pop_weights_mat)
            
            # Selecting the best parents in the population for mating.
            parents = ga.select_mating_pool(pop_weights_vector, 
                                            fitness.copy(), 
                                            num_parents_mating)
            

            # Generating next generation using crossover.
            offspring_crossover = ga.crossover(parents,
                                            offspring_size=(pop_weights_vector.shape[0]-parents.shape[0], pop_weights_vector.shape[1]))
            

            # Adding some variations to the offsrping using mutation.
            offspring_mutation = ga.mutation(offspring_crossover, 
                                            mutation_percent=mutation_percent)

            # Creating the new population based on the parents and offspring.
            pop_weights_vector[0:parents.shape[0], :] = parents
            pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

            #adding the loss of current generation to tracking function
            pop_weights_mat = ga.vector_to_mat(pop_weights_vector, pop_weights_mat)
            best_weight_generation = pop_weights_mat [0, :]
            self.ginit(best_weight_generation[0], best_weight_generation[1])
            Yh, loss=self.forward()
            self.GAloss.append(loss)

        plt.plot(np.squeeze(self.GAloss))
        plt.ylabel('Loss')
        plt.xlabel('Generation')
        plt.show()
    
        return
            
    def fitness(self, weights_mat):
        fitness = np.empty(shape=(weights_mat.shape[0]))
        for sol_idx in range(weights_mat.shape[0]):
            curr_sol_mat = weights_mat[sol_idx, :]                        
            self.ginit(curr_sol_mat[0], curr_sol_mat[1])
            Yh, loss=self.forward()
            if(loss == 0):
                fitness[sol_idx] = 99999
            else:
                fitness[sol_idx] = abs(1/loss)
        return fitness
    
    def ginit(self, W1, W2):
        self.param['W1'] = W1 
        self.param['b1'] = np.zeros((self.dims[1], 1))        
        self.param['W2'] = W2 
        self.param['b2'] = np.zeros((self.dims[2], 1))                
        return 
     
    

df = pd.read_csv('wisconsin-cancer-dataset.csv',header=None)
df = df[~df[6].isin(['?'])]
df = df.astype(float)
df.iloc[:,10].replace(2, 0,inplace=True)
df.iloc[:,10].replace(4, 1,inplace=True)

df.head(3)
scaled_df=df
names = df.columns[0:10]
scaler = MinMaxScaler() 
scaled_df = scaler.fit_transform(df.iloc[:,0:10]) 
scaled_df = pd.DataFrame(scaled_df, columns=names)

x=scaled_df.iloc[0:500,1:10].values.transpose()
y=df.iloc[0:500,10:].values.transpose()

xval=scaled_df.iloc[501:683,1:10].values.transpose()
yval=df.iloc[501:683,10:].values.transpose()

nn = dlnet(x,y)
nn.lr=0.07
nn.dims = [9, 15, 1]

nn.GA()   
nn.gd(x, y, iter = 100000)


loss_difference_per_loop = []
print(len(nn.GDloss))
gd_count = 0
for ga_loss in nn.GAloss:
    print(nn.GDloss[gd_count][0])
    gd_loss_val = nn.GDloss[gd_count][0][0]
    ga_loss_val = ga_loss[0][0]
    loss_difference_per_loop.append(gd_loss_val - ga_loss_val)
    gd_count = gd_count + 1

plt.plot(np.squeeze(loss_difference_per_loop))
plt.ylabel('GAloss - GDloss')
plt.xlabel('loop')
plt.show()
