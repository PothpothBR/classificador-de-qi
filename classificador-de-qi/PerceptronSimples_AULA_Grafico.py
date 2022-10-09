# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:12:08 2021

@author: malga
"""

import numpy as np 
import pandas as pd


#CASO QUEIRAM CARREGAR OS ATRIBUTOS DIRETO DE UM ARQUIVO (importar o pandas nesse caso)
# dados = pd.read_csv('Nome_arq.csv')
# neste caso ter o cuidado nas entradas e saídas.

# Atributos
peso = np.array([113, 122, 107, 98, 115, 120])
pH = np.array([6.8, 4.7, 5.2, 3.6, 2.9, 4.2])


# Define o número de épocas da simulação e o número de atributos
numEpocas = 70000
numAmostras = 6




# bias
bias = 1

# Entrada do Perceptron.
X = np.vstack((peso, pH))   # Matriz de 2 colunas Ou X = np.asarray([peso, pH])
Y = np.array([-1, 1, -1, -1, 1, 1]) # Nossa saída


# Taxa de aprendizado.
eta = 0.1

# Define o vetor de pesos inicializamos com ZERO
W = np.zeros([1,3])           # (1 linha e 3 colunas) = Duas entradas + o bias !

# Array para amazernar os erros.
e = np.zeros(6)


def funcaoAtivacao(valor):
    #A função de ativação Degrau Bipolar
    
    if valor < 0.0:
        return(-1)
    else: 
        return(1)
  
#Parte principal envolvendo o treinamento

for j in range(numEpocas):
    for k in range(numAmostras): # 6 vezes pois é nosso numero de amostras
        
        # Insere o bias no vetor de entrada.
        Xb = np.hstack((bias, X[:,k]))# O Hstack vai empilhar o bias em todas as linhas (:) indice K
        
        # Calcula o vetor campo induzido (multiplicação vetorial)
        #numpy.dot retorna o produto escalar dos vetores de entrada.
        V = np.dot(W, Xb)
        
        # Calcula a saída do perceptron.
        Yr = funcaoAtivacao(V) #recebe o valor do campo induzido
        
        # Calcula o erro: e = (Y - Yr)
        e[k] = Y[k] - Yr # saída que a gente conhece - a saída da rede
        
        # Treinando a rede.
        W = W + eta*e[k]*Xb #peso + a taxa de aprendizado*erro*a nossa entrada ajustada com o bias
     
#print(W)
print("Vetor de errors (e) = " + str(e)) #transformamos o vetor numérico em um strig.


#PARA GERAR GRAFICOS 

import matplotlib.pyplot as plt
 
y = np.array([6.8, 4.7, 5.2, 3.6, 2.9, 4.2])
x = np.array([113, 122, 107, 98, 115, 120])

#x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
#y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("MAÇAS E LARANJAS")
plt.xlabel("Peso")
plt.ylabel("PH")

#plt.plot(x, y)

#Cor na linha
plt.plot(x, y, color = 'r')
plt.scatter(x, y, c=[0, 1, 0, 0 ,1, 1])

plt.grid()

plt.show()

#PARA MAIORES INFORMAÇÕES CONSULTE A DOCUMENTAÇÃO DO MATPLOTLIB
# https://www.w3schools.com/python/matplotlib_line.asp
