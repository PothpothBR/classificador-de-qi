from random import  shuffle
from clip_dataset import clipDataset
import numpy
import pandas
import json


# importa o dataset
data, info = None, None
try:
    data = pandas.read_csv("./res/clipped.csv")
except:
    data = pandas.read_csv("./res/ABBREV.csv")
    data = clipDataset(data)
finally:
    with open("./res/info.json") as file:
        info = pandas.DataFrame(json.loads(file.read()).items(), columns=["name", "count"])

class SLP:
    def __init__(self, batch: pandas.DataFrame, mini_batch_size: int, eta: float, bias: float) -> None:
        # shuffle here
        for _ in range(20):
            batch = batch.sample(frac=1).reset_index(drop=True)
        classes = batch.pop("Shrt_Desc").to_list()
        nclass = classes[0]
        classes = list(map(lambda a: 0 if a == nclass else 1, classes))
        self.entries = [batch[i:i+mini_batch_size][batch.columns[1:]].to_numpy() for i in range(int(len(batch)/mini_batch_size))]
        self.outs = [classes[i:i+mini_batch_size] for i in range(int(len(batch)/mini_batch_size))]
        self.eta = eta
        self.bias = bias
        self.nodes = None
        self.errors = numpy.zeros(mini_batch_size)
        self.weights = numpy.random.random([1, len(batch.columns[1:])+1])[0]
        
        print(
"""
Initialized with:
    {:<6} Batchs
    {:<6} Entries P/Batch
    {:<6} Learning Rate
    {:<6} Bias
""".format(len(self.outs), len(self.outs[0]), eta, bias)
        )
        
    @staticmethod
    def binaryStep(val: float) -> float:
        return 0 if val < 0. else 1
        
    def fit(self, epochs: int) -> None:
        for epoch in range(epochs):
            bsequence = list(range(len(self.outs)))
            shuffle(bsequence)
            for batch in bsequence:
                esequence = list(range(len(self.entries[batch])))
                shuffle(esequence)
                for entry in esequence:
                    self.nodes = numpy.hstack((self.bias, self.entries[batch][entry]))
                    out = SLP.binaryStep(numpy.dot(self.weights, self.nodes))
                    self.errors[entry] = self.outs[batch][entry] - out
                    self.weights = self.weights + self.eta*self.errors[entry]*self.nodes
            print("Epoch "+ str(epoch)+ " finalized!", end="\r")

netw = SLP(data, 323, 0.2, 1)

netw.fit(2000)

print(netw.errors)