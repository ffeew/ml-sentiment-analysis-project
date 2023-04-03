import numpy as np

class counter:

    def __init__(self):

        self.data = []
        self.e = {}
        self.y_count = {}
        self.k = 1

    def read_file(self, filename):

        raw = []

        with open(filename, "r") as file:
            raw = file.readlines()
        for i in range(len(raw)):
            if raw[i]!="\n":
                self.data.append(raw[i].strip("\n").split())
        #print(self.data)

    def count_e(self):
         
        for pair in self.data:
            if not pair[1] in self.e.keys():
                self.e[pair[1]] = {}
                self.y_count[pair[1]] =0
            if not pair[0] in self.e[pair[1]].keys():
                self.e[pair[1]][pair[0]]=1
                self.y_count[pair[1]] +=1
            else:
                self.e[pair[1]][pair[0]] +=1
                self.y_count[pair[1]] +=1
        
        for y in self.e.keys():
            for x in self.e[y].keys():
                if x=="#UNK#":
                    self.e[y][x] = self.k/(self.y_count[y]+self.k)
                else:
                    self.e[y][x] = self.e[y][x]/(self.y_count[y]+self.k)
        
        print(self.e)


if __name__ == "__main__":
    count = counter()
    count.read_file("FR/train")
    count.count_e()