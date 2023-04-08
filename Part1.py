import numpy as np
import json

class gen_e:

    def __init__(self):

        self.data = []
        self.e = {}
        self.y_count = {}
        self.x = []
        self.k = 1

        # try:
        #     self.e = self.load_e("FR/count_e.json")
        #     self.y_count = self.load_y("FR/count_y.json")
        #     self.x = self.load_x("FR/x_set.json")
        # except:
        #     self.count_e("FR/train")


    def read_file(self, filename):

        #returns each row of the file as a list of strings separated by spaces

        raw = []
        data = []

        with open(filename, "r", encoding="utf8") as file:
            raw = file.readlines()
        for i in range(len(raw)):
            data.append(raw[i].strip("\n").split())
        return data

    def count_e(self, filename):

        self.data = self.read_file(filename)
         
        for pair in self.data:
            if len(pair):
                if not pair[0] in self.x:
                    self.x.append(pair[0])
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
                # if x=="#UNK#":
                #     self.e[y][x] = self.k/(self.y_count[y]+self.k)
                # else:
                self.e[y][x] = self.e[y][x]/(self.y_count[y]+self.k)
        with open("FR/count_e.json","w",encoding="utf8") as dict_file:
            json.dump(self.e,dict_file,indent=4)
        
        with open("FR/count_y.json","w",encoding="utf8") as y_file:
            json.dump(self.y_count,y_file, indent = 4)

        with open("FR/x_set.json","w",encoding="utf8") as x_file:
            for x_ in self.x:
                x_file.write(x_+"\n")
        
        #print(self.e)

    def predict_y(self, dataset,filename=""):

    
        # print(dataset.T.shape)
        # print(dataset[0][0])
        # print(self.e.keys())
        y_p = []
        for k in range(len(dataset)):
            if len(dataset[k]):
                max_p = 0
                max_y = ""
                for y in self.e.keys():
                    if self.get_e(y,dataset[k][0])>max_p:
                        max_p = self.get_e(y,dataset[k][0])
                        max_y = y
                y_p.append(max_y)
            else:
                y_p.append("")
        print(len(y_p))

        if(len(filename)):
            with open(filename, "w",encoding="utf8") as file:
                for i in range(len(y_p)):
                    file.write((str(dataset[i][0]) if len(dataset[i]) else "")+" "+y_p[i]+"\n")

        return y_p
    
    def load_e(self,filename="FR/count_e.json"):
        with open(filename,"r",encoding="utf8") as file:
            self.e = json.loads(file.read())
    
    def load_y(self,filename="FR/count_y.json"):
        with open(filename,"r",encoding="utf8") as file:
            self.y_count = json.loads(file.read())
    
    def load_x(self,filename="FR/x_set.json"):
        with open(filename,"r",encoding="utf8") as file:
            self.x = file.read().split("\n")

    def get_e(self, y, o):

        if o in self.x:
            if o in self.e[y].keys():
                return self.e[y][o]
            else:
                return 0
        else:
            return self.k/(self.y_count[y]+self.k)

if __name__ == "__main__":
    count = gen_e()
    x_p = count.read_file("FR/dev.in")
    #print(x_p)
    count.count_e("FR/train")
    count.predict_y(x_p, "FR/dev.p1.out")