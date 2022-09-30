import math
import re
import numpy as np
import sympy as simp

class Polynomial:

    def __init__(self,order,coeff):
        self.order = order
        self.coeff = coeff
    
    @staticmethod
    def from_string(str):
        p = len(str)
        order = 0
        count3 = 0
        count4 = 0
        for i in range(p):
            if str[i] == "^":
                count3  = count3 + 1
                if int(str[i+1]) > int(order):
                    order = str[i+1]
            if str[i] == "x":
                count4 = count4 + 1
        if count3 == 0 and count4 != 0:
            order = 1
        if count3 == 0 and count4 == 0:
            order = 0
        coeff = np.zeros(int(order)+1,dtype=int)
        power = np.zeros(int(order)+1,dtype=int)
        splitty2 = str.split()
        k = 0
        while(1):
            if re.search("^\+$",splitty2[k]):
                splitty2[k] = ""
                del splitty2[k]
            if re.search("^-$",splitty2[k]):
                splitty2[k+1] = "-" + splitty2[k+1]
                del splitty2[k]   
            k = k+1
            if k == len(splitty2):
                break
        splitty = splitty2
        zeroterms = int(order) + 1 - len(splitty)
        for i in range(zeroterms):
            splitty.append("")
        q = len(splitty)
        for i in range(int(order)+1):
            l = len(splitty[i])
            for j in range(l):
                count = 0
                count1 = 0
                if splitty[i][j] == "x":
                    count = count + 1
                if splitty[i][j] == "^":
                    count1 = count1 + 1
                    power[i] = int(splitty[i][j+1])
                if count != 0 and count1 == 0:
                    power[i] = 1
        for i in range(q):
            for j in range(i,q):
                if power[i] < power[j]:
                    splitty[i], splitty[j] = splitty[j], splitty[i]
                    power[i], power[j] = power[j], power[i]
        for i in range(int(order)):
            if power[i] != int(int(order)-i):
                for j in reversed(range(i+1,q)):
                    power[j] = power[j-1]
                    splitty[j] = splitty[j-1]
                power[i] = 0   
                splitty[i] = "" 
            else:
                power[i] = power[i]  
                splitty[i] = splitty[i] 
        for i in range(int(order)):
            power[int(order)-i] = i
        if re.search("\d",splitty[int(order)]):
            coeff[int(order)] = re.findall("\d",splitty[int(order)])[0]
            if len(re.findall("\d",splitty[int(order)])) > 1:
                coeff[int(order)] = coeff[int(order)]*10 + int(re.findall("\d",splitty[int(order)])[1])
        for i in range(int(order)+1):
            for j in range(len(splitty[i])):
                if splitty[i][j] == "*":
                    coeff[i] = int(splitty[i][j-1])
                    if j > 1:
                        if splitty[i][j-2].isdigit():
                            coeff[i] = int(splitty[i][j-2])*10 + int(splitty[i][j-1])
                if splitty[i][j] == "x" and splitty[i][j-1] != "*":
                    coeff[i] = 1    
            if re.search("-",splitty[i]):
                coeff[i] = coeff[i]*(-1)
        return Polynomial(order,coeff)

    def __repr__(self):
        string = ""
        mono = [None]*(int(self.order)+1)
        for i in range(int(self.order)+1):
            if self.coeff[i] < 0:
                mono[i] = str(self.coeff[i]*-1) + "*x^" + str(int(self.order)-i)
            if self.coeff[i] >= 0:
                mono[i] = str(self.coeff[i]) + "*x^" + str(int(self.order)-i)
            if i != int(self.order)+1:
                if self.coeff[i] >= 0:
                    mono[i] = " + " + mono[i] 
                if self.coeff[i] < 0:
                    mono[i] = " - " + mono[i] 
            string = string + mono[i]
        string = string[1:len(string)]            
        return string

    def __add__(self,other):
        if int(self.order) > int(other.order):
            big_order = int(self.order)
            small_order = int(other.order)
            N = np.zeros(abs(int(self.order) - int(other.order)),dtype=int)
            other.coeff = np.append(N,other.coeff)
        if int(self.order) <= int(other.order):
            N = np.zeros(abs(int(other.order) - int(self.order)),dtype=int)
            big_order = int(other.order)
            small_order = int(self.order)
            self.coeff = np.concatenate((N,self.coeff))
        coeff = np.zeros(int(big_order)+1,dtype=int)
        for i in range(0,int(big_order)+1):
            coeff[i] = self.coeff[i] + other.coeff[i]
        order=big_order
        count2 = 0
        for i in range(int(order)+1):
            if coeff[i] == 0:
                count2 = count2 + 1
            if coeff[0] != 0:
                break
        coeff = coeff[count2:]
        order = order - count2
        return Polynomial(order,coeff)

    def __neg__(self):
        coeff = np.zeros(int(self.order)+1,dtype=int)
        for i in range(int(self.order)+1):
            coeff[i] = self.coeff[i]*-1
        order = self.order
        return Polynomial(order,coeff)
    
    def __mul__(self,other):
        order = int(self.order) + int(other.order)
        coeff = np.zeros(int(order)+1,dtype=int)
        power1 = np.zeros(int(self.order)+1,dtype=int)
        power2 = np.zeros(int(other.order)+1,dtype=int)
        for i in range(int(self.order)+1):
            power1[int(self.order)-i] = i
        for i in range(int(other.order)+1):
            power2[int(other.order)-i] = i
        for i in range(int(self.order)+1):
            for j in range(int(other.order)+1):
                coeff[int(order)-(power1[i]+power2[j])] = coeff[int(order)-(power1[i]+power2[j])] + self.coeff[i]*other.coeff[j]  
        return Polynomial(order,coeff)

    def __sub__(self,other):
        N = np.zeros(abs(int(self.order) - int(other.order)),dtype=int)
        if int(self.order) >= int(other.order):
            other.coeff = np.append(N,other.coeff)
            big_order = int(self.order)
        if int(self.order) < int(other.order):
            self.coeff = np.append(N,self.coeff)
            big_order = int(other.order)
        coeff = np.zeros(int(big_order)+1,dtype=int)
        for i in range(int(big_order)+1):
            coeff[i] = self.coeff[i] - other.coeff[i]
        order = big_order
        count2 = 0
        for i in range(int(order)+1):
            if coeff[i] == 0:
                count2 = count2 + 1
            if coeff[i] != 0:
                break
        if count2 != 0:
            coeff = coeff[count2:]
            order = order - count2 
        return Polynomial(order,coeff)
    
    def __eq__(self,other):
        count = 0
        if int(self.order) != int(other.order):
            return False
        if int(self.order) == int(other.order):
            for i in range(int(self.order)):
                if self.coeff[i] != other.coeff[i]: 
                    count = count +1
        if count == 0:
            return True
        if count != 0:
            return False
    
    def __truediv__(self,other): 
        c = RationalPolynomial(str(self),str(other))
        return c
            
class RationalPolynomial:

    def __init__(self,numerator,denominator):
        self.numerator = numerator
        self.denominator = denominator
        self._reduce()
    
    @staticmethod
    def from_string(str):
        splitty2 = str.split("/")
        p = len(splitty2[0])
        if p > 1:
            splitty2[0] = splitty2[0][1:p-1] 
        p = len(splitty2[1])
        if p > 1:
            splitty2[1] = splitty2[1][1:p-1]     
        return RationalPolynomial(splitty2[0],splitty2[1])
    
    def _reduce(self):
        self.numerator = self.numerator.replace("^","**")
        self.denominator = self.denominator.replace("^","**")
        total = "(" + self.numerator + ")" + "/" + "(" + self.denominator + ")"
        total = simp.cancel(total)
        total = str(total)
        total = total.replace("**","^")
        splitty2 = total.split("/")
        p = len(splitty2[0])
        if p > 1:
            splitty2[0] = splitty2[0][1:p-1] 
        p = len(splitty2[1])
        if p > 1:
            splitty2[1] = splitty2[1][1:p-1] 
        self.numerator = splitty2[0]
        self.denominator = splitty2[1]

    def __repr__(self):
        string = "(" + self.numerator + ")" + "/" + "(" + self.denominator + ")"
        return string

    def __add__(self,other):
        a = Polynomial.from_string(str(self.numerator))
        b = Polynomial.from_string(str(self.denominator))
        c = Polynomial.from_string(str(other.numerator))
        d = Polynomial.from_string(str(other.denominator))
        numerator1 = a*d
        numerator2 = b*c
        numerator = str(numerator1 + numerator2)
        denominator = str(b*d)
        c = RationalPolynomial(numerator,denominator)
        return (c)

    def __sub__(self,other):
        a = Polynomial.from_string(str(self.numerator))
        b = Polynomial.from_string(str(self.denominator))
        c = Polynomial.from_string(str(other.numerator))
        d = Polynomial.from_string(str(other.denominator))
        numerator1 = a*d
        numerator2 = b*c
        numerator = str(numerator1 - numerator2)
        denominator = str(b*d)
        c = RationalPolynomial(numerator,denominator)
        return(c) 

    def __mul__(self,other):
        a = Polynomial.from_string(str(self.numerator))
        b = Polynomial.from_string(str(self.denominator))
        c = Polynomial.from_string(str(other.numerator))
        d = Polynomial.from_string(str(other.denominator))
        numerator = str(a*c)
        denominator = str(b*d)
        c = RationalPolynomial(numerator,denominator)
        return(c) 
    
    def __truediv__(self,other):
        a = Polynomial.from_string(str(self.numerator))
        b = Polynomial.from_string(str(self.denominator))
        c = Polynomial.from_string(str(other.numerator))
        d = Polynomial.from_string(str(other.denominator))
        numerator = str(a*d)
        denominator = str(b*c)
        c = RationalPolynomial(numerator,denominator)
        return(c) 

    def __eq__(self,other):
        if self.numerator == other.numerator:
            if self.denominator == other.denominator:
                return True
        return False

