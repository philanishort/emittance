#Create Tabbed widget in Python GUI Application  
import tkinter as tk  
from tkinter import ttk 
from tkinter import messagebox 
import math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
#import matplotlib, sys
#matplotlib.use('TkAgg')
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from tkinter.filedialog import askopenfilename

win = tk.Tk()  
win.title("Emittance Calculator")
win.geometry("1020x670")  
#Create Tab Control  
tabControl=ttk.Notebook(win)  
#Tab1  
tab1=ttk.Frame(tabControl)  
tabControl.add(tab1, text='AX-line') 
 
#Tab2  
tab2=ttk.Frame(tabControl)  
tabControl.add(tab2, text='Q-line')  
tabControl.pack(expand=1, fill="both")  

#Tab3 
tab3=ttk.Frame(tabControl)  
tabControl.add(tab3, text='E_AX-line')  
tabControl.pack(expand=1, fill="both")  

#Tab Name Labels  
#ttk.Label(tab1, text="This is Tab 1").grid(column=0,row=0,padx=10,pady=10)

ttk.Label(tab1, text="Energy in keV").grid(column=0,row=0,pady=(20,0))
ttk.Label(tab1, text="Atomic mass").grid(column=0,row=1)
ttk.Label(tab1, text="Charge State").grid(column=0,row=2)
ttk.Label(tab1, text="Q5cur_0").grid(column=0,row=3,pady=15)
ttk.Label(tab1, text="Q6cur_0").grid(column=2,row=3,pady=15)
ttk.Label(tab1, text="Q6cur_1").grid(column=0,row=5)
ttk.Label(tab1, text="Q6cur_2").grid(column=0,row=6)
ttk.Label(tab1, text="Q6cur_3").grid(column=0,row=7)
ttk.Label(tab1, text="Q6cur_4").grid(column=0,row=8)
ttk.Label(tab1, text="Q6cur_5").grid(column=0,row=9,pady=(4,20))
ttk.Label(tab1, text="Q5cur_1").grid(column=0,row=10)
ttk.Label(tab1, text="Q5cur_2").grid(column=0,row=11)
ttk.Label(tab1, text="Q5cur_3").grid(column=0,row=12)
ttk.Label(tab1, text="Q5cur_4").grid(column=0,row=13)
ttk.Label(tab1, text="Q5cur_5").grid(column=0,row=14,pady=(4,20))

E_Ek = ttk.Entry(tab1, width=10)
E_Ek.grid(column=1,row=0, pady=(20,4))
E_m = ttk.Entry(tab1, width=10)
E_m.grid(column=1,row=1,pady=4)

E_Q = ttk.Entry(tab1, width=10)
E_Q.grid(column=1,row=2,pady=4)

Q5_Cur0 = ttk.Entry(tab1, width=10)
Q5_Cur0.grid(column=1,row=3,pady=15)
Q6_Cur0 = ttk.Entry(tab1, width=10)
Q6_Cur0.grid(column=3,row=3,pady=15)

Q6_Cur1 = ttk.Entry(tab1, width=10)
Q6_Cur1.grid(column=1,row=5,pady=4)
Q6_Cur2 = ttk.Entry(tab1, width=10)
Q6_Cur2.grid(column=1,row=6,pady=4)
Q6_Cur3 = ttk.Entry(tab1, width=10)
Q6_Cur3.grid(column=1,row=7,pady=4)
Q6_Cur4 = ttk.Entry(tab1, width=10)
Q6_Cur4.grid(column=1,row=8,pady=4)
Q6_Cur5 = ttk.Entry(tab1, width=10)
Q6_Cur5.grid(column=1,row=9,pady=(4,20))

Q5_Cur1 = ttk.Entry(tab1, width=10)
Q5_Cur1.grid(column=1,row=10,pady=4)
Q5_Cur2 = ttk.Entry(tab1, width=10)
Q5_Cur2.grid(column=1,row=11,pady=4)
Q5_Cur3 = ttk.Entry(tab1, width=10)
Q5_Cur3.grid(column=1,row=12,pady=4)
Q5_Cur4 = ttk.Entry(tab1, width=10)
Q5_Cur4.grid(column=1,row=13,pady=4)
Q5_Cur5 = ttk.Entry(tab1, width=10)
Q5_Cur5.grid(column=1,row=14,pady=(4,20))


ttk.Label(tab1, text="width_x1").grid(column=2,row=5)
ttk.Label(tab1, text="width_x2").grid(column=2,row=6)
ttk.Label(tab1, text="width_x3").grid(column=2,row=7)
ttk.Label(tab1, text="width_x4").grid(column=2,row=8)
ttk.Label(tab1, text="width_x5").grid(column=2,row=9,pady=(4,20))

Entry_x1 = ttk.Entry(tab1, width=10)
Entry_x1.grid(column=3,row=5,pady=4)
Entry_x2 = ttk.Entry(tab1, width=10)
Entry_x2.grid(column=3,row=6,pady=4)
Entry_x3 = ttk.Entry(tab1, width=10)
Entry_x3.grid(column=3,row=7,pady=4)
Entry_x4 = ttk.Entry(tab1, width=10)
Entry_x4.grid(column=3,row=8,pady=4)
Entry_x5 = ttk.Entry(tab1, width=10)
Entry_x5.grid(column=3,row=9,pady=(4,20))


ttk.Label(tab1, text="width_y1").grid(column=2,row=10)
ttk.Label(tab1, text="width_y2").grid(column=2,row=11)
ttk.Label(tab1, text="width_y3").grid(column=2,row=12)
ttk.Label(tab1, text="width_y4").grid(column=2,row=13)
ttk.Label(tab1, text="width_y5").grid(column=2,row=14,pady=(4,20))


Entry_y1 = ttk.Entry(tab1, width=10)
Entry_y1.grid(column=3,row=10,pady=4)
Entry_y2 = ttk.Entry(tab1, width=10)
Entry_y2.grid(column=3,row=11,pady=4)
Entry_y3 = ttk.Entry(tab1, width=10)
Entry_y3.grid(column=3,row=12,pady=4)
Entry_y4 = ttk.Entry(tab1, width=10)
Entry_y4.grid(column=3,row=13,pady=4)
Entry_y5 = ttk.Entry(tab1, width=10)
Entry_y5.grid(column=3,row=14,pady=(4,20))

xx1 = ttk.Label(tab1, text = "x emittance: ")
xx1.grid(row=17, column=0)

yy1 = ttk.Label(tab1, text = "y emittance: ")
yy1.grid(row=17, column=2)

xy1 = ttk.Label(tab1, text = "4d emittance: ")
xy1.grid(row=18, column=2)

L1 = 0.2175
L2 = 0.07
L3 = 0.435
def Constant_Quad():
    Ek = float(E_Ek.get())
    mass = float(E_m.get())
    charge =float(E_Q.get())
    c = 299800000
    E_0 = 938.28*math.pow(10, 6)
    a = 0.056
    E_k = Ek*1000*charge
    B_r = (math.sqrt(E_k*E_k+2*mass*E_k*E_0))/(charge*c)
    cons = a*B_r
    return cons

def getMatrix1 (B,cons,L):
       b = 0.001858376948749*B+0.002381221835086
       l = 0.13
       k= math.sqrt(b/cons)
       
       quad1 = np.matrix([[math.cos(k*l),(1/k)*math.sin(k*l)],
                          [(-k)*math.sin(k*l),math.cos(k*l)]])
        
       drift1 = np.matrix([[1,L],[0,1]])
	    
       
       sigma1 = np.dot(drift1,quad1)
       return sigma1


def getMatrix2 (B,cons,L):
       b = 0.001858376948749*B+0.002381221835086
       l = 0.13
       k= math.sqrt(b/cons)
       
       quad2 = np.matrix([[math.cosh(k*l),(1/k)*math.sinh(k*l)],
                          [(k)*math.sinh(k*l),math.cosh(k*l)]])
        
       drift2 = np.matrix([[1,L],[0,1]])
	    
       
       sigma2 = np.dot(drift2,quad2)
       return sigma2

def loadme1(Q5_Cur0,Q6_Cur0,Q6_Cur1,Q6_Cur2,Q6_Cur3,Q6_Cur4,Q6_Cur5,
Q5_Cur1,Q5_Cur2,Q5_Cur3,Q5_Cur4,Q5_Cur5,Entry_x1, Entry_x2, Entry_x3, Entry_x4, Entry_x5, Entry_y1, Entry_y2, Entry_y3, Entry_y4, Entry_y5):

    try:
	    filename = askopenfilename(filetypes=(("text file",".txt"),("All files","*.*"))) 
	    result = np.genfromtxt(filename)
    
	    
	    Q5_Cur0.insert(10,result[0,1])
	    Q6_Cur0.insert(10,result[0,4])
	    Q6_Cur1.insert(10,result[0,0])
	    Q6_Cur2.insert(10,result[1,0])
	    Q6_Cur3.insert(10,result[2,0])
	    Q6_Cur4.insert(10,result[3,0])
	    Q6_Cur5.insert(10,result[4,0])
	    Q5_Cur1.insert(10,result[0,3])
	    Q5_Cur2.insert(10,result[1,3])
	    Q5_Cur3.insert(10,result[2,3])
	    Q5_Cur4.insert(10,result[3,3])
	    Q5_Cur5.insert(10,result[4,3])
	    
	    Entry_x1.insert(10,result[0,2])
	    Entry_x2.insert(10,result[1,2])
	    Entry_x3.insert(10,result[2,2])
	    Entry_x4.insert(10,result[3,2])
	    Entry_x5.insert(10,result[4,2])
	   
	    Entry_y1.insert(10,result[0,5])
	    Entry_y2.insert(10,result[1,5])
	    Entry_y3.insert(10,result[2,5])
	    Entry_y4.insert(10,result[3,5])
	    Entry_y5.insert(10,result[4,5])

    except IndexError:
        messagebox.showinfo("Error message", "Invalid input")
    except TypeError:
        messagebox.showinfo("Error message", "Invalid input")
    except OSError:
        messagebox.showinfo("Error message", "Invalid input")
   
def Calculate_Quad():
    

    Q5_B0 = float(Q5_Cur0.get())
    Q6_B0 = float(Q6_Cur0.get())
    Q6_B1 = float(Q6_Cur1.get())
    Q6_B2 = float(Q6_Cur2.get())
    Q6_B3 = float(Q6_Cur3.get())
    Q6_B4 = float(Q6_Cur4.get())
    Q6_B5 = float(Q6_Cur5.get())
    Q5_B1 = float(Q5_Cur1.get())
    Q5_B2 = float(Q5_Cur2.get())
    Q5_B3 = float(Q5_Cur3.get())
    Q5_B4 = float(Q5_Cur4.get())
    Q5_B5 = float(Q5_Cur5.get()) 
   
    x1 = float(Entry_x1.get())
    x2 = float(Entry_x2.get())
    x3 = float(Entry_x3.get())
    x4 = float(Entry_x4.get())
    x5 = float(Entry_x5.get())

    y1 = float(Entry_y1.get())
    y2 = float(Entry_y2.get())
    y3 = float(Entry_y3.get())
    y4 = float(Entry_y4.get())
    y5 = float(Entry_y5.get())

    Sx_1 = getMatrix1(Q6_B1, Constant_Quad(),L1)*getMatrix2(Q5_B0, Constant_Quad(),L2)
    Sx_2 = getMatrix1(Q6_B2, Constant_Quad(),L1)*getMatrix2(Q5_B0, Constant_Quad(),L2)
    Sx_3 = getMatrix1(Q6_B3, Constant_Quad(),L1)*getMatrix2(Q5_B0, Constant_Quad(),L2)
    Sx_4 = getMatrix1(Q6_B4, Constant_Quad(),L1)*getMatrix2(Q5_B0, Constant_Quad(),L2)
    Sx_5 = getMatrix1(Q6_B5, Constant_Quad(),L1)*getMatrix2(Q5_B0, Constant_Quad(),L2)      
    Sy_1 = getMatrix2(Q6_B0, Constant_Quad(),L1)*getMatrix1(Q5_B1, Constant_Quad(),L2) 
    Sy_2 = getMatrix2(Q6_B0, Constant_Quad(),L1)*getMatrix1(Q5_B2, Constant_Quad(),L2)
    Sy_3 = getMatrix2(Q6_B0, Constant_Quad(),L1)*getMatrix1(Q5_B3, Constant_Quad(),L2)
    Sy_4 = getMatrix2(Q6_B0, Constant_Quad(),L1)*getMatrix1(Q5_B4, Constant_Quad(),L2)
    Sy_5 = getMatrix2(Q6_B0, Constant_Quad(),L1)*getMatrix1(Q5_B5, Constant_Quad(),L2)
 
    x_val = np.matrix([[x1**2],[x2**2],[x3**2],[x4**2],[x5**2]]) 
    y_val = np.matrix([[y1**2],[y2**2],[y3**2],[y4**2],[y5**2]])     

    Q1_xmatrix = np.matrix( [[math.pow(Sx_1.item(0, 0), 2), 2*Sx_1.item(0, 0)*Sx_1.item(0, 1), math.pow(Sx_1.item(0, 1),2)],
				   [math.pow(Sx_2.item(0, 0), 2) ,2*(Sx_2.item(0, 0))*Sx_2.item(0, 1),math.pow(Sx_2.item(0, 1),2)],
	                           [math.pow(Sx_3.item(0, 0), 2) ,2*(Sx_3.item(0, 0))*Sx_3.item(0, 1),math.pow(Sx_3.item(0, 1),2)],
	                           [math.pow(Sx_4.item(0, 0), 2) ,2*(Sx_4.item(0, 0))*Sx_4.item(0, 1),math.pow(Sx_4.item(0, 1),2)],
	                           [math.pow(Sx_5.item(0, 0), 2) ,2*(Sx_5.item(0, 0))*Sx_5.item(0, 1),math.pow(Sx_5.item(0, 1),2)]])
				   
	
    Q2_ymatrix = np.matrix( [[math.pow(Sy_1.item(0, 0), 2), 2*Sy_1.item(0, 0)*Sy_1.item(0, 1), math.pow(Sy_1.item(0, 1),2)],
				   [math.pow(Sy_2.item(0, 0), 2), 2*Sy_2.item(0, 0)*Sy_2.item(0, 1), math.pow(Sy_2.item(0, 1),2)],
				   [math.pow(Sy_3.item(0, 0), 2), 2*Sy_3.item(0, 0)*Sy_3.item(0, 1), math.pow(Sy_3.item(0, 1),2)],
	                           [math.pow(Sy_4.item(0, 0), 2), 2*Sy_4.item(0, 0)*Sy_4.item(0, 1), math.pow(Sy_4.item(0, 1),2)],
	                           [math.pow(Sy_5.item(0, 0), 2), 2*Sy_5.item(0, 0)*Sy_5.item(0, 1), math.pow(Sy_5.item(0, 1),2)]])



    Q1_Xmatrix = Q1_xmatrix.I
    Q2_Ymatrix = Q2_ymatrix.I
    
    #global R1
    #global R2
    Result1 = np.dot(Q1_Xmatrix,x_val)
    Result2 = np.dot(Q2_Ymatrix,y_val)
    print(Result1)

    return Result1,Result2


def Clear_Quad():

    E_Ek.delete(first=0,last=22)
    E_m.delete(first=0,last=22)
    E_Q.delete(first=0,last=22)
    Q5_Cur0.delete(first=0,last=22)
    Q6_Cur0.delete(first=0,last=22)
    Q6_Cur1.delete(first=0,last=22)
    Q6_Cur2.delete(first=0,last=22)
    Q6_Cur3.delete(first=0,last=22)
    Q6_Cur4.delete(first=0,last=22)
    Q6_Cur5.delete(first=0,last=22)
    Q5_Cur1.delete(first=0,last=22)
    Q5_Cur2.delete(first=0,last=22)
    Q5_Cur3.delete(first=0,last=22)
    Q5_Cur4.delete(first=0,last=22)
    Q5_Cur5.delete(first=0,last=22) 
   
    Entry_x1.delete(first=0,last=22)
    Entry_x2.delete(first=0,last=22)
    Entry_x3.delete(first=0,last=22)
    Entry_x4.delete(first=0,last=22)
    Entry_x5.delete(first=0,last=22)

    Entry_y1.delete(first=0,last=22)
    Entry_y2.delete(first=0,last=22)
    Entry_y3.delete(first=0,last=22)
    Entry_y4.delete(first=0,last=22)
    Entry_y5.delete(first=0,last=22)
    xx1.config(text = "x emittance: ")
    yy1.config(text = "y emittance: ")
    dataPlot1 = FigureCanvasTkAgg(f1, tab1)
    dataPlot1.draw()
    dataPlot1.get_tk_widget().grid(column=8,columnspan=9,row=0,rowspan=10,padx=50)
    dataPlot2 = FigureCanvasTkAgg(f2, tab1)
    dataPlot2.draw()
    dataPlot2.get_tk_widget().grid(column=8,columnspan=9,row=9,rowspan=14,padx=50)


def Result_Quad():
    try:
	    R1_f,R2_f = Calculate_Quad()

	    B1_final  = [[R1_f[0, 0], R1_f[1, 0]],[R1_f[1, 0], R1_f[2, 0]]]
	    B2_final  = [[R2_f[0, 0], R2_f[1, 0]],[R2_f[1, 0], R2_f[2, 0]]]

	    '''B3_final  = np.matrix([[R1_f[0, 0], R1_f[1, 0],0,0],
                                   [R1_f[1, 0], R1_f[2, 0],0,0],
                                   [0,0,R2_f[0, 0], R2_f[1, 0]],
                                   [0,0,R2_f[1, 0], R2_f[2, 0]]])'''

	    #print(R1_f)
	    print("Second Thingy")

	    print(R2_f)

	    pre_ansx = np.linalg.det(B1_final)
	    pre_ansy = np.linalg.det(B2_final)
	    #pre_ans = np.linalg.det(B3_final)
	    
	    answerx = math.sqrt(abs(pre_ansx))
	    answery = math.sqrt(abs(pre_ansy))
	    #ans = math.sqrt(abs(pre_ans))
	    

	    answerx = np.around(answerx,2)
	    answery = np.around(answery,2)
	    #answer = np.around(ans,2)
	 

	    xx1.config(text = "x emittance: %s."%answerx)
	    yy1.config(text = "y emittance: %s."%answery)
	    #xy1.config(text = "4d emittance: %s."%answer)

	    print("sigma11 : %s."%R1_f[0, 0])
	    print("sigma12 : %s."%R1_f[1, 0])
	    print("sigma22 : %s."%R1_f[2, 0])

	    print("sigma33 : %s."%R2_f[0, 0])
	    print("sigma31 : %s."%R2_f[1, 0])
	    print("sigma44 : %s."%R2_f[2, 0])

    except ValueError:
        messagebox.showinfo("Error message", "Invalid input")
    except TypeError:
        messagebox.showinfo("Error message", "Invalid input")
    except IndexError:
        messagebox.showinfo("Error message", "Invalid input")

  
def Plot_Quad():
    try:

	    R11_f,R22_f = Calculate_Quad()

	    B11_final  = [[R11_f[0, 0], R11_f[1, 0]],[R11_f[1, 0], R11_f[2, 0]]]
	    B22_final  = [[R22_f[0, 0], R22_f[1, 0]],[R22_f[1, 0], R22_f[2, 0]]]
	     
	    pre_ansx11 = np.linalg.det(B11_final)
	    pre_ansy11 = np.linalg.det(B22_final)
	    
	    answerx1 = math.sqrt(abs(pre_ansx11))
	    answery1 = math.sqrt(abs(pre_ansy11))
	    
	    beta1 = R11_f[0, 0]/answerx1
	    alpha1 = -R11_f[1, 0]/answerx1
	    gamma1 = (1+alpha1**2)/beta1

	  
	    beta2 = R22_f[0, 0]/answery1
	    alpha2 = -R22_f[1, 0]/answery1
	    gamma2 = (1+alpha2**2)/beta2

	    f1 = plt.Figure(figsize=(5,3), dpi=80,tight_layout=True)  
	    f2 = plt.Figure(figsize=(5,3), dpi=80,tight_layout=True)
	    
	    x1 = arange(-100,100,0.2)
	    x2 = arange(-100,100,0.2)

	    c1 = []
	    c2 = []
	    for i in range(len(x1)):
               k1 = [abs(beta1),2*alpha1*x1[i],gamma1*x1[i]**2-answerx1]
               roots = np.roots(k1)
               c1.append(roots)

	    pos1 = []
	    neg1 = []
	    pos2 = []
	    neg2 = []

	    for i in range(len(c1)):
	         pos1.append(c1[i][0])
	         neg1.append(c1[i][1])
	    for i in range(len(pos1)):
                if np.imag(pos1[i]) == 0:
                   pos1[i] = pos1[i]
                else:
                   pos1[i] = None
	    for i in range(len(neg1)):
	        if np.imag(neg1[i]) == 0:
                   neg1[i] = neg1[i]
	        else:
                   neg1[i] =None

#---------------second------------------------------------------------
	    for i in range(len(x2)):
               k1 = [abs(beta2),2*alpha2*x2[i],gamma2*x2[i]**2-answery1]
               roots = np.roots(k1)
               c2.append(roots)

	    for i in range(len(c2)):
	         pos2.append(c2[i][0])
	         neg2.append(c2[i][1])
	    for i in range(len(pos2)):
                if np.imag(pos2[i]) == 0:
                   pos2[i] = pos2[i]
                else:
                   pos2[i] = None

	    for i in range(len(neg2)):
	        if np.imag(neg2[i]) == 0:
                   neg2[i] = neg2[i]
	        else:
                   neg2[i] =None

	    a1 = f1.add_subplot(111)
	    a1.plot(x1,pos1,'*b')
	    a1.plot(x1,neg1,'*b')
	    a1.set_xlabel("X")
	    a1.set_ylabel("X'")
	    a1.set_title("Tranverse beam ellipses",fontsize=20)
	    
	    a2 = f2.add_subplot(111)  
	    a2.plot(x2,pos2,'*b')
	    a2.plot(x2,neg2,'*b')
	    a2.set_xlabel("Y")
	    a2.set_ylabel("Y'") 
	    
	   
	    dataPlot1 = FigureCanvasTkAgg(f1, tab1)
	    dataPlot1.draw()
	    dataPlot1.get_tk_widget().grid(column=8,columnspan=9,row=0,rowspan=10,padx=50)
	    dataPlot2 = FigureCanvasTkAgg(f2, tab1)
	    dataPlot2.draw()
	    dataPlot2.get_tk_widget().grid(column=8,columnspan=9,row=10,rowspan=14,padx=50)
    except ValueError:
        messagebox.showinfo("Error message", "Invalid input")
    except TypeError:
        messagebox.showinfo("Error message", "Invalid input")
    except IndexError:
        messagebox.showinfo("Error message", "Invalid input")

f1 = plt.Figure(figsize=(5,3), dpi=80,tight_layout=True)  
f2 = plt.Figure(figsize=(5,3), dpi=80,tight_layout=True)
a1 = f1.add_subplot(111)    
a2 = f2.add_subplot(111)


a1.set_xlabel("X")
a1.set_ylabel("X'")

a2.set_xlabel("Y")
a2.set_ylabel("Y'")

a1.set_title("Tranverse beam ellipses",fontsize=20)
      
dataPlot1 = FigureCanvasTkAgg(f1, tab1)
dataPlot1.draw()
dataPlot1.get_tk_widget().grid(column=8,columnspan=9,row=0,rowspan=10,padx=50)

dataPlot2 = FigureCanvasTkAgg(f2, tab1)
dataPlot2.draw()
dataPlot2.get_tk_widget().grid(column=8,columnspan=9,row=10,rowspan=20,padx=50)

button1 = ttk.Button(tab1, text="Calculate",command=Result_Quad).grid(column=1,row=16,padx=4)
button2 = ttk.Button(tab1, text="Plot",command=Plot_Quad).grid(column=2,row=16,padx=4)
button3 = ttk.Button(tab1, text='load',command=lambda:loadme1(Q5_Cur0,Q6_Cur0,Q6_Cur1,Q6_Cur2,Q6_Cur3,Q6_Cur4,Q6_Cur5,
Q5_Cur1,Q5_Cur2,Q5_Cur3,Q5_Cur4,Q5_Cur5,Entry_x1, Entry_x2, Entry_x3, Entry_x4, Entry_x5, Entry_y1, Entry_y2, Entry_y3, Entry_y4, Entry_y5)).grid(column=0,row=16,padx=4)

button4 = ttk.Button(tab1, text="Clear", command=Clear_Quad).grid(column=3,row=16,padx=4)

#--------------------------TAB 2 SOLENOIDS-------------------------------

#ttk.Label(tab2, text="This is Tab 2").grid(column=0,row=0,padx=10,pady=10) 

ttk.Label( tab2,text="Energy in keV").grid(column=0,row=0,pady=(20,0))
ttk.Label( tab2,text="Atomic mass").grid(column=0,row=1)
ttk.Label( tab2,text="Charge State").grid(column=0,row=2)
ttk.Label( tab2,text="L4Qcur_01").grid(column=1,row=3,pady=15)
ttk.Label( tab2,text="L4Qcur_02").grid(column=3,row=3,pady=15)
ttk.Label( tab2,text="L5Qcur_1").grid(column=0,row=5)
ttk.Label( tab2,text="L5Qcur_2").grid(column=0,row=6)
ttk.Label( tab2,text="L5Qcur_3").grid(column=0,row=7)
ttk.Label( tab2,text="L5Qcur_4").grid(column=0,row=8)
ttk.Label( tab2,text="L5Qcur_5").grid(column=0,row=9)
ttk.Label( tab2,text="L5Qcur_6").grid(column=0,row=10)
ttk.Label( tab2,text="L5Qcur_7").grid(column=0,row=11)
ttk.Label( tab2,text="L5Qcur_8").grid(column=0,row=12)
ttk.Label( tab2,text="L5Qcur_9").grid(column=0,row=13)
ttk.Label( tab2,text="L5Qcur_10").grid(column=0,row=14)


e_Ek = ttk.Entry(tab2, width=8)
e_Ek.grid(column=1,row=0,pady=(20,5))
e_m = ttk.Entry(tab2, width=8)
e_m.grid(column=1,row=1,pady=5)
e_Q = ttk.Entry(tab2, width=8)
e_Q.grid(column=1,row=2,pady=5)


L4Q_Cur0 = ttk.Entry(tab2, width=8)
L4Q_Cur0.grid(column=2,row=3,pady=15)
L5Q_Cur0 = ttk.Entry(tab2, width=8)
L5Q_Cur0.grid(column=4,row=3,pady=15)

L5Q_Cur1 = ttk.Entry(tab2, width=8)
L5Q_Cur1.grid(column=1,row=5,pady=4)
L5Q_Cur2 = ttk.Entry(tab2, width=8)
L5Q_Cur2.grid(column=1,row=6,pady=4)
L5Q_Cur3 = ttk.Entry(tab2, width=8)
L5Q_Cur3.grid(column=1,row=7,pady=4)
L5Q_Cur4 = ttk.Entry(tab2, width=8)
L5Q_Cur4.grid(column=1,row=8,pady=4)
L5Q_Cur5 = ttk.Entry(tab2, width=8)
L5Q_Cur5.grid(column=1,row=9,pady=4)
L5Q_Cur6 = ttk.Entry(tab2, width=8)
L5Q_Cur6.grid(column=1,row=10,pady=4)
L5Q_Cur7 = ttk.Entry(tab2, width=8)
L5Q_Cur7.grid(column=1,row=11,pady=4)
L5Q_Cur8 = ttk.Entry(tab2, width=8)
L5Q_Cur8.grid(column=1,row=12,pady=4)
L5Q_Cur9 = ttk.Entry(tab2, width=8)
L5Q_Cur9.grid(column=1,row=13,pady=4)
L5Q_Cur10 = ttk.Entry(tab2, width=8)
L5Q_Cur10.grid(column=1,row=14,pady=4)

 
ttk.Label( tab2, text="width_x1").grid(column=2,row=5)
ttk.Label( tab2, text="width_x2").grid(column=2,row=6)
ttk.Label( tab2, text="width_x3").grid(column=2,row=7)
ttk.Label( tab2, text="width_x4").grid(column=2,row=8)
ttk.Label( tab2, text="width_x5").grid(column=2,row=9)
ttk.Label( tab2, text="width_x6").grid(column=2,row=10)
ttk.Label( tab2, text="width_x7").grid(column=2,row=11)
ttk.Label( tab2, text="width_x8").grid(column=2,row=12)
ttk.Label( tab2, text="width_x9").grid(column=2,row=13)
ttk.Label( tab2, text="width_x10").grid(column=2,row=14)

entry_x1 = ttk.Entry(tab2, width=8)
entry_x1.grid(column=3,row=5,pady=4)
entry_x2 = ttk.Entry(tab2, width=8)
entry_x2.grid(column=3,row=6,pady=4)
entry_x3 = ttk.Entry(tab2, width=8)
entry_x3.grid(column=3,row=7,pady=4)
entry_x4 = ttk.Entry(tab2, width=8)
entry_x4.grid(column=3,row=8,pady=4)
entry_x5 = ttk.Entry(tab2, width=8)
entry_x5.grid(column=3,row=9,pady=4)
entry_x6 = ttk.Entry(tab2, width=8)
entry_x6.grid(column=3,row=10,pady=4)
entry_x7 = ttk.Entry(tab2, width=8)
entry_x7.grid(column=3,row=11,pady=4)
entry_x8 = ttk.Entry(tab2, width=8)
entry_x8.grid(column=3,row=12,pady=4)
entry_x9 = ttk.Entry(tab2, width=8)
entry_x9.grid(column=3,row=13,pady=4)
entry_x10 = ttk.Entry(tab2, width=8)
entry_x10.grid(column=3,row=14,pady=4)


ttk.Label(tab2,text="width_y1").grid(column=4,row=5)
ttk.Label(tab2,text="width_y2").grid(column=4,row=6)
ttk.Label(tab2,text="width_y3").grid(column=4,row=7)
ttk.Label(tab2,text="width_y4").grid(column=4,row=8)
ttk.Label(tab2,text="width_y5").grid(column=4,row=9)
ttk.Label(tab2,text="width_y6").grid(column=4,row=10)
ttk.Label(tab2,text="width_y7").grid(column=4,row=11)
ttk.Label(tab2,text="width_y8").grid(column=4,row=12)
ttk.Label(tab2,text="width_y9").grid(column=4,row=13)
ttk.Label(tab2,text="width_y10").grid(column=4,row=14)


entry_y1 = ttk.Entry(tab2, width=8)
entry_y1.grid(column=5,row=5,pady=4)
entry_y2 = ttk.Entry(tab2, width=8)
entry_y2.grid(column=5,row=6,pady=4)
entry_y3 = ttk.Entry(tab2, width=8)
entry_y3.grid(column=5,row=7,pady=4)
entry_y4 = ttk.Entry(tab2, width=8)
entry_y4.grid(column=5,row=8,pady=4)
entry_y5 = ttk.Entry(tab2, width=8)
entry_y5.grid(column=5,row=9,pady=4)
entry_y6 = ttk.Entry(tab2, width=8)
entry_y6.grid(column=5,row=10,pady=4)
entry_y7 = ttk.Entry(tab2, width=8)
entry_y7.grid(column=5,row=11,pady=4)
entry_y8 = ttk.Entry(tab2, width=8)
entry_y8.grid(column=5,row=12,pady=4)
entry_y9 = ttk.Entry(tab2, width=8)
entry_y9.grid(column=5,row=13,pady=4)
entry_y10 = ttk.Entry(tab2, width=8)
entry_y10.grid(column=5,row=14,pady=4)

xx2 = ttk.Label(tab2, text = "x emittance: ")
xx2.grid(column=0,row=20)

yy2 = ttk.Label(tab2, text = "y emittance: ")
yy2.grid(column=2,row=20)

xy2 = ttk.Label(tab2, text = "4d emittance: ")
xy2.grid(column=0,row=22)

LL1 = 1
LL2 = 0.5


def Constant_Sol():
    Ek = float(e_Ek.get())
    mass = float(e_m.get())
    charge =float(e_Q.get())
    c = 299800000
    E_0 = 938.28*math.pow(10, 6)
    a = 0.056
    E_k = Ek*1000*charge
    B_r = (math.sqrt(E_k*E_k+2*mass*E_k*E_0))/(charge*c)
    cons = 2*B_r
    return cons

def getMatrix (B1,B2,cons,L1,L2):
       b1 = B1 #0.0035*B1+0.0004
       b2 = B2 #0.0035*B2+0.0004
       l = 0.45
       k1= b1/cons 
       k2= b2/cons 
       C1 = math.cos(k1*l)
       S1 = math.sin(k1*l)
       C2 = math.cos(k2*l) 
       S2 = math.sin(k2*l)
       
       sol1 =np.matrix([[math.pow(C1, 2),(1/k1)*S1*C1,S1*C1,(1/k1)*math.pow(S1, 2),0,0],
                          [-k1*S1*C1,math.pow(C1, 2),-k1*math.pow(S1, 2),S1*C1,0,0],
                          [-S1*C1,-(1/k1)*math.pow(S1,2),math.pow(C1, 2),(1/k1)*S1*C1,0,0],
                          [k1*math.pow(S1, 2),-S1*C1,-k1*S1*C1,math.pow(C1,2),0,0],
                          [0,0,0,0,1,0],[0,0,0,0,0,1]])
        
       sol2 =np.matrix([[math.pow(C2, 2),(1/k2)*S2*C2,S2*C2,(1/k2)*math.pow(S2, 2),0,0],
                          [-k2*S2*C2,math.pow(C2, 2),-k2*math.pow(S2, 2),S2*C2,0,0],
                          [-S2*C2,-(1/k2)*math.pow(S2,2),math.pow(C2, 2),(1/k2)*S2*C2,0,0],
       
                   [k2*math.pow(S2, 2),-S2*C2,-k2*S2*C2,math.pow(C2,2),0,0],
                          [0,0,0,0,1,0],[0,0,0,0,0,1]])
       
     
       drift1  = np.matrix([[1,L1,0,0,0,0],[0,1,0,0,0,0],[0,0,1,L1,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
       drift2  = np.matrix([[1,L2,0,0,0,0],[0,1,0,0,0,0],[0,0,1,L2,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        
       
       
       sigma_0 = np.dot(drift2,sol2)
       sigma_1 = np.dot(drift1,sol1)  
       #print(sigma_0)
       return sigma_0*sigma_1

def loadme2(L4Q_Cur0,L5Q_Cur0,L5Q_Cur1,L5Q_Cur2,L5Q_Cur3,L5Q_Cur4,L5Q_Cur5,
L5Q_Cur6,L5Q_Cur7,L5Q_Cur8,L5Q_Cur9,L5Q_Cur10,entry_x1, entry_x2, entry_x3, entry_x4, entry_x5, entry_x6, entry_x7, entry_x8
,entry_x9, entry_x10, entry_y1, entry_y2, entry_y3, entry_y4, entry_y5, entry_y6, entry_y7, entry_y8, entry_y9, entry_y10):

    try:
	    filename = askopenfilename(filetypes=(("text file",".txt"),("All files","*.*")))     
	    result = np.genfromtxt(filename)
	    
	    L4Q_Cur0.insert(10,result[0,0])
	    L5Q_Cur0.insert(10,result[6,0])
	    L5Q_Cur1.insert(10,result[0,1])
	    L5Q_Cur2.insert(10,result[1,1])
	    L5Q_Cur3.insert(10,result[2,1])
	    L5Q_Cur4.insert(10,result[3,1])
	    L5Q_Cur5.insert(10,result[4,1])
	    L5Q_Cur6.insert(10,result[5,1])
	    L5Q_Cur7.insert(10,result[6,1])
	    L5Q_Cur8.insert(10,result[7,1])
	    L5Q_Cur9.insert(10,result[8,1])
	    L5Q_Cur10.insert(10,result[9,1])




	    entry_y1.insert(10,result[0,3])
	    entry_y2.insert(10,result[1,3])
	    entry_y3.insert(10,result[2,3])
	    entry_y4.insert(10,result[3,3])
	    entry_y5.insert(10,result[4,3])
	    entry_y6.insert(10,result[5,3])
	    entry_y7.insert(10,result[6,3])
	    entry_y8.insert(10,result[7,3])
	    entry_y9.insert(10,result[8,3])
	    entry_y10.insert(10,result[9,3])
	    entry_x1.insert(10,result[0,2])
	    entry_x2.insert(10,result[1,2])
	    entry_x3.insert(10,result[2,2])
	    entry_x4.insert(10,result[3,2])
	    entry_x5.insert(10,result[4,2])
	    entry_x6.insert(10,result[5,2])
	    entry_x7.insert(10,result[6,2])
	    entry_x8.insert(10,result[7,2])
	    entry_x9.insert(10,result[8,2])
	    entry_x10.insert(10,result[9,2])

    except IndexError:
        messagebox.showinfo("Error message", "Invalid input")
    except TypeError:
        messagebox.showinfo("Error message", "Invalid input")
    except OSError:
        messagebox.showinfo("Error message", "Invalid input")

    
def Calculate_Sol():
   
    L4Q_B0 = float(L4Q_Cur0.get())
    L5Q_B0 = float(L5Q_Cur0.get())
    L5Q_B1 = float(L5Q_Cur1.get())
    L5Q_B2 = float(L5Q_Cur2.get())
    L5Q_B3 = float(L5Q_Cur3.get())
    L5Q_B4 = float(L5Q_Cur4.get())
    L5Q_B5 = float(L5Q_Cur5.get())
    L5Q_B6 = float(L5Q_Cur6.get())
    L5Q_B7 = float(L5Q_Cur7.get())
    L5Q_B8 = float(L5Q_Cur8.get())
    L5Q_B9 = float(L5Q_Cur9.get())
    L5Q_B10 = float(L5Q_Cur10.get()) 
   
    x1 = float(entry_x1.get())
    x2 = float(entry_x2.get())
    x3 = float(entry_x3.get())
    x4 = float(entry_x4.get())
    x5 = float(entry_x5.get())
    x6 = float(entry_x6.get())
    x7 = float(entry_x7.get())
    x8 = float(entry_x8.get())
    x9 = float(entry_x9.get())
    x10 = float(entry_x10.get())

    y1 = float(entry_y1.get())
    y2 = float(entry_y2.get())
    y3 = float(entry_y3.get())
    y4 = float(entry_y4.get())
    y5 = float(entry_y5.get())
    y6 = float(entry_y6.get())
    y7 = float(entry_y7.get())
    y8 = float(entry_y8.get())
    y9 = float(entry_y9.get())
    y10 = float(entry_y10.get())



    Sx_1 = getMatrix(L4Q_B0,L5Q_B1, Constant_Sol(),LL1,LL2) 
    Sx_2 = getMatrix(L4Q_B0,L5Q_B2, Constant_Sol(),LL1,LL2) 
    Sx_3 = getMatrix(L4Q_B0,L5Q_B3, Constant_Sol(),LL1,LL2) 
    Sx_4 = getMatrix(L4Q_B0,L5Q_B4, Constant_Sol(),LL1,LL2) 
    Sx_5 = getMatrix(L4Q_B0,L5Q_B5, Constant_Sol(),LL1,L2) 
    Sx_6 = getMatrix(L5Q_B0,L5Q_B6, Constant_Sol(),LL1,LL2)    
    Sx_7 = getMatrix(L5Q_B0,L5Q_B7, Constant_Sol(),LL1,LL2) 
    Sx_8 = getMatrix(L5Q_B0,L5Q_B8, Constant_Sol(),LL1,LL2) 
    Sx_9 = getMatrix(L5Q_B0,L5Q_B9, Constant_Sol(),LL1,LL2) 
    Sx_10 = getMatrix(L5Q_B0,L5Q_B10, Constant_Sol(),LL1,LL2) 
    #print(Sx_10)
	
    Sy_1 = getMatrix(L4Q_B0,L5Q_B1, Constant_Sol(),LL1,LL2) 
    Sy_2 = getMatrix(L4Q_B0,L5Q_B2, Constant_Sol(),LL1,LL2) 
    Sy_3 = getMatrix(L4Q_B0,L5Q_B3, Constant_Sol(),LL1,LL2) 
    Sy_4 = getMatrix(L4Q_B0,L5Q_B4, Constant_Sol(),LL1,LL2) 
    Sy_5 = getMatrix(L4Q_B0,L5Q_B5, Constant_Sol(),LL1,LL2) 
    Sy_6 = getMatrix(L5Q_B0,L5Q_B6, Constant_Sol(),LL1,LL2)    
    Sy_7 = getMatrix(L5Q_B0,L5Q_B7, Constant_Sol(),LL1,LL2) 
    Sy_8 = getMatrix(L5Q_B0,L5Q_B8, Constant_Sol(),LL1,LL2) 
    Sy_9 = getMatrix(L5Q_B0,L5Q_B9, Constant_Sol(),LL1,LL2) 
    Sy_10 = getMatrix(L5Q_B0,L5Q_B10, Constant_Sol(),LL1,LL2) 
    SS = getMatrix(0.08,0.1091, Constant_Sol(),1.,0.5) 
    print(SS)
 
    xy_val = np.matrix([[x1**2],[x2**2],[x3**2],[x4**2],[x5**2],[x6**2],[x7**2],[x8**2],[x9**2],[x10**2], 
	                [y1**2],[y2**2],[y3**2],[y4**2],[y5**2],[y6**2],[y7**2],[y8**2],[y9**2],[y10**2]])

    big_matrix =np.matrix([[math.pow(Sx_1.item(0, 0), 2), 2 * Sx_1.item(0, 0) * Sx_1.item(0, 1), 2 * Sx_1.item(0, 0) * Sx_1.item(0, 2),
	    2*Sx_1.item(0, 0)*Sx_1.item(0, 3),math.pow(Sx_1.item(0, 1), 2), 2 * Sx_1.item(0, 1)*Sx_1.item(0, 2),2*Sx_1.item(0, 1)* Sx_1.item(0, 3),  math.pow(Sx_1.item(0, 2), 2), 2 * Sx_1.item(0, 2)*Sx_1.item(0, 3),math.pow(Sx_1.item(0, 3), 2)],       
	[math.pow(Sx_2.item(0, 0), 2), 2 * Sx_2.item(0, 0) * Sx_2.item(0, 1), 2 * Sx_2.item(0, 0) * Sx_2.item(0, 2),
	    2 * Sx_2.item(0, 0) * Sx_2.item(0, 3), math.pow(Sx_2.item(0, 1), 2), 2 * Sx_2.item(0, 1) * Sx_2.item(0, 2), 2 * Sx_2.item(0, 1) * Sx_2.item(0, 3),
	    math.pow(Sx_2.item(0, 2), 2), 2 * Sx_2.item(0, 2) * Sx_2.item(0, 3), math.pow(Sx_2.item(0, 3), 2)],
	[math.pow(Sx_3.item(0, 0), 2), 2 * Sx_3.item(0, 0) * Sx_3.item(0, 1), 2 * Sx_3.item(0, 0) * Sx_3.item(0, 2),
	    2 * Sx_3.item(0, 0) * Sx_3.item(0, 3), math.pow(Sx_3.item(0, 1), 2), 2 * Sx_3.item(0, 1) * Sx_3.item(0, 2), 2 * Sx_3.item(0, 1) * Sx_3.item(0, 3),
	    math.pow(Sx_3.item(0, 2), 2), 2 * Sx_3.item(0, 2) * Sx_3.item(0, 3), math.pow(Sx_3.item(0, 3), 2)],
	[math.pow(Sx_4.item(0, 0), 2), 2 * Sx_4.item(0, 0) * Sx_4.item(0, 1), 2 * Sx_4.item(0, 0) * Sx_4.item(0, 2),
	    2 * Sx_4.item(0, 0) * Sx_4.item(0, 3), math.pow(Sx_4.item(0, 1), 2), 2 * Sx_4.item(0, 1) * Sx_4.item(0, 2), 2 * Sx_4.item(0, 1) * Sx_4.item(0, 3),
	    math.pow(Sx_4.item(0, 2), 2), 2 * Sx_4.item(0, 2) * Sx_4.item(0, 3), math.pow(Sx_4.item(0, 3), 2)],
	[math.pow(Sx_5.item(0, 0), 2), 2 * Sx_5.item(0, 0) * Sx_5.item(0, 1), 2 * Sx_5.item(0, 0) * Sx_5.item(0, 2),
	    2 * Sx_5.item(0, 0) * Sx_5.item(0, 3), math.pow(Sx_5.item(0, 1), 2), 2 * Sx_5.item(0, 1) * Sx_5.item(0, 2), 2 * Sx_5.item(0, 1) * Sx_5.item(0, 3),
	    math.pow(Sx_5.item(0, 2), 2), 2 * Sx_5.item(0, 2) * Sx_5.item(0, 3), math.pow(Sx_5.item(0, 3), 2)],
	[math.pow(Sx_6.item(0, 0), 2), 2 * Sx_6.item(0, 0) * Sx_6.item(0, 1), 2 * Sx_6.item(0, 0) * Sx_6.item(0, 2),
	    2 * Sx_6.item(0, 0) * Sx_6.item(0, 3), math.pow(Sx_6.item(0, 1), 2), 2 * Sx_6.item(0, 1) * Sx_6.item(0, 2), 2 * Sx_6.item(0, 1) * Sx_6.item(0, 3),
	    math.pow(Sx_6.item(0, 2), 2), 2 * Sx_6.item(0, 2) * Sx_6.item(0, 3), math.pow(Sx_6.item(0, 3), 2)],
	[math.pow(Sx_7.item(0, 0), 2), 2 * Sx_7.item(0, 0) * Sx_7.item(0, 1), 2 * Sx_7.item(0, 0) * Sx_7.item(0, 2),
	    2 * Sx_7.item(0, 0) * Sx_7.item(0, 3), math.pow(Sx_7.item(0, 1), 2), 2 * Sx_7.item(0, 1) * Sx_7.item(0, 2), 2 * Sx_7.item(0, 1) * Sx_7.item(0, 3),
	    math.pow(Sx_7.item(0, 2), 2), 2 * Sx_7.item(0, 2) * Sx_7.item(0, 3), math.pow(Sx_7.item(0, 3), 2)],
	[math.pow(Sx_8.item(0, 0), 2), 2 * Sx_8.item(0, 0) * Sx_8.item(0, 1), 2 * Sx_8.item(0, 0) * Sx_8.item(0, 2),
	    2 * Sx_8.item(0, 0) * Sx_8.item(0, 3), math.pow(Sx_8.item(0, 1), 2), 2 * Sx_8.item(0, 1) * Sx_8.item(0, 2), 2 * Sx_8.item(0, 1) * Sx_8.item(0, 3),
	    math.pow(Sx_8.item(0, 2), 2), 2 * Sx_8.item(0, 2) * Sx_8.item(0, 3), math.pow(Sx_8.item(0, 3), 2)],
	[math.pow(Sx_9.item(0, 0), 2), 2 * Sx_9.item(0, 0) * Sx_9.item(0, 1), 2 * Sx_9.item(0, 0) * Sx_9.item(0, 2),
	    2 * Sx_9.item(0, 0) * Sx_9.item(0, 3), math.pow(Sx_9.item(0, 1), 2), 2 * Sx_9.item(0, 1) * Sx_9.item(0, 2), 2 * Sx_9.item(0, 1) * Sx_9.item(0, 3),
	    math.pow(Sx_9.item(0, 2), 2), 2 * Sx_9.item(0, 2) * Sx_9.item(0, 3), math.pow(Sx_9.item(0, 3), 2)],
	[math.pow(Sx_10.item(0, 0), 2), 2 * Sx_10.item(0, 0) * Sx_10.item(0, 1), 2 * Sx_10.item(0, 0) * Sx_10.item(0, 2),
	    2 * Sx_10.item(0, 0) * Sx_10.item(0, 3), math.pow(Sx_10.item(0, 1), 2), 2 * Sx_10.item(0, 1) * Sx_10.item(0, 2), 2 * Sx_10.item(0, 1) * Sx_10.item(0, 3),
	    math.pow(Sx_10.item(0, 2), 2), 2 * Sx_10.item(0, 2) * Sx_10.item(0, 3), math.pow(Sx_10.item(0, 3), 2)],
	[math.pow(Sy_1.item(2, 0), 2), 2 * Sy_1.item(2, 0) * Sy_1.item(2, 1), 2 * Sy_1.item(2, 0) * Sy_1.item(2, 2),
	    2 * Sy_1.item(2, 0) * Sy_1.item(2, 3), math.pow(Sy_1.item(2, 1), 2), 2 * Sy_1.item(2, 1) * Sy_1.item(2, 2), 2 * Sy_1.item(2, 1) * Sy_1.item(2, 3),
	    math.pow(Sy_1.item(2, 2), 2), 2 * Sy_1.item(2, 2) * Sy_1.item(2, 3), math.pow(Sy_1.item(2, 3), 2)],
	[math.pow(Sy_2.item(2, 0), 2), 2 * Sy_2.item(2, 0) * Sy_2.item(2, 1), 2 * Sy_2.item(2, 0) * Sy_2.item(2, 2),
	    2 * Sy_2.item(2, 0) * Sy_2.item(2, 3), math.pow(Sy_2.item(2, 1), 2), 2 * Sy_2.item(2, 1) * Sy_2.item(2, 2), 2 * Sy_2.item(2, 1) * Sy_2.item(2, 3),
	    math.pow(Sy_2.item(2, 2), 2), 2 * Sy_2.item(2, 2) * Sy_2.item(2, 3), math.pow(Sy_2.item(2, 3), 2)],
	[math.pow(Sy_3.item(2, 0), 2), 2 * Sy_3.item(2, 0) * Sy_3.item(2, 1), 2 * Sy_3.item(2, 0) * Sy_3.item(2, 2),
	    2 * Sy_3.item(2, 0) * Sy_3.item(2, 3), math.pow(Sy_3.item(2, 1), 2), 2 * Sy_3.item(2, 1) * Sy_3.item(2, 2), 2 * Sy_3.item(2, 1) * Sy_3.item(2, 3),
	    math.pow(Sy_3.item(2, 2), 2), 2 * Sy_3.item(2, 2) * Sy_3.item(2, 3), math.pow(Sy_3.item(2, 3), 2)],
	[math.pow(Sy_4.item(2, 0), 2), 2 * Sy_4.item(2, 0) * Sy_4.item(2, 1), 2 * Sy_4.item(2, 0) * Sy_4.item(2, 2),
	    2 * Sy_4.item(2, 0) * Sy_4.item(2, 3), math.pow(Sy_4.item(2, 1), 2), 2 * Sy_4.item(2, 1) * Sy_4.item(2, 2), 2 * Sy_4.item(2, 1) * Sy_4.item(2, 3),
	    math.pow(Sy_4.item(2, 2), 2), 2 * Sy_4.item(2, 2) * Sy_4.item(2, 3), math.pow(Sy_4.item(2, 3), 2)],
	[math.pow(Sy_5.item(2, 0), 2), 2 * Sy_5.item(2, 0) * Sy_5.item(2, 1), 2 * Sy_5.item(2, 0) * Sy_5.item(2, 2),
	    2 * Sy_5.item(2, 0) * Sy_5.item(2, 3), math.pow(Sy_5.item(2, 1), 2), 2 * Sy_5.item(2, 1) * Sy_5.item(2, 2), 2 * Sy_5.item(2, 1) * Sy_5.item(2, 3),
	    math.pow(Sy_5.item(2, 2), 2), 2 * Sy_5.item(2, 2) * Sy_5.item(2, 3), math.pow(Sy_5.item(2, 3), 2)],
	[math.pow(Sy_6.item(2, 0), 2), 2 * Sy_6.item(2, 0) * Sy_6.item(2, 1), 2 * Sy_6.item(2, 0) * Sy_6.item(2, 2),
	    2 * Sy_6.item(2, 0) * Sy_6.item(2, 3), math.pow(Sy_6.item(2, 1), 2), 2 * Sy_6.item(2, 1) * Sy_6.item(2, 2), 2 * Sy_6.item(2, 1) * Sy_6.item(2, 3),
	    math.pow(Sy_6.item(2, 2), 2), 2 * Sy_6.item(2, 2) * Sy_6.item(2, 3), math.pow(Sy_6.item(2, 3), 2)],
	[math.pow(Sy_7.item(2, 0), 2), 2 * Sy_7.item(2, 0) * Sy_7.item(2, 1), 2 * Sy_7.item(2, 0) * Sy_7.item(2, 2),
	    2 * Sy_7.item(2, 0) * Sy_7.item(2, 3), math.pow(Sy_7.item(2, 1), 2), 2 * Sy_7.item(2, 1) * Sy_7.item(2, 2), 2 * Sy_7.item(2, 1) * Sy_7.item(2, 3),
	    math.pow(Sy_7.item(2, 2), 2), 2 * Sy_7.item(2, 2) * Sy_7.item(2, 3), math.pow(Sy_7.item(2, 3), 2)],
	[math.pow(Sy_8.item(2, 0), 2), 2 * Sy_8.item(2, 0) * Sy_8.item(2, 1), 2 * Sy_8.item(2, 0) * Sy_8.item(2, 2),
	    2 * Sy_8.item(2, 0) * Sy_8.item(2, 3), math.pow(Sy_8.item(2, 1), 2), 2 * Sy_8.item(2, 1) * Sy_8.item(2, 2), 2 * Sy_8.item(2, 1) * Sy_8.item(2, 3),
	    math.pow(Sy_8.item(2, 2), 2), 2 * Sy_8.item(2, 2) * Sy_8.item(2, 3), math.pow(Sy_8.item(2, 3), 2)],
	[math.pow(Sy_9.item(2, 0), 2), 2 * Sy_9.item(2, 0) * Sy_9.item(2, 1), 2 * Sy_9.item(2, 0) * Sy_9.item(2, 2),
	    2 * Sy_9.item(2, 0) * Sy_9.item(2, 3), math.pow(Sy_9.item(2, 1), 2), 2 * Sy_9.item(2, 1) * Sy_9.item(2, 2), 2 * Sy_9.item(2, 1) * Sy_9.item(2, 3),
	    math.pow(Sy_9.item(2, 2), 2), 2 * Sy_9.item(2, 2) * Sy_9.item(2, 3), math.pow(Sy_9.item(2, 3), 2)],
	[math.pow(Sy_10.item(2, 0), 2), 2 * Sy_10.item(2, 0) * Sy_10.item(2, 1), 2 * Sy_10.item(2, 0) * Sy_10.item(2, 2),
	    2 * Sy_10.item(2, 0) * Sy_10.item(2, 3), math.pow(Sy_10.item(2, 1), 2), 2 * Sy_10.item(2, 1) * Sy_10.item(2, 2), 2 * Sy_10.item(2, 1) * Sy_10.item(2, 3),
	    math.pow(Sy_10.item(2, 2), 2), 2 * Sy_10.item(2, 2) * Sy_10.item(2, 3), math.pow(Sy_10.item(2, 3), 2)]])

       
    Big_matrix = big_matrix.I
    R1 = np.dot(Big_matrix ,xy_val)
    return R1


def Clear_Sol():
    
    e_Ek.delete(first=0,last=22)
    e_m.delete(first=0,last=22)
    e_Q.delete(first=0,last=22)
    L4Q_Cur0.delete(first=0,last=22)
    L5Q_Cur0.delete(first=0,last=22)
    L5Q_Cur1.delete(first=0,last=22)
    L5Q_Cur2.delete(first=0,last=22)
    L5Q_Cur3.delete(first=0,last=22)
    L5Q_Cur4.delete(first=0,last=22)
    L5Q_Cur5.delete(first=0,last=22)
    L5Q_Cur6.delete(first=0,last=22)
    L5Q_Cur7.delete(first=0,last=22)
    L5Q_Cur8.delete(first=0,last=22)
    L5Q_Cur9.delete(first=0,last=22)
    L5Q_Cur10.delete(first=0,last=22) 

    entry_x1.delete(first=0,last=22)
    entry_x2.delete(first=0,last=22)
    entry_x3.delete(first=0,last=22)
    entry_x4.delete(first=0,last=22)
    entry_x5.delete(first=0,last=22)
    entry_x6.delete(first=0,last=22)
    entry_x7.delete(first=0,last=22)
    entry_x8.delete(first=0,last=22)
    entry_x9.delete(first=0,last=22)
    entry_x10.delete(first=0,last=22)

    entry_y1.delete(first=0,last=22)
    entry_y2.delete(first=0,last=22)
    entry_y3.delete(first=0,last=22)
    entry_y4.delete(first=0,last=22)
    entry_y5.delete(first=0,last=22)
    entry_y6.delete(first=0,last=22)
    entry_y7.delete(first=0,last=22)
    entry_y8.delete(first=0,last=22)
    entry_y9.delete(first=0,last=22)
    entry_y10.delete(first=0,last=22)
    
    dataPlot11 = FigureCanvasTkAgg(f11, tab2)
    dataPlot11.draw()
    dataPlot11.get_tk_widget().grid(column=8,columnspan=9,row=0,rowspan=10,padx=50)

    dataPlot22 = FigureCanvasTkAgg(f22, tab2)
    dataPlot22.draw()
    dataPlot22.get_tk_widget().grid(column=8,columnspan=9,row=10,rowspan=11,padx=50)
    
    xx2.config(text = "x emittance: ")
    yy2.config(text = "y emittance: ")

def Result_Sol():
    try:
	    R1_f =  Calculate_Sol()

	    B1_final  = [[R1_f.item((0, 0)), R1_f.item((1, 0))],[R1_f.item((1, 0)), R1_f.item((4, 0))]]
	    B2_final  = [[R1_f.item((7, 0)), R1_f.item((8, 0))],[R1_f.item((8, 0)), R1_f.item((9, 0))]]
	    #print(R1_f)
	    B3_final  = np.matrix([[R1_f.item((0,0)),R1_f.item((1,0)),R1_f.item((2,0)),R1_f.item((3,0))],
                                   [R1_f.item((1,0)),R1_f.item((4,0)),R1_f.item((5,0)),R1_f.item((6,0))], 
                                   [R1_f.item((2,0)),R1_f.item((5,0)),R1_f.item((7,0)),R1_f.item((8,0))],
                                   [R1_f.item((3,0)),R1_f.item((6,0)),R1_f.item((8,0)),R1_f.item((9,0))]])

	    print(B3_final)

	     
	    pre_ansx = np.linalg.det(B1_final)
	    pre_ansy = np.linalg.det(B2_final)
	    pre_ans = np.linalg.det(B3_final)

	    answerx = math.sqrt(abs(pre_ansx))
	    answery = math.sqrt(abs(pre_ansy))
	    answer = math.sqrt(abs(pre_ans))


	    '''print("sigma_11: %s."%R1_f.item((0, 0)))
	    print("sigma_12: %s."%R1_f.item((1, 0)))
	    print("sigma_22: %s."%R1_f.item((4, 0)))

	    print("sigma_33: %s."%R1_f.item((7, 0)))
	    print("sigma_13: %s."%R1_f.item((8, 0)))
	    print("sigma_44: %s."%R1_f.item((9, 0)))'''


	    xx2.config(text = "x emittance: %s."%np.round(answerx,2))
	    yy2.config(text = "y emittance: %s."%np.round(answery,2))
	    xy2.config(text = "4d emittance: %s."%np.round(answer,2))          

    except ValueError:
        messagebox.showinfo("Error message", "Invalid input")
    except TypeError:
        messagebox.showinfo("Error message", "Invalid input")
    except AttributeError:
        messagebox.showinfo("Error message", "Invalid input")
    except IndexError:
        messagebox.showinfo("Error message", "Invalid input")


def Plot_Sol():
    try:
	    R11_f =  Calculate_Sol()

	    B11_final  = [[R11_f.item((0, 0)), R11_f.item((1, 0))],[R11_f.item((1, 0)), R11_f.item((4, 0))]]
	    B22_final  = [[R11_f.item((7, 0)), R11_f.item((8, 0))],[R11_f.item((8, 0)), R11_f.item((9, 0))]]
	     
	    pre_ansx1 = np.linalg.det(B11_final)
	    pre_ansy1 = np.linalg.det(B22_final)
	    
	    answerx = math.sqrt(abs(pre_ansx1))
	    answery = math.sqrt(abs(pre_ansy1))
	    
	    beta1 = abs(R11_f.item((0, 0))/answerx)
	    alpha1 = -R11_f.item((1, 0))/answerx
	    gamma1 = (1+alpha1**2)/beta1

	    beta2 = abs(R11_f.item((7, 0))/answery)
	    alpha2 = -R11_f.item((8, 0))/answery
	    gamma2 = (1+alpha2**2)/beta2
	    print("beta: %s."%beta1)
	    print("alpha: %s."%alpha1)
	    print("gamma: %s."%gamma1)

	    print("beta: %s."%beta2)
	    print("alpha: %s."%alpha2)
	    print("gamma: %s."%gamma2)


	    f11 = plt.Figure(figsize=(5,3), dpi=80,tight_layout=True)
	    f22 = plt.Figure(figsize=(5,3), dpi=80,tight_layout=True)

	    x1 = arange(-100,100,0.2)
	    x2 = arange(-100,100,0.2)
	    #x = arange(-100,100,0.2)


	    c1 = []
	    c2 = []
	    for i in range(len(x1)):
               k1 = [abs(beta1),2*alpha1*x1[i],gamma1*x1[i]**2-answerx]
               roots = np.roots(k1)
               c1.append(roots)

	    pos1 = []
	    neg1 = []
	    pos2 = []
	    neg2 = []

	    for i in range(len(c1)):
	         pos1.append(c1[i][0])
	         neg1.append(c1[i][1])
	    for i in range(len(pos1)):
                if np.imag(pos1[i]) == 0:
                   pos1[i] = pos1[i]
                else:
                   pos1[i] = None
	    for i in range(len(neg1)):
	        if np.imag(neg1[i]) == 0:
                   neg1[i] = neg1[i]
	        else:
                   neg1[i] =None

#---------------second------------------------------------------------
	    for i in range(len(x2)):
               k1 = [abs(beta2),2*alpha2*x2[i],gamma2*x2[i]**2-answery]
               roots = np.roots(k1)
               c2.append(roots)

	    for i in range(len(c2)):
	         pos2.append(c2[i][0])
	         neg2.append(c2[i][1])
	    for i in range(len(pos2)):
                if np.imag(pos2[i]) == 0:
                   pos2[i] = pos2[i]
                else:
                   pos2[i] = None

	    for i in range(len(neg2)):
	        if np.imag(neg2[i]) == 0:
                   neg2[i] = neg2[i]
	        else:
                   neg2[i] =None


	    a11 = f11.add_subplot(111)
	    a11.plot(x1,pos1,'*b')
	    a11.plot(x1,neg1,'*b')
	    a11.set_xlabel("X")
	    a11.set_ylabel("X'")
	    a11.set_title("Tranverse beam ellipses",fontsize=20)
	    
	    a22 = f22.add_subplot(111)  
	    a22.plot(x2,pos2,'*b')
	    a22.plot(x2,neg2,'*b')
	    a22.set_xlabel("Y")
	    a22.set_ylabel("Y'") 
	  
	    dataPlot11 = FigureCanvasTkAgg(f11, tab2)
	    dataPlot11.draw()
	    dataPlot11.get_tk_widget().grid(column=8,columnspan=9,row=0,rowspan=10,padx=50)

	    dataPlot22 = FigureCanvasTkAgg(f22, tab2)
	    dataPlot22.draw()
	    dataPlot22.get_tk_widget().grid(column=8,columnspan=9,row=10,rowspan=20,padx=50)


    except ValueError:
        messagebox.showinfo("Error message", "Invalid input")
    except TypeError:
        messagebox.showinfo("Error message", "Invalid input")
    except AttributeError:
        messagebox.showinfo("Error message", "Invalid input")
    except IndexError:
        messagebox.showinfo("Error message", "Invalid input")



f11 = plt.Figure(figsize=(5,3), dpi=80,tight_layout=True)  
f22 = plt.Figure(figsize=(5,3), dpi=80,tight_layout=True)
a11 = f11.add_subplot(111)
a11.set_xlabel("X")
a11.set_ylabel("X'")
a11.set_title("Tranverse beam ellipses",fontsize=20)
    
a22 = f22.add_subplot(111) 
a22.set_xlabel("Y")
a22.set_ylabel("Y'")

plt.tight_layout()
  
dataPlot11 = FigureCanvasTkAgg(f11, tab2)
dataPlot11.draw()
dataPlot11.get_tk_widget().grid(column=8,columnspan=9,row=0,rowspan=10,padx=50)

dataPlot22 = FigureCanvasTkAgg(f22, tab2)
dataPlot22.draw()
dataPlot22.get_tk_widget().grid(column=8,columnspan=9, row=10,rowspan=20,padx=50)


btn1 = ttk.Button(tab2,text="Calculate", command=Result_Sol).grid(column=1,row=17,padx=4,pady=(20,0))
btn2 = ttk.Button(tab2,text="Plot", command=Plot_Sol).grid(column=2,row=17,padx=4,pady=(20,0))

btn3 = ttk.Button(tab2,text='load',command=lambda:loadme2(L4Q_Cur0,L5Q_Cur0,L5Q_Cur1,L5Q_Cur2,L5Q_Cur3,L5Q_Cur4,L5Q_Cur5,
L5Q_Cur6,L5Q_Cur7,L5Q_Cur8,L5Q_Cur9,L5Q_Cur10,entry_x1, entry_x2, entry_x3, entry_x4, entry_x5, entry_x6, entry_x7, entry_x8
,entry_x9, entry_x10, entry_y1, entry_y2, entry_y3, entry_y4, entry_y5, entry_y6, entry_y7, entry_y8, entry_y9, entry_y10)).grid(column=0,row=17,padx=4,pady=(20,0))

btn4 = ttk.Button(tab2,text="Clear", command=Clear_Sol).grid(column=3,row=17,padx=4,pady=(20,0))
#--------------------------TAB 3-------------------------------

ttk.Label(tab3, text="Energy in keV").grid(column=0,row=0,pady=(20,0))
ttk.Label(tab3, text="Atomic mass").grid(column=0,row=1)
ttk.Label(tab3, text="Charge State").grid(column=0,row=2)
ttk.Label(tab3, text="Q5cur_0").grid(column=0,row=3)
ttk.Label(tab3, text="Q6cur_0").grid(column=0,row=4)
ttk.Label(tab3, text="Q6cur_1").grid(column=0,row=5,pady=(20,4))
ttk.Label(tab3, text="Q6cur_2").grid(column=0,row=6)
ttk.Label(tab3, text="Q6cur_3").grid(column=0,row=7)
ttk.Label(tab3, text="Q6cur_4").grid(column=0,row=8)
ttk.Label(tab3, text="Q6cur_5").grid(column=0,row=9,pady=(4,20))
ttk.Label(tab3, text="Q5cur_1").grid(column=0,row=10,pady=(20,4))
ttk.Label(tab3, text="Q5cur_2").grid(column=0,row=11)
ttk.Label(tab3, text="Q5cur_3").grid(column=0,row=12)
ttk.Label(tab3, text="Q5cur_4").grid(column=0,row=13)
ttk.Label(tab3, text="Q5cur_5").grid(column=0,row=14,pady=(4,20))

ttk.Label(tab3, text="Q1cur_0").grid(column=2,row=0,pady=(20,0))
ttk.Label(tab3, text="Q2cur_0").grid(column=2,row=1)
ttk.Label(tab3, text="Q3cur_0").grid(column=2,row=2)
ttk.Label(tab3, text="Q4cur_0").grid(column=2,row=3)



E_EK = ttk.Entry( tab3, width=10)
E_EK.grid(column=1,row=0, pady=(20,4))
E_M = ttk.Entry( tab3, width=10)
E_M.grid(column=1,row=1,pady=4)

E_Q1 = ttk.Entry( tab3, width=10)
E_Q1.grid(column=1,row=2,pady=4)

Q5_Cur_0 = ttk.Entry( tab3, width=10)
Q5_Cur_0.grid(column=1,row=3,pady=4)
Q6_Cur_0 = ttk.Entry( tab3, width=10)
Q6_Cur_0.grid(column=1,row=4,pady=4)

Q6_Cur_1 = ttk.Entry( tab3, width=10)
Q6_Cur_1.grid(column=1,row=5,pady=(20,4))
Q6_Cur_2 = ttk.Entry( tab3, width=10)
Q6_Cur_2.grid(column=1,row=6,pady=4)
Q6_Cur_3 = ttk.Entry( tab3, width=10)
Q6_Cur_3.grid(column=1,row=7,pady=4)
Q6_Cur_4 = ttk.Entry( tab3, width=10)
Q6_Cur_4.grid(column=1,row=8,pady=4)
Q6_Cur_5 = ttk.Entry( tab3, width=10)
Q6_Cur_5.grid(column=1,row=9,pady=(4,20))

Q5_Cur_1 = ttk.Entry( tab3, width=10)
Q5_Cur_1.grid(column=1,row=10,pady=(20,4))
Q5_Cur_2 = ttk.Entry( tab3, width=10)
Q5_Cur_2.grid(column=1,row=11,pady=4)
Q5_Cur_3 = ttk.Entry( tab3, width=10)
Q5_Cur_3.grid(column=1,row=12,pady=4)
Q5_Cur_4 = ttk.Entry( tab3, width=10)
Q5_Cur_4.grid(column=1,row=13,pady=4)
Q5_Cur_5 = ttk.Entry( tab3, width=10)
Q5_Cur_5.grid(column=1,row=14,pady=(4,20))

Q1_Cur_0 = ttk.Entry( tab3, width=10)
Q1_Cur_0.grid(column=3,row=0,pady=(20,4))
Q2_Cur_0 = ttk.Entry( tab3, width=10)
Q2_Cur_0.grid(column=3,row=1,pady=4)
Q3_Cur_0 = ttk.Entry( tab3, width=10)
Q3_Cur_0.grid(column=3,row=2,pady=4)
Q4_Cur_0 = ttk.Entry( tab3, width=10)
Q4_Cur_0.grid(column=3,row=3,pady=4)


ttk.Label( tab3, text="width_x1").grid(column=2,row=5,pady=(20,4))
ttk.Label( tab3, text="width_x2").grid(column=2,row=6)
ttk.Label( tab3, text="width_x3").grid(column=2,row=7)
ttk.Label( tab3, text="width_x4").grid(column=2,row=8)
ttk.Label( tab3, text="width_x5").grid(column=2,row=9,pady=(4,20))

ENTRY_x1 = ttk.Entry( tab3, width=10)
ENTRY_x1.grid(column=3,row=5,pady=(20,4))
ENTRY_x2 = ttk.Entry( tab3, width=10)
ENTRY_x2.grid(column=3,row=6,pady=4)
ENTRY_x3 = ttk.Entry( tab3, width=10)
ENTRY_x3.grid(column=3,row=7,pady=4)
ENTRY_x4 = ttk.Entry( tab3, width=10)
ENTRY_x4.grid(column=3,row=8,pady=4)
ENTRY_x5 = ttk.Entry( tab3, width=10)
ENTRY_x5.grid(column=3,row=9,pady=(4,20))


ttk.Label( tab3, text="width_y1").grid(column=2,row=10,pady=(20,4))
ttk.Label( tab3, text="width_y2").grid(column=2,row=11)
ttk.Label( tab3, text="width_y3").grid(column=2,row=12)
ttk.Label( tab3, text="width_y4").grid(column=2,row=13)
ttk.Label( tab3, text="width_y5").grid(column=2,row=14,pady=(4,20))


ENTRY_y1 = ttk.Entry( tab3, width=10)
ENTRY_y1.grid(column=3,row=10,pady=(20,4))
ENTRY_y2 = ttk.Entry( tab3, width=10)
ENTRY_y2.grid(column=3,row=11,pady=4)
ENTRY_y3 = ttk.Entry( tab3, width=10)
ENTRY_y3.grid(column=3,row=12,pady=4)
ENTRY_y4 = ttk.Entry( tab3, width=10)
ENTRY_y4.grid(column=3,row=13,pady=4)
ENTRY_y5 = ttk.Entry( tab3, width=10)
ENTRY_y5.grid(column=3,row=14,pady=(4,20))

xx3 = ttk.Label( tab3, text = "x emittance: ")
xx3.grid(row=17, column=0)

yy3 = ttk.Label( tab3, text = "y emittance: ")
yy3.grid(row=17, column=2)

def Constant_Quad1():
    Ek = float(E_EK.get())
    mass = float(E_M.get())
    charge =float(E_Q1.get())
    c = 299800000
    E_0 = 938.28*math.pow(10, 6)
    a = 0.056
    E_k = Ek*1000*charge
    B_r = (math.sqrt(E_k*E_k+2*mass*E_k*E_0))/(charge*c)
    cons = a*B_r
    return cons

def getMatrix_1 (B,cons,L):
       #b = 0.001858376948749*B+0.002381221835086
       l = 0.13
       k= math.sqrt(B/cons)
       
       quad1 = np.matrix([[math.cos(k*l),(1/k)*math.sin(k*l)],
                          [(-k)*math.sin(k*l),math.cos(k*l)]])
        
       drift1 = np.matrix([[1,L],[0,1]])
	    
       
       sigma1 = np.dot(drift1,quad1)
       return sigma1


def getMatrix_2 (B,cons,L):
       #b = 0.001858376948749*B+0.002381221835086
       l = 0.13
       k= math.sqrt(B/cons)
       
       quad2 = np.matrix([[math.cosh(k*l),(1/k)*math.sinh(k*l)],
                          [(k)*math.sinh(k*l),math.cosh(k*l)]])
        
       drift2 = np.matrix([[1,L],[0,1]])
	    
       
       sigma2 = np.dot(drift2,quad2)
       return sigma2

def Drift(L):
     drift = np.matrix([[1,L],[0,1]])
     return drift

def loadme3(Q1_Cur_0,Q2_Cur_0,Q3_Cur_0,Q4_Cur_0,Q5_Cur_0,Q6_Cur_0,Q6_Cur_1,Q6_Cur_2,Q6_Cur_3,Q6_Cur_4,Q6_Cur_5,
Q5_Cur_1,Q5_Cur_2,Q5_Cur_3,Q5_Cur_4,Q5_Cur_5,ENTRY_x1, ENTRY_x2, ENTRY_x3, ENTRY_x4, ENTRY_x5, ENTRY_y1, ENTRY_y2, ENTRY_y3, ENTRY_y4, ENTRY_y5):

    try:

	    filename = askopenfilename(filetypes=(("text file",".txt"),("All files","*.*"))) 
	    result = np.genfromtxt(filename)
	    
	    Q5_Cur_0.insert(10,result[0,1])
	    Q6_Cur_0.insert(10,result[0,4])

	    Q6_Cur_1.insert(10,result[0,0])
	    Q6_Cur_2.insert(10,result[1,0])
	    Q6_Cur_3.insert(10,result[2,0])
	    Q6_Cur_4.insert(10,result[3,0])
	    Q6_Cur_5.insert(10,result[4,0])

	    Q5_Cur_1.insert(10,result[0,5])
	    Q5_Cur_2.insert(10,result[1,5])
	    Q5_Cur_3.insert(10,result[2,5])
	    Q5_Cur_4.insert(10,result[3,5])
	    Q5_Cur_5.insert(10,result[4,5])

	    Q1_Cur_0.insert(10,result[0,2])
	    Q2_Cur_0.insert(10,result[0,6])
	    Q3_Cur_0.insert(10,result[0,2])
	    Q4_Cur_0.insert(10,result[0,2])
	 


	    ENTRY_x1.insert(10,result[0,3])
	    ENTRY_x2.insert(10,result[1,3])
	    ENTRY_x3.insert(10,result[2,3])
	    ENTRY_x4.insert(10,result[3,3])
	    ENTRY_x5.insert(10,result[4,3])
	   
	    ENTRY_y1.insert(10,result[0,7])
	    ENTRY_y2.insert(10,result[1,7])
	    ENTRY_y3.insert(10,result[2,7])
	    ENTRY_y4.insert(10,result[3,7])
	    ENTRY_y5.insert(10,result[4,7])
    except IndexError:
        messagebox.showinfo("Error message", "Invalid input")
    except TypeError:
        messagebox.showinfo("Error message", "Invalid input")
    except OSError:
        messagebox.showinfo("Error message", "Invalid input")



def Calculate_Quad1():

    Q5_B0 = float(Q5_Cur_0.get())
    Q6_B0 = float(Q6_Cur_0.get())
    Q6_B1 = float(Q6_Cur_1.get())
    Q6_B2 = float(Q6_Cur_2.get())
    Q6_B3 = float(Q6_Cur_3.get())
    Q6_B4 = float(Q6_Cur_4.get())
    Q6_B5 = float(Q6_Cur_5.get())
    Q5_B1 = float(Q5_Cur_1.get())
    Q5_B2 = float(Q5_Cur_2.get())
    Q5_B3 = float(Q5_Cur_3.get())
    Q5_B4 = float(Q5_Cur_4.get())
    Q5_B5 = float(Q5_Cur_5.get()) 

    Q1_B0 = float(Q1_Cur_0.get())
    Q2_B0 = float(Q2_Cur_0.get())
    Q3_B0 = float(Q3_Cur_0.get())
    Q4_B0 = float(Q4_Cur_0.get())  
    
   
    x1 = float(ENTRY_x1.get())
    x2 = float(ENTRY_x2.get())
    x3 = float(ENTRY_x3.get())
    x4 = float(ENTRY_x4.get())
    x5 = float(ENTRY_x5.get())

    y1 = float(ENTRY_y1.get())
    y2 = float(ENTRY_y2.get())
    y3 = float(ENTRY_y3.get())
    y4 = float(ENTRY_y4.get())
    y5 = float(ENTRY_y5.get())

    Sx_1 = getMatrix_1(Q6_B1, Constant_Quad1(),L1)*getMatrix_2(Q5_B0, Constant_Quad1(),L2)*getMatrix_1(Q4_B0, Constant_Quad1(),L2)*getMatrix_1(Q3_B0, Constant_Quad1(),L3)*getMatrix_2(Q2_B0, Constant_Quad1(),L2)*getMatrix_1(Q1_B0, Constant_Quad1(),L2)*Drift(L1)
    Sx_2 = getMatrix_1(Q6_B2, Constant_Quad1(),L1)*getMatrix_2(Q5_B0, Constant_Quad1(),L2)*getMatrix_1(Q4_B0, Constant_Quad1(),L2)*getMatrix_1(Q3_B0, Constant_Quad1(),L3)*getMatrix_2(Q2_B0, Constant_Quad1(),L2)*getMatrix_1(Q1_B0, Constant_Quad1(),L2)*Drift(L1)
    Sx_3 = getMatrix_1(Q6_B3, Constant_Quad1(),L1)*getMatrix_2(Q5_B0, Constant_Quad1(),L2)*getMatrix_1(Q4_B0, Constant_Quad1(),L2)*getMatrix_1(Q3_B0, Constant_Quad1(),L3)*getMatrix_2(Q2_B0, Constant_Quad1(),L2)*getMatrix_1(Q1_B0, Constant_Quad1(),L2)*Drift(L1)
    Sx_4 = getMatrix_1(Q6_B4, Constant_Quad1(),L1)*getMatrix_2(Q5_B0, Constant_Quad1(),L2)*getMatrix_1(Q4_B0, Constant_Quad1(),L2)*getMatrix_1(Q3_B0, Constant_Quad1(),L3)*getMatrix_2(Q2_B0, Constant_Quad1(),L2)*getMatrix_1(Q1_B0, Constant_Quad1(),L2)*Drift(L1)
    Sx_5 = getMatrix_1(Q6_B5, Constant_Quad1(),L1)*getMatrix_2(Q5_B0, Constant_Quad1(),L2)*getMatrix_1(Q4_B0, Constant_Quad1(),L2)*getMatrix_1(Q3_B0, Constant_Quad1(),L3)*getMatrix_2(Q2_B0, Constant_Quad1(),L2)*getMatrix_1(Q1_B0, Constant_Quad1(),L2)*Drift(L1)      
    

    Sy_1 = getMatrix_2(Q6_B0, Constant_Quad1(),L1)*getMatrix_1(Q5_B1, Constant_Quad1(),L2)*getMatrix_2(Q4_B0, Constant_Quad1(),L2)*getMatrix_2(Q3_B0, Constant_Quad1(),L3)*getMatrix_1(Q2_B0, Constant_Quad1(),L2)*getMatrix_2(Q1_B0, Constant_Quad1(),L2)*Drift(L1) 

    Sy_2 = getMatrix_2(Q6_B0, Constant_Quad1(),L1)*getMatrix_1(Q5_B2, Constant_Quad1(),L2)*getMatrix_2(Q4_B0, Constant_Quad1(),L2)*getMatrix_2(Q3_B0, Constant_Quad1(),L3)*getMatrix_1(Q2_B0, Constant_Quad1(),L2)*getMatrix_2(Q1_B0, Constant_Quad1(),L2)*Drift(L1) 

    Sy_3 = getMatrix_2(Q6_B0, Constant_Quad1(),L1)*getMatrix_1(Q5_B3, Constant_Quad1(),L2)*getMatrix_2(Q4_B0, Constant_Quad1(),L2)*getMatrix_2(Q3_B0, Constant_Quad1(),L3)*getMatrix_1(Q2_B0, Constant_Quad1(),L2)*getMatrix_2(Q1_B0, Constant_Quad1(),L2)*Drift(L1) 

    Sy_4 = getMatrix_2(Q6_B0, Constant_Quad1(),L1)*getMatrix_1(Q5_B4, Constant_Quad1(),L2)*getMatrix_2(Q4_B0, Constant_Quad1(),L2)*getMatrix_2(Q3_B0, Constant_Quad1(),L3)*getMatrix_1(Q2_B0, Constant_Quad1(),L2)*getMatrix_2(Q1_B0, Constant_Quad1(),L2)*Drift(L1) 

    Sy_5 = getMatrix_2(Q6_B0, Constant_Quad1(),L1)*getMatrix_1(Q5_B5, Constant_Quad1(),L2)*getMatrix_2(Q4_B0, Constant_Quad1(),L2)*getMatrix_2(Q3_B0, Constant_Quad1(),L3)*getMatrix_1(Q2_B0, Constant_Quad1(),L2)*getMatrix_2(Q1_B0, Constant_Quad1(),L2)*Drift(L1) 
 
    x_val = np.matrix([[x1**2],[x2**2],[x3**2],[x4**2],[x5**2]]) 
    y_val = np.matrix([[y1**2],[y2**2],[y3**2],[y4**2],[y5**2]])     

    Q1_xmatrix = np.matrix( [[math.pow(Sx_1.item(0, 0), 2), 2*Sx_1.item(0, 0)*Sx_1.item(0, 1), math.pow(Sx_1.item(0, 1),2)],
				   [math.pow(Sx_2.item(0, 0), 2) ,2*(Sx_2.item(0, 0))*Sx_2.item(0, 1),math.pow(Sx_2.item(0, 1),2)],
                                   [math.pow(Sx_3.item(0, 0), 2) ,2*(Sx_3.item(0, 0))*Sx_3.item(0, 1),math.pow(Sx_3.item(0, 1),2)],
                                   [math.pow(Sx_4.item(0, 0), 2) ,2*(Sx_4.item(0, 0))*Sx_4.item(0, 1),math.pow(Sx_4.item(0, 1),2)],
                                   [math.pow(Sx_5.item(0, 0), 2) ,2*(Sx_5.item(0, 0))*Sx_5.item(0, 1),math.pow(Sx_5.item(0, 1),2)]])
				   
	
    Q2_ymatrix = np.matrix( [[math.pow(Sy_1.item(0, 0), 2), 2*Sy_1.item(0, 0)*Sy_1.item(0, 1), math.pow(Sy_1.item(0, 1),2)],
				   [math.pow(Sy_2.item(0, 0), 2), 2*Sy_2.item(0, 0)*Sy_2.item(0, 1), math.pow(Sy_2.item(0, 1),2)],
				   [math.pow(Sy_3.item(0, 0), 2), 2*Sy_3.item(0, 0)*Sy_3.item(0, 1), math.pow(Sy_3.item(0, 1),2)],
                                   [math.pow(Sy_4.item(0, 0), 2), 2*Sy_4.item(0, 0)*Sy_4.item(0, 1), math.pow(Sy_4.item(0, 1),2)],
                                   [math.pow(Sy_5.item(0, 0), 2), 2*Sy_5.item(0, 0)*Sy_5.item(0, 1), math.pow(Sy_5.item(0, 1),2)]])



    Q1_Xmatrix = Q1_xmatrix.I
    Q2_Ymatrix = Q2_ymatrix.I
    
    Result1 = np.dot(Q1_Xmatrix,x_val)
    Result2 = np.dot(Q2_Ymatrix,y_val)
    return Result1,Result2

def Clear_Quad1():
    
    E_EK.delete(first=0,last=22)
    E_M.delete(first=0,last=22)
    E_Q1.delete(first=0,last=22)
    Q5_Cur_0.delete(first=0,last=22)
    Q6_Cur_0.delete(first=0,last=22)
    Q6_Cur_1.delete(first=0,last=22)
    Q6_Cur_2.delete(first=0,last=22)
    Q6_Cur_3.delete(first=0,last=22)
    Q6_Cur_4.delete(first=0,last=22)
    Q6_Cur_5.delete(first=0,last=22)
    Q5_Cur_1.delete(first=0,last=22)
    Q5_Cur_2.delete(first=0,last=22)
    Q5_Cur_3.delete(first=0,last=22)
    Q5_Cur_4.delete(first=0,last=22)
    Q5_Cur_5.delete(first=0,last=22) 

    Q1_Cur_0.delete(first=0,last=22)
    Q2_Cur_0.delete(first=0,last=22)
    Q3_Cur_0.delete(first=0,last=22)
    Q4_Cur_0.delete(first=0,last=22)
   
    ENTRY_x1.delete(first=0,last=22)
    ENTRY_x2.delete(first=0,last=22)
    ENTRY_x3.delete(first=0,last=22)
    ENTRY_x4.delete(first=0,last=22)
    ENTRY_x5.delete(first=0,last=22)

    ENTRY_y1.delete(first=0,last=22)
    ENTRY_y2.delete(first=0,last=22)
    ENTRY_y3.delete(first=0,last=22)
    ENTRY_y4.delete(first=0,last=22)
    ENTRY_y5.delete(first=0,last=22)
    
    xx3.config(text = "x emittance: ")
    yy3.config(text = "y emittance: ")

    dataPlot111 = FigureCanvasTkAgg(f_1, tab3)
    dataPlot111.draw()
    dataPlot111.get_tk_widget().grid(column=8,columnspan=9,row=0,rowspan=10,padx=50)
    dataPlot222 = FigureCanvasTkAgg(f_2, tab3)
    dataPlot222.draw()
    dataPlot222.get_tk_widget().grid(column=8,columnspan=9,row=9,rowspan=14,padx=50)


def Result_Quad1():
    try:
	    R1_f,R2_f = Calculate_Quad1()

	    B1_final  = [[R1_f[0, 0], R1_f[1, 0]],[R1_f[1, 0], R1_f[2, 0]]]
	    B2_final  = [[R2_f[0, 0], R2_f[1, 0]],[R2_f[1, 0], R2_f[2, 0]]]
	     
	    pre_ansx = np.linalg.det(B1_final)
	    pre_ansy = np.linalg.det(B2_final)
	    
	    answerx = math.sqrt(abs(pre_ansx))
	    answery = math.sqrt(abs(pre_ansy))
	    
	    answerx = np.around(answerx,2)
	    answery = np.around(answery,2)

	    xx3.config(text = "x emittance: %s."%answerx)
	    yy3.config(text = "y emittance: %s."%answery)
            
	    print("sigma11 : %s."%R1_f[0, 0])
	    print("sigma22 : %s."%R1_f[2, 0])
	    print("sigma33 : %s."%R2_f[0, 0])
	    print("sigma44 : %s."%R2_f[2, 0])

    except ValueError:
        messagebox.showinfo("Error message", "Invalid input")
    except TypeError:
        messagebox.showinfo("Error message", "Invalid input")
    except AttributeError:
        messagebox.showinfo("Error message", "Invalid input")
  
def Plot_Quad1():
    try:

	    R11_f,R22_f = Calculate_Quad1()

	    B11_final  = [[R11_f[0, 0], R11_f[1, 0]],[R11_f[1, 0], R11_f[2, 0]]]
	    B22_final  = [[R22_f[0, 0], R22_f[1, 0]],[R22_f[1, 0], R22_f[2, 0]]]
	     
	    pre_ansx11 = np.linalg.det(B11_final)
	    pre_ansy11 = np.linalg.det(B22_final)
	    
	    answerx1 = math.sqrt(abs(pre_ansx11))
	    answery1 = math.sqrt(abs(pre_ansy11))
	    
	    beta1 = R11_f[0, 0]/answerx1
	    alpha1 = -R11_f[1, 0]/answerx1
	    gamma1 = R11_f[2, 0]/answerx1

	    beta2 = R22_f[0, 0]/answery1
	    alpha2 = -R22_f[1, 0]/answery1
	    gamma2 = R22_f[2, 0]/answery1
	    
	    
	    f_1 = plt.Figure(figsize=(5,3), dpi=80,tight_layout=True) 
	    f_2 = plt.Figure(figsize=(5,3), dpi=80,tight_layout=True)
	 
	    c1 = []
	    c2 = []

	    x1 = arange(-100,100,0.2)
	    x2 = arange(-100,100,0.2)

	    for i in range(len(x1)):
               k1 = [abs(beta1),2*alpha1*x1[i],gamma1*x1[i]**2-answerx1]
               roots = np.roots(k1)
               c1.append(roots)

	    pos1 = []
	    neg1 = []
	    pos2 = []
	    neg2 = []

	    for i in range(len(c1)):
	         pos1.append(c1[i][0])
	         neg1.append(c1[i][1])
	    for i in range(len(pos1)):
                if np.imag(pos1[i]) == 0:
                   pos1[i] = pos1[i]
                else:
                   pos1[i] = None
	    for i in range(len(neg1)):
	        if np.imag(neg1[i]) == 0:
                   neg1[i] = neg1[i]
	        else:
                   neg1[i] =None

#---------------second------------------------------------------------
	    for i in range(len(x2)):
               k1 = [abs(beta2),2*alpha2*x2[i],gamma2*x2[i]**2-answery1]
               roots = np.roots(k1)
               c2.append(roots)

	    for i in range(len(c2)):
	         pos2.append(c2[i][0])
	         neg2.append(c2[i][1])
	    for i in range(len(pos2)):
                if np.imag(pos2[i]) == 0:
                   pos2[i] = pos2[i]
                else:
                   pos2[i] = None

	    for i in range(len(neg2)):
	        if np.imag(neg2[i]) == 0:
                   neg2[i] = neg2[i]
	        else:
                   neg2[i] =None

	    a_1 = f_1.add_subplot(111)
	    a_1.plot(x1,pos1,'*b')
	    a_1.plot(x1,neg1,'*b')
	    a_1.set_xlabel("X")
	    a_1.set_ylabel("X'")
	    a_1.set_title("Tranverse beam ellipses",fontsize=20)
	    
	    a_2 = f_2.add_subplot(111)  
	    a_2.plot(x2,pos2,'*b')
	    a_2.plot(x2,neg2,'*b')
	    a_2.set_xlabel("Y")
	    a_2.set_ylabel("Y'") 
	   
	    dataPlot_1 = FigureCanvasTkAgg(f_1, tab3)
	    dataPlot_1.draw()
	    dataPlot_1.get_tk_widget().grid(column=8,columnspan=9,row=0,rowspan=10,padx=50)
	    dataPlot_2 = FigureCanvasTkAgg(f_2, tab3)
	    dataPlot_2.draw()
	    dataPlot_2.get_tk_widget().grid(column=8,columnspan=9,row=10,rowspan=20,padx=50)

    except ValueError:
        messagebox.showinfo("Error message", "Invalid input")
    except TypeError:
        messagebox.showinfo("Error message", "Invalid input")
    except AttributeError:
        messagebox.showinfo("Error message", "Invalid input")

f_1 = plt.Figure(figsize=(5,3), dpi=80,tight_layout=True) 
f_2 = plt.Figure(figsize=(5,3), dpi=80,tight_layout=True)
a_1 = f_1.add_subplot(111)    
a_2 = f_2.add_subplot(111) 
a_1.set_title("Tranverse beam ellipses",fontsize=20)

a_1.set_xlabel("X")
a_1.set_ylabel("X'")

a_2.set_xlabel("Y")
a_2.set_ylabel("Y'")
      
dataPlot_1 = FigureCanvasTkAgg(f_1, tab3)
dataPlot_1.draw()
dataPlot_1.get_tk_widget().grid(column=8,columnspan=9,row=0,rowspan=10,padx=50)

dataPlot_2 = FigureCanvasTkAgg(f_2, tab3)
dataPlot_2.draw()
dataPlot_2.get_tk_widget().grid(column=8,columnspan=9,row=10,rowspan=20,padx=50)

BTN_1 = ttk.Button(tab3, text="Calculate",command=Result_Quad1).grid(column=1,row=16,padx=4)
BTN_2 = ttk.Button(tab3, text="Plot",command=Plot_Quad1).grid(column=2,row=16,padx=4)
BTN_3 = ttk.Button(tab3, text='load',command=lambda:loadme3(Q1_Cur_0,Q2_Cur_0,Q3_Cur_0,Q4_Cur_0,Q5_Cur_0,Q6_Cur_0,Q6_Cur_1,Q6_Cur_2,Q6_Cur_3,Q6_Cur_4,Q6_Cur_5,
Q5_Cur_1,Q5_Cur_2,Q5_Cur_3,Q5_Cur_4,Q5_Cur_5,ENTRY_x1, ENTRY_x2, ENTRY_x3, ENTRY_x4, ENTRY_x5, ENTRY_y1, ENTRY_y2, ENTRY_y3, ENTRY_y4, ENTRY_y5)).grid(column=0,row=16,padx=4)

BTN_4 = ttk.Button(tab3, text="Clear", command=Clear_Quad1).grid(column=3,row=16,padx=4)

#Calling Main()  
win.mainloop()  
