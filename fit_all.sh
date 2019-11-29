#!/bin/bash


#====================Solenoid==========
#----------folder1----------

#cd /home/ndumiso/Documents/silicon-8.35-kev1

#for i in 1 2 3 4 5
#do
#   root -b -l -q *$i.C getbin.C ;
#done


#paste Wednesday-14-16-45.txt values1.txt >> temp.txt

#awk '{ print $2" "$1" "$6" "$7}' temp.txt>> /home/ndumiso/Desktop/emit_python_calc/mes_sol.txt

#----------folder2------------

#cd /home/ndumiso/Documents/silicon-8.35-kev2

#for i in 1 2 3 4 5
#do
#   root -b -l -q *$i.C getbin.C ;
#done


#paste Wednesday-14-24-12.txt values1.txt >> temp.txt

#awk '{ print $2" "$1" "$6" "$7}' temp.txt>>/home/ndumiso/Desktop/emit_python_calc/mes_sol.txt


#================Quadrupole==============
#-----------folder_x-----------

cd /home/ndumiso/Documents/silicon-8.35-kev3x

for i in 1 2 3 4 5
do
   root -b -l -q *$i.C getbin.C ;
done

paste Wednesday-14-55-35x.txt values1.txt >> temp.txt

awk '{ print $1" "$2" "$6}' temp.txt>> /home/ndumiso/Desktop/emit_python_calc/temp5.txt

#----------folder_y--------------

cd /home/ndumiso/Documents/silicon-8.35-kev2y

for i in 1 2 3 4 5
do
   root -b -l -q *$i.C getbin.C ;
done

paste Wednesday-15-12-09y.txt values1.txt >> temp.txt

awk '{ print $1" "$2" "$7}' temp.txt>> /home/ndumiso/Desktop/emit_python_calc/temp6.txt

cd /home/ndumiso/Desktop/emit_python_calc
 
paste temp5.txt temp6.txt >> mes_quad_2.txt





