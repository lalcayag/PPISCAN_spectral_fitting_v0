# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 23:32:44 2019

@author: lalc
"""

# In[MySQL]
import mysql.connector

mydb = mysql.connector.connect(
  host="ri-veadbs04",
  user="lalc",
#  database = "oesterild_light_masts",
  passwd="La469lc"
)

cursor = mydb.cursor()
cursor.execute("show databases")
for (databases) in cursor:
     print(databases)
     
     
mydb2 = mysql.connector.connect(
  host="ri-veadbs03",
  user="lalc",
#  database = "oesterild_light_masts",
  passwd="La469lc"
)

cursor = mydb.cursor()
cursor.execute("show databases")
for (databases) in cursor:
     print(databases)     
     
sql_select_Query = "select * from "
cursor.execute(sql_select_Query)
records = cursor.fetchall()    
cursor.execute("SHOW TABLES") 
for (table_name,) in cursor:
        print(table_name)
        
     
     
cursor.execute("desc caldata_2016_01_20hz")
col = [column[0] for column in cursor.fetchall()]     