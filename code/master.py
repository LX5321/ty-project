import itertools
import mysql.connector
from colorama import init
from colorama import Fore, Back, Style
from mysql.connector import Error
import masterenvironment as en
from os import system
from os import chdir
init()

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def readNodes():
    global lineList
    with open(en.fileName) as f:
        lineList = f.readlines()
    lineList = [line.rstrip('\n') for line in open(en.fileName)]


def db_connect():
    curr_node = 0
    curr_chunk = 0
    print("Connecting to DB ", end="")
    try:
        mydb = mysql.connector.connect(
            host=en.host, user=en.user, passwd=en.passwd, database=en.database)
        mycursor = mydb.cursor()
        print(Fore.GREEN+"[SUCCESS]"+Style.RESET_ALL)
        pending = []
        query = "select * from {} where PredictedOutcome {}".format("diagnosis", "IS NULL")
        mycursor.execute(query)
        myresult = mycursor.fetchall()
        for x in myresult:
            pending.append(x[0])

        x = list(divide_chunks(pending, en.chunkSize))
        executeStatus = 0
        nodeCount = len(lineList)
        nodeCount = nodeCount - 1
        chunkCount = len(x)
        while(executeStatus != 1):
            temp = x[curr_chunk]
            temp = str(temp)
            temp = temp[1:-1]
            temp = temp.replace(" ", "")
            query = "ssh pi@{} python3 hive-ml/slave.py {}".format(lineList[curr_node], temp)
            query = str(query)
            system(query)
            curr_chunk = curr_chunk + 1
            if(curr_node == nodeCount):
                curr_node = 0
            else:
                curr_node = curr_node+1
            if(curr_chunk == chunkCount):
                executeStatus = 1

    except Error as e:
        print(Fore.RED+"[FAILED]"+Style.RESET_ALL)
        print(e)
        exit(0)

if __name__ == "__main__":
    readNodes()
    db_connect()
