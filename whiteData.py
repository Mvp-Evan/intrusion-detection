import random
import csv


API_TABLE = [
    "ntcreatefile", "ntreadfile", "ntsetinformationfile", "ntopenfile", "ntwritefile", "deviceiocontrol",
    "createdirectory", "findfirstfile", "ntdevicetocontrolfile", "ntqueryinformationfile", "regopenkey",
    "ntopenkey", "atdelayexecution", "findwindow", "openservice"]

if __name__ == '__main__':
    data = ""
    out_file = open("/Users/jianxinyu/Downloads/data.csv", "a+")
    writer = csv.writer(out_file)
    api_num = 0

    for i in range(342):

        while True:
            stop = random.randint(0, 9)

            if stop != 0:
                api_num += 1
                position = random.randint(0, len(API_TABLE) - 1)
                data += API_TABLE[position] + " "

            else:
                if data != "":
                    writer.writerow(["1", api_num, data])
                    api_num = 0
                    data = ""
                    break
