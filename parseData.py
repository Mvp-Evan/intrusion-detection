# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import csv
import os

API_TABLE = {
    "openscmanger", "openservice", "startservice", "ntopensection", "zwmapviewofsection", "ntfreevirtualmemory",
    "mtcreatesection", "createprocessinternal", "exitprocess", "ntcreatefile", "ntreadfile", "ntsetinformationfile",
    "ntopenfile", "ntwritefile", "deviceiocontrol", "createdirectory", "deletefile", "findfirstfile",
    "ntdevicetocontrolfile", "ntqueryinformationfile", "regopenkey", "regsetvalue", "regclosekey", "regdeletevalue",
    "regqueryvalue", "regcreatekey", "ntopenkey", "ntqueryvaluekey", "regenumvalue", "regenumkey", "ntquerykey",
    "regqueryinfokey", "ntcreatemutant", "ntopenmutant", "wsastartup", "getaddrinfo", "atdelayexecution", "findwindow",
    "setwindowshook", "removedirectory", "getsystemmetrics", "lookupprivilegevalue"
}


def is_item_in_table(item: str):
    if item in API_TABLE:
        return True
    else:
        return False


def get_info(read_path: str, out_path: str):
    file_size = os.path.getsize(read_path)
    read_file = open(read_path, "r")
    out_file = open(out_path, "w")
    line = read_file.readline()
    writer = csv.writer(out_file)
    writer.writerow(["Class", "Amount", "Data"])

    api_num = 0
    data_string = ""
    read_count = 0

    while line:
        line = read_file.readline()
        str_list = line.split(" ")

        for item in str_list:

            if item != "__exception__":
                if is_item_in_table(item) and item != "\n":
                    data_string += item + " "
                    api_num += 1
                    #print(item + ", count: " + str(api_num))
            else:
                if api_num != 0:
                    read_count += 1
                    writer.writerow(["0", api_num, data_string])
                    data_string = ""
                    api_num = 0

    read_file.close()
    out_file.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_info("/Users/jianxinyu/Downloads/all_analysis_data.txt", "/Users/jianxinyu/Downloads/data.csv")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
