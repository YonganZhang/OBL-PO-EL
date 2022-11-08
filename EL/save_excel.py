import xlsxwriter

def write_excel_xls(path, sheet_name, value):
    index = len(value)
    workbook = xlsxwriter.Workbook(path)
    sheet = workbook.add_worksheet(sheet_name)
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(j, 2*i, value[i][j][0])
            sheet.write(j, 2*i+1, value[i][j][1])

    workbook.close()
