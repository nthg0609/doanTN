import os
import pandas as pd
import openpyxl

excel_file = "d:/DoAn_DaLieu/5_Results/archive/FINAL_REPORT_v001_20260404_015555.xlsx"
if not os.path.exists(excel_file):
    print("File not found")
else:
    xls = pd.ExcelFile(excel_file)
    print("Sheets in xlsx:", xls.sheet_names)
    for sheet in xls.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet)
        print(f"\n--- Sheet: {sheet} ---")
        print(df.head(20))
